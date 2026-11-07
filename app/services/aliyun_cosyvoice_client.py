"""Alibaba Cloud Model Studio CosyVoice v3.5 adapter.

The DashScope SDK is synchronous and callback-based. This module keeps that
API behind the backend's existing async TTS contract and never exposes the
provider API key to mobile clients.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.audio_utils import pcm_to_wav, wav_duration_ms
from app.services.tts_provider_service import TtsRuntimeConfig, tts_provider_service
from sqlalchemy import text

logger = logging.getLogger(__name__)


class AliyunCosyVoiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class AliyunCosyVoiceAudio:
    wav_bytes: bytes
    content_type: str
    duration_ms: int
    used_reference: bool


@dataclass(frozen=True)
class AliyunCosyVoiceStreamChunk:
    pcm_bytes: bytes
    sample_rate: int
    channels: int
    format: str
    duration_ms: int


_EMOTION_INSTRUCTIONS = {
    "neutral": "请用自然、清晰、平稳的语气表达。",
    "warm": "请用温暖、亲切、关心的语气表达。",
    "happy": "请用开心、轻快、有活力的语气表达。",
    "sad": "请用悲伤、低缓、克制的语气表达。",
    "surprised": "请用惊讶、语调略微上扬的语气表达。",
    "angry": "请用生气但不过度喊叫、坚定有力的语气表达。",
    "fear": "请用紧张、担忧、略带不安的语气表达。",
}


_VOICE_INSTRUCTION_CACHE: dict[str, tuple[float, str | None]] = {}
_VOICE_INSTRUCTION_TTL = 60.0


def _voice_instruction(voice: str, emotion: str | None) -> str:
    """席位级指令优先（preset_voices.tts_instruction）——方言/老年口音保真的
    唯一正解：克隆只保音色不保口音，口音靠合成时指令（2026-11-05 统一试听
    暴露：纯普通话文本直读方言席位=普通话）。席位指令自带语气描述，命中时
    覆盖情绪指令；普通话席位回落情绪指令（原行为）。矩阵远端 voice_id 经
    voice_account_map 反查；查库失败静默回落。"""
    import time as _time

    now = _time.monotonic()
    cached = _VOICE_INSTRUCTION_CACHE.get(voice)
    if cached is not None and now - cached[0] < _VOICE_INSTRUCTION_TTL:
        inst = cached[1]
    else:
        inst = None
        try:
            from app.core.database import SessionLocal

            with SessionLocal() as db:
                row = db.execute(text(
                    "select tts_instruction from preset_voices where voice_id = :v"
                ), {"v": voice}).fetchone()
                if not row:
                    row = db.execute(text(
                        "select p.tts_instruction from preset_voices p"
                        " join voice_account_map m"
                        "   on m.local_voice_key = 'preset:' || p.display_name"
                        " where m.remote_voice_id = :v limit 1"
                    ), {"v": voice}).fetchone()
                inst = row[0] if row else None
        except Exception:
            inst = None
        _VOICE_INSTRUCTION_CACHE[voice] = (now, inst)
    return inst if inst else _EMOTION_INSTRUCTIONS.get(emotion or "neutral")


def _speech_rate_param() -> float | None:
    """朗读语速参数（重整方案 D1.3.2）。

    - 默认 1.0 时**不传参**：与现网行为完全一致，部署零风险；
    - 仅当配置偏离默认且落在官方合法域 0.5–2.0 内才注入 speech_rate；
    - 音调（pitch_rate）与采样率不动——禁止用改采样率元数据/丢采样
      "提速"（D1.3.3）。
    线上实际模型以 TtsRuntimeConfig（数据库覆盖，source=database）为准，
    当前为 cosyvoice-v2 + 克隆音色（settings 默认 v3.5-flash 并非生效值）。
    2026-09-25 容器内实测（synthesize_stream，克隆音色）：1.1 档时长 -9%，
    1.2 档 -17%，引擎接受、无报错。
    """
    try:
        rate = float(settings.tts_speech_rate)
    except (TypeError, ValueError):
        return None
    if rate < 0.5 or rate > 2.0:
        logger.warning("tts_speech_rate=%r out of range [0.5, 2.0], ignoring", rate)
        return None
    if abs(rate - 1.0) < 1e-6:
        return None
    return rate


def _load_sdk() -> tuple[Any, Any, Any, Any, Any]:
    """Import lazily so other providers can run without initializing DashScope."""
    try:
        import dashscope
        from dashscope.audio.tts_v2 import (
            AudioFormat,
            ResultCallback,
            SpeechSynthesizer,
            VoiceEnrollmentService,
        )
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise AliyunCosyVoiceError(
            "dashscope is not installed; install backend dependencies"
        ) from exc
    return dashscope, AudioFormat, ResultCallback, SpeechSynthesizer, VoiceEnrollmentService


class AliyunCosyVoiceClient:
    provider = "aliyun_cosyvoice"

    @property
    def sample_rate(self) -> int:
        # 音质下限 44100：22.05k 输出带宽 ~11kHz，高频被切产生"空/大喇叭"听感
        # （2026-11-04 实测：44.1k 的 9kHz 以上频段能量 +12.5dB RMS）。环境变量
        # ALIYUN_COSYVOICE_STREAM_SAMPLE_RATE 停留在 22050（容器 env 无法热改），
        # 在此取 max 强制提升；SDK 与客户端（AudioTrack 按 reply_start 动态建轨）
        # 均已验证支持 44100。如需回退改此下限。
        return max(44100, int(settings.aliyun_cosyvoice_stream_sample_rate))

    @staticmethod
    def voice_prefix(user_id: str) -> str:
        return "dh" + hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:8]

    # ---- 账号矩阵辅助：voice_id → 本地音色键 → 各账号远端 voice_id ----

    @staticmethod
    def _local_voice_key(voice_id: str | None) -> str | None:
        """主账号 voice_id 反查本地键：preset:<名> / dh:<id>；未识别返回 None
        （含默认音色——仅主账号持有，不参与矩阵切换）。"""
        if not voice_id:
            return None
        from app.core.database import SessionLocal

        with SessionLocal() as db:
            row = db.execute(text(
                "select display_name from preset_voices where voice_id = :v"
            ), {"v": voice_id}).fetchone()
            if row:
                return f"preset:{row[0]}"
            row = db.execute(text(
                "select digital_human_id from digital_humans"
                " where voice_provider_voice_id = :v"
            ), {"v": voice_id}).fetchone()
            if row:
                return f"dh:{row[0]}"
        return None

    @staticmethod
    def _remote_voice(local_key: str | None, account_id: str | None, fallback: str) -> str:
        """矩阵账号取映射的远端 voice_id；主账号（None）用原 id。无映射返回空。"""
        if not account_id or not local_key:
            return fallback
        from app.core.database import SessionLocal

        with SessionLocal() as db:
            row = db.execute(text(
                "select remote_voice_id from voice_account_map"
                " where account_id = :a and local_voice_key = :k"
            ), {"a": account_id, "k": local_key}).fetchone()
        return row[0] if row else ""

    def _runtime(self) -> TtsRuntimeConfig:
        runtime = tts_provider_service.runtime()
        if runtime.type != self.provider:
            raise AliyunCosyVoiceError("active TTS provider is not aliyun_cosyvoice")
        return runtime

    @staticmethod
    def _model(runtime: TtsRuntimeConfig) -> str:
        return runtime.model or settings.aliyun_cosyvoice_model

    def _endpoints(self, runtime: TtsRuntimeConfig | None = None) -> tuple[str, str]:
        runtime = runtime or self._runtime()
        if runtime.endpoint:
            http_url = runtime.endpoint.rstrip("/")
            websocket_url = http_url.replace("https://", "wss://", 1)
            websocket_url = websocket_url.replace("http://", "ws://", 1)
            websocket_url = websocket_url.replace("/api/v1", "/api-ws/v1/inference")
            return http_url, websocket_url
        workspace = (runtime.workspace_id or "").strip()
        if workspace:
            return (
                f"https://{workspace}.cn-beijing.maas.aliyuncs.com/api/v1",
                f"wss://{workspace}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
            )
        # Alibaba Cloud documents these legacy endpoints as still supported.
        return (
            "https://dashscope.aliyuncs.com/api/v1",
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        )

    def _configure_sdk(
        self, runtime: TtsRuntimeConfig | None = None
    ) -> tuple[Any, Any, Any, Any, Any]:
        runtime = runtime or self._runtime()
        api_key = (runtime.api_key or "").strip()
        if not api_key:
            raise AliyunCosyVoiceError("DASHSCOPE_API_KEY is not configured")
        sdk = _load_sdk()
        dashscope = sdk[0]
        http_url, websocket_url = self._endpoints(runtime)
        dashscope.api_key = api_key
        dashscope.base_http_api_url = http_url
        dashscope.base_websocket_api_url = websocket_url
        return sdk

    def _resolve_voice(
        self, voice_id: str | None, runtime: TtsRuntimeConfig | None = None
    ) -> str:
        runtime = runtime or self._runtime()
        resolved = (voice_id or runtime.default_voice_id or "").strip()
        if not resolved:
            raise AliyunCosyVoiceError(
                "no cloned voice is bound and ALIYUN_COSYVOICE_DEFAULT_VOICE_ID is empty"
            )
        return resolved

    @staticmethod
    def _pcm_format(audio_format: Any, sample_rate: int) -> Any:
        attribute = {
            16000: "PCM_16000HZ_MONO_16BIT",
            22050: "PCM_22050HZ_MONO_16BIT",
            24000: "PCM_24000HZ_MONO_16BIT",
            44100: "PCM_44100HZ_MONO_16BIT",
            48000: "PCM_48000HZ_MONO_16BIT",
        }.get(int(sample_rate))
        if attribute is None:
            raise AliyunCosyVoiceError(
                f"unsupported CosyVoice PCM sample rate: {sample_rate}"
            )
        value = getattr(audio_format, attribute, None)
        if value is None:
            raise AliyunCosyVoiceError(
                f"installed DashScope SDK does not expose {attribute}"
            )
        return value

    async def health_check(self) -> dict[str, Any]:
        runtime = self._runtime()
        http_url, _ = self._endpoints(runtime)
        try:
            sdk = self._configure_sdk(runtime)

            def _query() -> None:
                service = sdk[4]()
                service.list_voices(page_index=0, page_size=1)

            await asyncio.wait_for(
                asyncio.to_thread(_query),
                timeout=float(settings.aliyun_cosyvoice_timeout_seconds),
            )
            return {
                "status": "ok",
                "provider": self.provider,
                "endpoint": http_url,
                "model": self._model(runtime),
            }
        except Exception as exc:
            return {
                "status": "error",
                "provider": self.provider,
                "endpoint": http_url,
                "model": self._model(runtime),
                "error": str(exc),
            }

    async def create_voice(
        self,
        *,
        audio_url: str,
        user_id: str,
        duration_ms: int | None = None,
    ) -> str:
        runtime = self._runtime()
        parsed = urlparse(audio_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise AliyunCosyVoiceError("voice enrollment requires a public HTTP(S) audio URL")
        if parsed.hostname in {"minio", "localhost", "127.0.0.1", "::1"}:
            raise AliyunCosyVoiceError(
                "voice enrollment URL points to an internal host; configure "
                "STORAGE_PUBLIC_ENDPOINT_URL"
            )

        sdk = self._configure_sdk(runtime)
        prompt_seconds = float(settings.aliyun_cosyvoice_max_prompt_audio_length)
        if duration_ms is not None:
            prompt_seconds = min(prompt_seconds, max(3.0, duration_ms / 1000))

        def _create() -> str:
            service = sdk[4]()
            return service.create_voice(
                target_model=self._model(runtime),
                prefix=self.voice_prefix(user_id),
                url=audio_url,
                language_hints=["zh"],
                max_prompt_audio_length=min(30.0, max(3.0, prompt_seconds)),
                enable_preprocess=settings.aliyun_cosyvoice_enable_preprocess,
            )

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_create),
                timeout=float(settings.aliyun_cosyvoice_timeout_seconds),
            )
        except Exception as exc:
            raise AliyunCosyVoiceError(f"voice enrollment failed: {exc}") from exc

    def delete_voice_sync(self, voice_id: str) -> None:
        """Delete a cloud voice during the durable erasure workflow."""
        runtime = tts_provider_service.runtime_for(self.provider)
        sdk = self._configure_sdk(runtime)
        try:
            sdk[4]().delete_voice(voice_id)
        except Exception as exc:
            message = str(exc).lower()
            if any(marker in message for marker in ("not found", "does not exist", "404")):
                return
            raise AliyunCosyVoiceError(f"voice deletion failed: {exc}") from exc

    async def synthesize_batch(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        voice_id: str | None = None,
        emotion: str | None = None,
        user_id: str | None = None,
        format_: str = "wav",
        output_sample_rate: int | None = None,
    ) -> AliyunCosyVoiceAudio:
        del reference_audio_bytes, reference_text, user_id, format_
        runtime = self._runtime()
        sdk = self._configure_sdk(runtime)
        sample_rate = int(output_sample_rate or self.sample_rate)
        audio_format = self._pcm_format(sdk[1], sample_rate)
        model_name = self._model(runtime)

        # 账号矩阵：预设/克隆音色走多 key 故障转移（429 换桶，前端无感知）；
        # 未注册到矩阵的音色（含默认音色）只在主账号存在，local_key=None
        # 时 with_failover 直接走主账号（现役行为）。
        local_key = self._local_voice_key(voice_id)
        main_voice = self._resolve_voice(voice_id, runtime)

        def _synth_kwargs(voice: str) -> dict:
            kw = dict(
                model=model_name,
                voice=voice,
                format=audio_format,
                language_hints=["zh"],
                volume=settings.aliyun_cosyvoice_volume,
            )
            speech_rate = _speech_rate_param()
            if speech_rate is not None:
                kw["speech_rate"] = speech_rate
            # instruction 是 v3+ 参数：v2 模型传了必报引擎 428（线上
            # cosyvoice-v2 + 克隆音色曾因此整路径不可用，2026-09-25 冒烟发现）
            if "v3" in model_name or "v4" in model_name:
                kw["instruction"] = _voice_instruction(voice, emotion)
            return kw

        async def _factory(account_id, api_key):
            voice = self._remote_voice(local_key, account_id, main_voice)
            if not voice:
                raise AliyunCosyVoiceError("voice not registered on this account")

            def _run() -> bytes:
                # dashscope SDK 的 api_key 是全局变量，但 SpeechSynthesizer
                # 把它存为实例属性（self.apikey）——构造后覆写即可实例级多账号
                synth = sdk[3](**_synth_kwargs(voice))
                if api_key:
                    synth.apikey = api_key
                return synth.call(text)

            return await asyncio.to_thread(_run)

        # 同上：矩阵缺失时直接单账号调用
        try:
            from app.services.dashscope_key_pool import dashscope_key_pool
        except ImportError:
            dashscope_key_pool = None

        try:
            pcm = await asyncio.wait_for(
                (dashscope_key_pool.with_failover(_factory, local_voice_key=local_key)
                 if dashscope_key_pool else _factory(None, "")),
                timeout=float(settings.aliyun_cosyvoice_timeout_seconds) + 15,
            )
        except Exception as exc:
            raise AliyunCosyVoiceError(f"speech synthesis failed: {exc}") from exc
        if not pcm or len(pcm) % 2:
            raise AliyunCosyVoiceError("speech synthesis returned invalid PCM16 audio")

        wav = pcm_to_wav(pcm, sample_rate)
        return AliyunCosyVoiceAudio(
            wav_bytes=wav,
            content_type="audio/wav",
            duration_ms=wav_duration_ms(wav),
            used_reference=voice_id is not None,
        )

    async def synthesize_stream(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        voice_id: str | None = None,
        emotion: str | None = None,
        user_id: str | None = None,
        output_sample_rate: int | None = None,
    ) -> AsyncIterator[AliyunCosyVoiceStreamChunk]:
        del reference_audio_bytes, reference_text, user_id
        runtime = self._runtime()
        voice = self._resolve_voice(voice_id, runtime)
        sdk = self._configure_sdk(runtime)
        sample_rate = int(output_sample_rate or self.sample_rate)
        audio_format = self._pcm_format(sdk[1], sample_rate)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=32)
        synthesizer_holder: dict[str, Any] = {}

        overflowed = False

        def _offer(item: tuple[str, Any]) -> None:
            nonlocal overflowed
            if overflowed:
                return
            if queue.full():
                overflowed = True
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(
                    ("error", "streaming TTS output buffer exceeded its limit")
                )
                return
            queue.put_nowait(item)

        class Callback(sdk[2]):
            def on_open(self) -> None:
                return None

            def on_data(self, data: bytes) -> None:
                loop.call_soon_threadsafe(_offer, ("data", data))

            def on_event(self, message: str) -> None:
                loop.call_soon_threadsafe(_offer, ("event", message))

            def on_complete(self) -> None:
                return None

            def on_error(self, message: str) -> None:
                loop.call_soon_threadsafe(_offer, ("error", message))

            def on_close(self) -> None:
                return None

        def _run() -> None:
            try:
                model_name = self._model(runtime)
                synth_kwargs = dict(
                    model=model_name,
                    voice=voice,
                    format=audio_format,
                    language_hints=["zh"],
                    additional_params={"word_timestamp_enabled": True},
                    callback=Callback(),
                )
                # 朗读语速（方案 D1.3.2）：默认 1.0 不传参，配置合法且非默认才注入
                speech_rate = _speech_rate_param()
                if speech_rate is not None:
                    synth_kwargs["speech_rate"] = speech_rate
                # instruction 是 v3+ 参数：cosyvoice-v2 传了必报引擎 428
                # （实测 A/B 通过、C+instruction 失败，2026-08-26）
                if "v3" in model_name or "v4" in model_name:
                    synth_kwargs["instruction"] = _voice_instruction(voice, emotion)
                synthesizer = sdk[3](**synth_kwargs)
                synthesizer_holder["value"] = synthesizer
                synthesizer.streaming_call(text)
                synthesizer.streaming_complete()
            except Exception as exc:
                loop.call_soon_threadsafe(_offer, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(_offer, ("done", None))

        worker = asyncio.create_task(asyncio.to_thread(_run))
        try:
            while True:
                kind, value = await asyncio.wait_for(
                    queue.get(),
                    timeout=float(settings.aliyun_cosyvoice_timeout_seconds),
                )
                if kind == "done":
                    break
                if kind == "error":
                    raise AliyunCosyVoiceError(f"streaming speech synthesis failed: {value}")
                if kind == "event":
                    logger.debug("aliyun_cosyvoice_event message=%s", value)
                    continue
                pcm = bytes(value)
                if not pcm or len(pcm) % 2:
                    raise AliyunCosyVoiceError("stream returned invalid PCM16 chunk")
                yield AliyunCosyVoiceStreamChunk(
                    pcm_bytes=pcm,
                    sample_rate=sample_rate,
                    channels=1,
                    format="pcm_s16le",
                    duration_ms=round((len(pcm) // 2) * 1000 / sample_rate),
                )
            await worker
        finally:
            if not worker.done():
                synthesizer = synthesizer_holder.get("value")
                if synthesizer is not None:
                    with contextlib.suppress(Exception):
                        synthesizer.close()
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker


    def open_stream_session(
        self, *, voice_id: str | None, output_sample_rate: int,
        speech_rate: float | None = None, emotion: str | None = None,
    ) -> AliyunCosyVoiceStreamSession:
        """整轮单会话流式合成（句间无引擎启动缝）。

        speech_rate：该音色的独立语速（0.5–2.0）。克隆音色会继承参考音频的
        语速（实测慢速参考产出 3.0 字/秒 vs 正常 4-5），按音色校正而不动
        全局（全局提速会连带把正常音色带快）。None 回退全局配置/不传参。
        """
        runtime = self._runtime()
        voice = self._resolve_voice(voice_id, runtime)
        return AliyunCosyVoiceStreamSession(
            self, voice=voice, sample_rate=int(output_sample_rate or self.sample_rate),
            speech_rate=self._valid_speech_rate(speech_rate), emotion=emotion,
        )

    @staticmethod
    def _valid_speech_rate(rate: float | None) -> float | None:
        if rate is None:
            return None
        rate = float(rate)
        return rate if 0.5 <= rate <= 2.0 and abs(rate - 1.0) > 1e-6 else None


class AliyunCosyVoiceStreamSession:
    """一轮回复共用一个 dashscope TTS 会话。

    dashscope streaming_call 支持多次追加文本、最后一次 streaming_complete。
    相比"每句独立会话"，消除句间 400-900ms 引擎启动静默（衔接卡顿根因）。
    """

    def __init__(self, client: "AliyunCosyVoiceClient", *, voice: str, sample_rate: int,
                 speech_rate: float | None = None, emotion: str | None = None):
        import asyncio

        self._loop = asyncio.get_running_loop()
        self._queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue(maxsize=64)
        self._done = False
        runtime = client._runtime()
        sdk = client._configure_sdk(runtime)
        model_name = client._model(runtime)
        audio_format = client._pcm_format(sdk[1], sample_rate)
        self._sample_rate = sample_rate
        self._speech_rate = speech_rate
        self._emotion = emotion
        outer_loop = self._loop
        queue = self._queue

        class Callback(sdk[2]):
            def on_open(self) -> None:
                return None

            def on_data(self, data: bytes) -> None:
                outer_loop.call_soon_threadsafe(queue.put_nowait, ("data", data))

            def on_event(self, message: str) -> None:
                return None

            def on_complete(self) -> None:
                outer_loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

            def on_error(self, message: str) -> None:
                outer_loop.call_soon_threadsafe(queue.put_nowait, ("error", message))

            def on_close(self) -> None:
                return None

        synth_kwargs = dict(
            model=model_name,
            voice=voice,
            format=audio_format,
            language_hints=["zh"],
            volume=settings.aliyun_cosyvoice_volume,
            additional_params={"word_timestamp_enabled": True},
            callback=Callback(),
        )
        # 朗读语速：按音色独立值优先（克隆音色慢速校正），回退全局配置
        speech_rate = self._speech_rate if self._speech_rate is not None else _speech_rate_param()
        if speech_rate is not None:
            synth_kwargs["speech_rate"] = speech_rate
        if "v3" in model_name or "v4" in model_name:
            # 会话级情绪（情感语音）：跟随本会话首段情绪，默认 neutral
            synth_kwargs["instruction"] = _voice_instruction(
                voice, self._emotion)
        self._synthesizer = sdk[3](**synth_kwargs)
        # 账号矩阵：会话建立时选一个"持有该音色"的健康账号（429 冷却中/无映射
        # 的账号自动跳过）。SDK 全局 key 会被并发覆写，靠实例属性注入所选 key。
        # 多账号限流矩阵为完整版组件；开源版单账号直连（pool 不存在时回退）。
        try:
            from app.services.dashscope_key_pool import dashscope_key_pool
            local_key = AliyunCosyVoiceClient._local_voice_key(voice)
            account_id, api_key = dashscope_key_pool.pick(local_key)
        except ImportError:
            account_id, api_key = None, ""
        remote = AliyunCosyVoiceClient._remote_voice(local_key, account_id, voice)
        if account_id and remote:
            self._synthesizer.voice = remote
            self._synthesizer.apikey = api_key

    def _offer_error(self, message: str) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("error", message))

    async def feed_text(self, text: str) -> None:
        if self._done:
            raise AliyunCosyVoiceError("session already finished")
        import asyncio

        def _feed() -> None:
            try:
                self._synthesizer.streaming_call(text)
            except Exception as exc:  # noqa: BLE001
                self._offer_error(f"streaming_call failed: {exc}")

        await asyncio.to_thread(_feed)

    async def finish(self) -> None:
        import asyncio

        def _final() -> None:
            try:
                self._synthesizer.streaming_complete()
            except Exception as exc:  # noqa: BLE001
                self._offer_error(f"streaming_complete failed: {exc}")

        await asyncio.to_thread(_final)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._synthesizer.close()

    async def chunks(self):
        """产出 PCM 块直至 done/error。"""
        import asyncio

        while True:
            kind, value = await asyncio.wait_for(
                self._queue.get(),
                timeout=float(settings.aliyun_cosyvoice_timeout_seconds),
            )
            if kind == "done":
                self._done = True
                return
            if kind == "error":
                self._done = True
                raise AliyunCosyVoiceError(f"streaming speech synthesis failed: {value}")
            pcm = bytes(value)
            if not pcm or len(pcm) % 2:
                self._done = True
                raise AliyunCosyVoiceError("stream returned invalid PCM16 chunk")
            yield AliyunCosyVoiceStreamChunk(
                pcm_bytes=pcm,
                sample_rate=self._sample_rate,
                channels=1,
                format="pcm_s16le",
                duration_ms=round((len(pcm) // 2) * 1000 / self._sample_rate),
            )


aliyun_cosyvoice_client = AliyunCosyVoiceClient()
