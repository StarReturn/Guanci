"""Qwen-Audio-TTS / Qwen3-TTS-VC 供应商客户端（阿里云百炼，独立于 CosyVoice）。

选择理由（2026-10 评估）：
- 限流独立桶：qwen3-tts 系 180 RPM，与 CosyVoice 家族 3 RPS 互不占用，
  双引擎并行 = TTS 总容量翻倍，且 RPM 形状对"同时响起"的通话突发更友好。
- 声音复刻：qwen-voice-enrollment，约 10-20s（部分场景 3s）音频即可注册；
  复刻音色与 target_model 绑定、不能跨模型复用——切换引擎需重新注册全部音色。

接口对齐（与 fish_speech/indextts 同风格，供 engine_factory 分发）：
- ``synthesize_batch(text, reference_audio_bytes, reference_text, voice_id=None) -> QwenTtsAudio``
  voice_id 优先（已绑定的复刻音色）；否则用参考音频走 enrollment 注册
  （进程内按 sha256(ref+model) 缓存映射，避免重复注册占用音色位）。
- ``health_check() -> dict``

边界（v1 明确不做）：
- 无流式合成（qwen3-tts-realtime 为 WebSocket 协议，v2 实时通话仍锁
  aliyun_cosyvoice，由 realtime_ws 的 provider 守卫保证）；文字聊天与
  语音条（HTTP batch 路径）完整可用。
- emotion 指令不支持（qwen3-tts 无 instruction 参数）：静默忽略，
  语气随文本自适应（官方行为）。

运维事实：
- 复刻音色账号级上限 1000 个（复刻+设计共用、跨模型共享），约一年未使用
  会被自动清理——参考音频底档必须保留在 MinIO，最坏可静默重注册恢复。
- HTTP 契约依据百炼《声音复刻》官方文档（2026-10 摘录）：
  enrollment POST {base}/services/audio/tts/customization（action=create_voice，
  audio 支持 base64 Data URI），轮询 query_voice 至 status=OK；
  合成 POST {base}/services/audio/tts/SpeechSynthesizer → output.audio.url 下载。
  参数名如有出入，联调时只需改 _enroll/_synthesize 两处。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time

import httpx

from app.core.config import settings
from app.services.tts_provider_service import tts_provider_service

logger = logging.getLogger(__name__)


class QwenTtsError(RuntimeError):
    """Qwen TTS 调用失败（注册/合成/配置）。"""


class QwenTtsAudio:
    def __init__(self, wav_bytes: bytes, duration_ms: int, voice_id: str | None):
        self.wav_bytes = wav_bytes
        self.content_type = "audio/wav"
        self.duration_ms = duration_ms
        self.voice_id = voice_id
        self.used_reference = voice_id is not None


class QwenTtsClient:
    provider = "qwen_tts"

    def __init__(self) -> None:
        # 进程级缓存：sha256(ref_wav + model) -> voice_id（重注册去重，省音色位）
        self._voice_cache: dict[str, str] = {}

    # ---- 配置 ----

    @property
    def sample_rate(self) -> int:
        return 24000

    @staticmethod
    def voice_prefix(user_id: str) -> str:
        import hashlib as _h

        return "dh" + _h.sha1(user_id.encode("utf-8")).hexdigest()[:8]

    def _runtime(self):
        runtime = tts_provider_service.runtime_for(self.provider)
        model = runtime.model or settings.qwen_tts_model
        api_key = (runtime.api_key or settings.dashscope_api_key or "").strip()
        if not api_key:
            raise QwenTtsError(
                "qwen_tts api key missing: set tts_providers.api_key or DASHSCOPE_API_KEY"
            )
        # base 必须含 /api/v1：customization/SpeechSynthesizer 都挂在该前缀下
        base = (runtime.endpoint or settings.qwen_tts_base_url).rstrip("/")
        if not base.endswith("/api/v1"):
            base = base + "/api/v1"
        return base, model, api_key

    def _headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # ---- 声音复刻（enrollment）----

    async def create_voice(
        self, *, audio_url: str | None = None, audio_wav: bytes | None = None,
        user_id: str, duration_ms: int | None = None,
    ) -> str:
        """注册复刻音色，轮询就绪后返回 voice_id。二选一：公网 URL 或 wav 字节
        （wav 会转存 MinIO 取 presigned URL——enrollment 只接受公网 url 字段）。"""
        base, model, api_key = self._runtime()
        if not audio_url and not audio_wav:
            raise QwenTtsError("voice enrollment requires audio_url or audio_wav")
        if not audio_url:
            import uuid as _uuid

            from app.services.storage import storage_service

            key = f"qwen_refs/{_uuid.uuid4().hex}.wav"
            storage_service.put_bytes(key=key, content=audio_wav, content_type="audio/wav")
            audio_url = storage_service.presigned_url(key)
        # body 必须 input 包裹（与 dashscope SDK VoiceEnrollment 同构，2026-10 实测）
        payload = {
            "model": "voice-enrollment",
            "input": {
                "action": "create_voice",
                "target_model": model,
                "prefix": self.voice_prefix(user_id),
                "url": audio_url,
            },
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{base}/services/audio/tts/customization",
                json=payload, headers=self._headers(api_key))
            voice_id = self._unwrap(r, "create_voice").get("voice_id")
            if not voice_id:
                raise QwenTtsError("voice enrollment returned no voice_id")
            # 轮询就绪（官方流程；一般秒级，上限 60s）。无 status 字段视为已就绪。
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                q = await client.post(
                    f"{base}/services/audio/tts/customization",
                    json={"model": "voice-enrollment",
                          "input": {"action": "query_voice", "voice_id": voice_id}},
                    headers=self._headers(api_key))
                out = self._unwrap(q, "query_voice")
                status = str(out.get("status", "")).upper()
                if not status or status == "OK":
                    return voice_id
                if status in {"FAILED", "DELETED"}:
                    raise QwenTtsError(f"voice enrollment failed: status={status}")
                await asyncio.sleep(1.5)
        raise QwenTtsError("voice enrollment poll timeout (60s)")

    def delete_voice_sync(self, voice_id: str) -> None:
        """删除复刻音色（尽力而为；不存在视同成功）。"""
        base, model, api_key = self._runtime()

        def _call() -> None:
            with httpx.Client(timeout=30) as client:
                r = client.post(
                    f"{base}/services/audio/tts/customization",
                    json={"model": "voice-enrollment",
                          "input": {"action": "delete_voice", "voice_id": voice_id}},
                    headers=self._headers(api_key))
                # 404/已删除同样视为成功，删除路径不能因竞态抛错
                if r.status_code not in (200, 404):
                    raise QwenTtsError(f"delete_voice http {r.status_code}: {r.text[:200]}")

        _call()

    # ---- 合成 ----

    async def synthesize_batch(
        self, *, text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        voice_id: str | None = None,
        emotion: str | None = None,
        output_sample_rate: int | None = None,
        speech_rate: float | None = None,
    ) -> QwenTtsAudio:
        del reference_text, emotion, speech_rate  # qwen3-tts 无对应参数：语气随文本自适应
        base, model, api_key = self._runtime()
        vid = voice_id or await self._voice_for_reference(reference_audio_bytes)
        if not vid:
            raise QwenTtsError(
                "no voice available: bind a preset voice or provide reference audio"
            )
        payload = {
            "model": model,
            "input": {"text": text, "voice": vid,
                      "format": "wav",
                      "sample_rate": int(output_sample_rate or self.sample_rate)},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{base}/services/audio/tts/SpeechSynthesizer",
                json=payload, headers=self._headers(api_key))
            out = self._unwrap(r, "synthesize")
            audio_url = (out.get("audio") or {}).get("url")
            if not audio_url:
                raise QwenTtsError("synthesize returned no audio url")
            dl = await client.get(audio_url, headers={"Authorization": f"Bearer {api_key}"})
            dl.raise_for_status()
            wav = self._fix_wav_sizes(dl.content)
        from app.services.audio_utils import wav_duration_ms

        return QwenTtsAudio(wav_bytes=wav, duration_ms=wav_duration_ms(wav), voice_id=vid)

    @staticmethod
    def _fix_wav_sizes(wav: bytes) -> bytes:
        """qwen 流式管道 wav 的 RIFF/data size 是 0x7FFFFFFF 占位符（与当年
        CosyVoice 管道输出同款问题），按实际长度重写，否则 wave 读出天文数字。"""
        import struct as _struct

        if len(wav) < 44 or wav[:4] != b"RIFF":
            return wav
        buf = bytearray(wav)
        idx = 12
        data_at = None
        while idx + 8 <= len(buf):
            cid = bytes(buf[idx:idx + 4])
            sz = _struct.unpack("<I", bytes(buf[idx + 4:idx + 8]))[0]
            if cid == b"data":
                data_at = idx
                break
            idx += 8 + sz + (sz % 2)
        if data_at is None:
            return wav
        real = len(buf) - data_at - 8
        buf[data_at + 4:data_at + 8] = _struct.pack("<I", real)
        buf[4:8] = _struct.pack("<I", len(buf) - 8)
        return bytes(buf)

    async def synthesize_stream(self, *, text: str, **kwargs):
        """逐句回退路径：无原生流式，整句合成后按 PCM 帧吐出。
        签名对齐 aliyun 客户端，供 tts_service.synthesize_stream 通用调用。"""
        audio = await self.synthesize_batch(text=text, **kwargs)
        import wave

        with wave.open(io := __import__("io").BytesIO(audio.wav_bytes), "rb") as wf:
            rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            while True:
                frame = wf.readframes(int(rate * 0.04))  # 40ms 帧
                if not frame:
                    break
                yield (rate, frame) if channels == 1 and width == 2 else (rate, frame)
        del io

    # ---- 参考音频 -> voice_id 注册缓存 ----

    async def _voice_for_reference(self, ref_wav: bytes | None) -> str | None:
        if not ref_wav:
            return None
        _, model, _ = self._runtime()
        key = hashlib.sha256(ref_wav + model.encode()).hexdigest()
        vid = self._voice_cache.get(key)
        if vid:
            return vid
        vid = await self.create_voice(audio_wav=ref_wav, user_id="qwen_ref")
        self._voice_cache[key] = vid
        logger.info("qwen_tts voice enrolled sha=%s vid=%s", key[:12], vid)
        return vid

    # ---- 健康 ----

    async def health_check(self) -> dict:
        base, model, api_key = self._runtime()
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{base}/services/audio/tts/customization",
                json={"model": "voice-enrollment", "action": "query_voice",
                      "voice_id": "healthcheck-probe"},
                headers=self._headers(api_key))
            # 能完成鉴权与路由即视为健康（probe 音色不存在属正常业务响应）
            return {"status": "ok", "provider": self.provider,
                    "model": model, "http": r.status_code}

    # ---- 内部 ----

    @staticmethod
    def _unwrap(resp: httpx.Response, stage: str) -> dict:
        if resp.status_code != 200:
            raise QwenTtsError(f"{stage} http {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        out = body.get("output") or {}
        if body.get("code") or out.get("code"):
            raise QwenTtsError(f"{stage} failed: {body.get('message') or out.get('message')}")
        return out


qwen_tts_client = QwenTtsClient()
