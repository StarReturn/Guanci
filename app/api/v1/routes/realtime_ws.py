"""WebSocket realtime endpoint — phone-call style voice dialogue.

A "phone call" UX rather than push-to-talk: the socket is held open
for the whole conversation, both sides can speak at any time, and the
digital human's reply can be **interrupted** (barge-in) by the user
starting a new utterance.

Endpoint: ``WS /api/v1/realtime/stream``

Auth:
  Browsers cannot set Authorization header on the WS handshake. So the
  JWT access token is passed as a query parameter ``token=``. The server
  decodes it once on connect; if invalid/expired the socket closes with
  code 4401. **Server-side access logs that record query strings will
  therefore contain JWTs — configure your reverse proxy to scrub
  ``token=`` from logs** (nginx does this by default for access_log
  since 1.13; for older nginx, see deploy/nginx snippet in the API doc).

Protocol (text frames = JSON control messages, binary frames = audio):

  Client → Server (text):
    {"type":"start","protocol_version":"1.0-draft","digital_human_id":"..."}
    {"type":"end"}                         # explicit end of user utterance
    {"type":"interrupt"}                   # barge-in: stop current reply
    {"type":"ping"}                        # keepalive (server echoes {"type":"pong"})
    {"type":"close"}                       # graceful close

  Client → Server (binary):
    PCM16 16kHz mono little-endian, or WAV (auto-detected). VAD splits
    utterances; each chunk is appended to the turn buffer.

  Server → Client (text):
    {"type":"ready","protocol_version":"1.0-draft","audio_input":{...},"renderer":{...},...}
    {"type":"asr_final","text":"..."}      # final user text (after end/VAD silence)
    {"type":"reply_start","audio_id":"...","sample_rate":24000,"channels":1,...}
    {"type":"motion_chunk","audio_id":"...","chunk_seq":N,"segments":[...],"energy":[...],...}
    {"type":"reply_chunk","seq":N,"byte_length":...,...}  # immediately followed by 1 binary PCM frame
    {"type":"sentence_end","audio_id":"...","sentence_seq":N,...}
    {"type":"reply_end","audio_id":"...","total_samples":...,"reply_text":"...",...}
    {"type":"motion_interrupted","audio_id":"...","at_sample":N}  # barge-in: stop playback
    {"type":"error","code":"...","detail":"...","retryable":bool,"fatal":bool,...}
    {"type":"pong"}                        # response to client ping

  The wire protocol source of truth is
  contracts/mobile-sdk/v1/schemas/websocket-{client,server}-message.schema.json
  plus error-codes.json. This docstring is informational.

Barge-in protocol:
  1. While the server is synthesizing/streaming a reply, the client
     sends ``{"type":"interrupt"}`` OR simply sends a new ``start`` +
     binary audio chunk.
  2. The server cancels the in-flight LLM/TTS task
     (``current_reply_task.cancel()``), flushes any queued reply chunks,
     and emits ``{"type":"motion_interrupted","audio_id":"...","at_sample":N}``
     for every reply that was in progress so the client can stop playback.
  3. The server then begins processing the new user utterance normally.

Mobile-side requirements (NOT enforced here, document in client SDK):
  * Microphone capture: 16kHz mono PCM16.
  * **AEC + noise suppression** before sending chunks. Use the OS's
    built-in acoustic echo cancellation (Android: AudioRecord with
    VOICE_COMMUNICATION source; iOS: AVAudioSession category
    playAndRecord + mode voiceChat). Without AEC the digital human's
    reply leaks into the mic and triggers false barge-in.
  * VAD is recommended client-side to coalesce silence gaps; the server
    also runs VAD as a safety net.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.jwt_utils import JWTError, decode
from app.services.asr_service import asr_service
from app.services.auth_service import audit_service, user_avatar_service
from app.services.digital_human_service import (
    DigitalHumanNotFound,
    digital_human_service,
)
from app.services.llm_service import dialogue_service, llm_provider_service
from app.services.memory_provider_service import (
    MemoryProviderNotConfigured,
    memory_provider_service,
)
from app.services.entitlement_stub import EntitlementDenied, entitlement_service
from app.services.phoneme_service import phoneme_service
from app.services.realtime_session_service import realtime_session_service
from app.services.tts_service import (
    resolve_voice_reference,
    provider_name,
    stream_sample_rate,
    synthesize_batch,
    synthesize_stream,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------#
# Protocol constants & structured-error factory (Gate 0A)
# --------------------------------------------------------------#
# Single source of truth for these constants is
# contracts/mobile-sdk/v1/{protocol-version.json, error-codes.json}.
# protocol_version stays "1.0-draft" until Gate 0B (server golden package)
# passes; only then is it raised to "1.0".
PROTOCOL_VERSION = "1.0-draft"
READY_HEARTBEAT_INTERVAL_MS = 30000
READY_RENDERER_ENGINE = "mobile_keyframe_v3"
READY_RENDERER_SCHEMA_REVISION = 2
READY_AUDIO_INPUT_FORMAT = "pcm_s16le"
READY_AUDIO_INPUT_SAMPLE_RATE = 16000
READY_AUDIO_INPUT_CHANNELS = 1


def _load_error_code_defaults() -> dict[str, dict[str, bool]]:
    """Load error-code -> {retryable, fatal} defaults from the contract file.

    The contract file is the single source of truth; if it cannot be read
    we fall back to safe defaults (retryable=False, fatal=False) so the
    server still boots, but a missing contract is itself a bug to fix.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[4]  # backend-mobile-keyframe/
    path = root / "contracts" / "mobile-sdk" / "v1" / "error-codes.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            code: {
                "retryable_default": bool(v.get("retryable_default", False)),
                "fatal_default": bool(v.get("fatal_default", False)),
            }
            for code, v in data.get("codes", {}).items()
        }
    except FileNotFoundError:
        # 契约文件可选（端侧 SDK 契约，不随后端分发）；缺失用空默认
        logger.warning("error_codes_contract_absent, using built-in defaults")
        return {}
    except Exception:
        logger.exception("error_codes_contract_load_failed path=%s", path)
        return {}


_ERROR_CODE_DEFAULTS: dict[str, dict[str, bool]] = _load_error_code_defaults()


async def _send_error(
    state: "_ConnectionState",
    code: str,
    detail: str,
    *,
    audio_id: str | None = None,
    sentence_seq: int | None = None,
    request_id: str | None = None,
    retryable: bool | None = None,
    fatal: bool | None = None,
) -> None:
    """Send a structured ``error`` message per contracts/mobile-sdk/v1.

    ``retryable`` / ``fatal`` default to the values declared in
    error-codes.json for the given ``code``; callers may override per-call.
    SDK clients MUST NOT parse ``detail`` to decide behavior.
    """
    defaults = _ERROR_CODE_DEFAULTS.get(code, {})
    msg: dict[str, Any] = {
        "type": "error",
        "code": code,
        "detail": detail,
        "retryable": bool(retryable if retryable is not None else defaults.get("retryable_default", False)),
        "fatal": bool(fatal if fatal is not None else defaults.get("fatal_default", False)),
    }
    if audio_id is not None:
        msg["audio_id"] = audio_id
    if sentence_seq is not None:
        msg["sentence_seq"] = sentence_seq
    if request_id is not None:
        msg["request_id"] = request_id
    await _send_json(state, msg)


# --------------------------------------------------------------#
# Connection state
# --------------------------------------------------------------#
class _ConnectionState:
    """Per-connection mutable state.

    All fields are touched only from the WebSocket receive loop OR from
    a single reply task — no locking needed because asyncio is
    cooperative and we never ``await`` between read-modify-write of
    these fields.
    """

    def __init__(
        self, ws: WebSocket, user_id: str, digital_human_id: str | None = None
    ) -> None:
        self.ws = ws
        self.user_id = user_id
        self.digital_human_id = digital_human_id
        self.session_id: str | None = None
        # Audio buffer for the current user utterance (PCM bytes).
        # Kept for parity / debug; the primary path now feeds the ASR
        # session incrementally (see current_asr_session).
        self.turn_buffer = bytearray()
        # Currently running reply task (LLM→TTS→stream). Cancelled on barge-in.
        self.current_reply_task: asyncio.Task | None = None
        # Audio_id of the reply currently being streamed, so we can
        # emit the right ``motion_interrupted`` message on barge-in.
        self.current_reply_audio_id: str | None = None
        # ---- Stage B: true streaming ASR session ----
        # Active ASR session for the current user turn (start→end). None
        # when no turn is in progress.
        self.current_asr_session: Any = None
        # Background task that drains ASR partial events and forwards them
        # to the client. Created on start, cancelled/awaited on end/interrupt.
        self.current_asr_event_task: asyncio.Task | None = None
        # Monotonic per-turn id (incremented on each start).
        self.current_user_turn_id: int = 0
        # Count of input PCM samples received this turn (for bounds/debug).
        self.current_input_sample_count: int = 0
        # Last forwarded partial text (for de-dup; same-text partials are
        # not re-sent to the client).
        self.current_partial_text: str = ""


# --------------------------------------------------------------#
# Outbound helpers
# --------------------------------------------------------------#
async def _send_json(state: _ConnectionState, msg: dict[str, Any]) -> None:
    """Send a JSON control message. Swallow send errors (socket gone)."""
    try:
        await state.ws.send_text(json.dumps(msg, ensure_ascii=False))
    except Exception:
        logger.debug("ws_send_failed", exc_info=True)


async def _send_bytes(state: _ConnectionState, data: bytes) -> None:
    try:
        await state.ws.send_bytes(data)
    except Exception:
        logger.debug("ws_send_bytes_failed", exc_info=True)


# --------------------------------------------------------------#
# Reply pipeline (LLM → TTS → timeline → stream chunks)
# --------------------------------------------------------------#
async def _run_reply(state: _ConnectionState, user_text: str) -> None:
    """Legacy batch reply pipeline (fish_speech / non-streaming TTS).

    Gate 0A froze a single realtime protocol that requires streaming
    motion_chunk + reply_chunk with byte_length/sample_count. The legacy
    batch path emitted a different, simpler reply_chunk shape and an
    ``interrupted`` event, which conflicts with the frozen contract.
    Rather than maintain two parallel wire shapes, the legacy path now
    refuses to serve realtime sessions: callers must configure a
    streaming-capable TTS provider (cosyvoice / aliyun_cosyvoice).

    The body below is preserved only to keep entitlement bookkeeping
    symmetrical with the streaming path; it never streams audio.
    """
    audio_id = str(uuid4())
    state.current_reply_audio_id = audio_id
    usage_request_id = f"realtime-{audio_id}"
    usage_reserved = False
    usage_confirmed = False

    try:
        try:
            entitlement_service.reserve(
                user_id=state.user_id,
                key="dialogue_audio_seconds_per_month",
                quantity=120,
                request_id=usage_request_id,
                resource_type="realtime_session",
                resource_id=state.session_id,
            )
            usage_reserved = True
        except EntitlementDenied as exc:
            await _send_error(
                state, "ENTITLEMENT_LIMIT_EXCEEDED", str(exc),
                request_id=usage_request_id,
            )
            return

        # Gate 0A: the legacy batch reply path cannot emit the frozen
        # streaming motion_chunk / reply_chunk contract. Refuse instead
        # of silently serving a non-conformant wire shape. Operators
        # must configure a streaming-capable TTS provider.
        logger.error(
            "realtime_legacy_path_invoked provider=%s — refusing to serve "
            "non-streaming TTS on the realtime channel",
            provider_name(),
        )
        await _send_error(
            state,
            "TTS_PROVIDER_UNSUPPORTED",
            f"TTS provider '{provider_name()}' does not support the realtime "
            "streaming protocol; configure cosyvoice or aliyun_cosyvoice.",
            audio_id=audio_id,
            fatal=True,
        )

    except asyncio.CancelledError:
        # Legacy path no longer streams, so there is nothing to interrupt;
        # we still emit motion_interrupted for contract symmetry in case a
        # client sent `interrupt` between task creation and this fatal error.
        await _send_json(state, {
            "type": "motion_interrupted",
            "audio_id": audio_id,
            "at_sample": 0,
        })
        raise
    finally:
        if usage_reserved and not usage_confirmed:
            try:
                entitlement_service.release(
                    user_id=state.user_id,
                    key="dialogue_audio_seconds_per_month",
                    request_id=usage_request_id,
                )
            except Exception:
                logger.exception("realtime_entitlement_release_failed")
        if state.current_reply_audio_id == audio_id:
            state.current_reply_audio_id = None
        if state.current_reply_task is not None and state.current_reply_task.done():
            state.current_reply_task = None


def _cancel_current_reply(state: _ConnectionState) -> None:
    """Cancel any in-flight reply task. Safe to call when no task is active."""
    if state.current_reply_task is None or state.current_reply_task.done():
        state.current_reply_task = None
        return
    state.current_reply_task.cancel()
    # We do NOT await here — the caller will see ``interrupted`` once the
    # task's except clause runs.


# --------------------------------------------------------------#
# Stage B: streaming ASR session lifecycle
# --------------------------------------------------------------#
async def _start_asr_turn(state: _ConnectionState) -> None:
    """Cancel any in-flight ASR turn, then start a fresh streaming session.

    Called when the client sends ``start``. Increments the turn id and
    resets per-turn counters. Partial events are drained by a background
    task that forwards them to the client (de-duplicated).
    """
    await _cancel_asr_turn(state)
    state.current_user_turn_id += 1
    state.current_input_sample_count = 0
    state.current_partial_text = ""
    state.turn_buffer = bytearray()

    from app.services.engine_factory import get_asr_engine
    engine = get_asr_engine()
    if not hasattr(engine, "create_session"):
        # Whisper batch path has no streaming session; fall back to the
        # legacy "buffer until end" behaviour. No partials are emitted.
        return

    try:
        session = await engine.create_session(
            sample_rate=16000, channels=1, encoding="pcm_s16le",
        )
        await session.start()
    except Exception as exc:
        logger.exception("asr_session_start_failed")
        await _send_error(state, "ASR_FAILED", f"ASR session start failed: {exc}")
        return

    state.current_asr_session = session
    state.current_asr_event_task = asyncio.create_task(_drain_asr_partials(state))


async def _drain_asr_partials(state: _ConnectionState) -> None:
    """Forward partial ASR results to the client (de-duplicated)."""
    session = state.current_asr_session
    if session is None:
        return
    try:
        async for result in session.events():
            if not result.is_final:
                text = result.text or ""
                if text != state.current_partial_text:
                    state.current_partial_text = text
                    await _send_json(state, {"type": "asr_partial", "text": text})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("asr_partial_drain_error", exc_info=True)
        await _send_error(state, "ASR_FAILED", f"ASR partial stream error: {exc}")


async def _cancel_asr_turn(state: _ConnectionState) -> None:
    """Cancel the active ASR session and its event drainer, if any."""
    task = state.current_asr_event_task
    state.current_asr_event_task = None
    session = state.current_asr_session
    state.current_asr_session = None
    if session is not None:
        try:
            await session.cancel()
        except Exception:
            logger.debug("asr_session_cancel_failed", exc_info=True)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def _finalize_asr_turn(state: _ConnectionState) -> str:
    """Finish the ASR session and return the final recognized text.

    Returns "" if no session was active or the final was empty. Sends
    ``asr_final`` to the client. Raises nothing — ASR failures are
    converted to structured errors before reaching this return.
    """
    session = state.current_asr_session
    state.current_asr_session = None
    # Stop the partial drainer once we are finishing.
    task = state.current_asr_event_task
    state.current_asr_event_task = None

    if session is None:
        # Whisper batch fallback: transcribe the buffered audio.
        if not state.turn_buffer:
            await _send_json(state, {"type": "asr_final", "text": ""})
            return ""
        audio_bytes = bytes(state.turn_buffer)
        state.turn_buffer = bytearray()
        try:
            text = await asyncio.to_thread(asr_service.transcribe, audio_bytes)
        except Exception as exc:
            logger.exception("ws_asr_failed")
            await _send_error(state, "ASR_FAILED", f"ASR failed: {exc}")
            return ""
        await _send_json(state, {"type": "asr_final", "text": text})
        return text.strip()

    try:
        final = await session.finish()
    except Exception as exc:
        logger.exception("ws_asr_finish_failed")
        await _send_error(state, "ASR_FAILED", f"ASR finish failed: {exc}")
        final = None
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    text = (final.text if final else "").strip()
    await _send_json(state, {"type": "asr_final", "text": text})
    return text


# --------------------------------------------------------------#
# Streaming reply pipeline (FunASR + GLM SSE + CosyVoice stream)
# --------------------------------------------------------------#
# 句子边界符号：中文 + 英文标点
_SENTENCE_ENDINGS = tuple("。！？，；.!?;,")
# 单段最长字符（避免 LLM 一段不结束导致 TTS 等太久）
_MAX_SENTENCE_CHARS = 80


async def _run_reply_streaming(state: _ConnectionState, user_text: str) -> None:
    """v4 流式回复管道：边收 LLM token → 边检测句子 → 边送 TTS → 边下发 PCM。

    首音频延迟预估 ~1.2s（流利级）。
    任一阶段 await 都可被 barge-in cancel。
    """
    from app.core.config import settings
    from app.services.emotion_classifier import extract_emotion
    from app.core.gpu_semaphore import gpu_semaphore
    from app.services.engine_factory import get_tts_engine

    audio_id = str(uuid4())
    state.current_reply_audio_id = audio_id
    gpu_acquired = False
    usage_request_id = f"realtime-{audio_id}"
    usage_reserved = False
    usage_confirmed = False

    try:
        try:
            entitlement_service.reserve(
                user_id=state.user_id,
                key="dialogue_audio_seconds_per_month",
                quantity=120,
                request_id=usage_request_id,
                resource_type="realtime_session",
                resource_id=state.session_id,
            )
            usage_reserved = True
        except EntitlementDenied as exc:
            await _send_error(
                state, "ENTITLEMENT_LIMIT_EXCEEDED", str(exc),
                request_id=usage_request_id,
            )
            return
        # ---- 获取 GPU 推理槽位 ----
        acquire = await gpu_semaphore.acquire()
        gpu_acquired = acquire.mode == "gpu"
        mode = acquire.mode  # "gpu" / "cpu"
        if mode == "cpu":
            await _send_error(
                state, "TTS_CAPACITY_EXHAUSTED",
                "TTS capacity is exhausted; retry shortly",
                audio_id=audio_id,
            )
            return

        # ---- LLM 流式 ----
        try:
            active_llm = await asyncio.to_thread(
                memory_provider_service.get_provider_for_role, "chat"
            )
        except MemoryProviderNotConfigured as exc:
            await _send_error(
                state, "LLM_FAILED", str(exc),
                request_id=usage_request_id,
            )
            return

        # 注入情绪标签 system prompt
        from app.services.emotion_classifier import build_emotion_system_prompt_suffix
        system_prompt = (active_llm.system_prompt or "")
        if state.digital_human_id:
            persona_prompt = digital_human_service.build_persona_prompt(
                state.user_id, state.digital_human_id
            )
            system_prompt = f"{system_prompt}\n\n{persona_prompt}".strip()
        if "[EMOTION:" not in system_prompt:
            system_prompt = system_prompt + "\n\n" + build_emotion_system_prompt_suffix()

        membership = entitlement_service.get(state.user_id)
        llm_messages = await asyncio.to_thread(
            dialogue_service.build_llm_messages,
            state.session_id, user_text, system_prompt=system_prompt,
            user_id=state.user_id, avatar_id=state.digital_human_id,
            cross_session_memory=bool(
                membership.entitlements.get("cross_session_memory", False)
            ),
            memory_retention_days=int(
                membership.entitlements.get("memory_retention_days", 0)
            ),
            memory_items_per_avatar=int(
                membership.entitlements.get("memory_items_per_avatar", 0)
            ),
        )

        # ---- 准备 zero-shot 参考音频 ----
        reference = resolve_voice_reference(
            state.user_id, state.digital_human_id
        )
        output_sample_rate = stream_sample_rate()

        # ---- 流式管道：LLM token → 句子切分 → TTS batch → motion_chunk 先发 → PCM 后发 ----
        reply_full_text = ""          # 完整回复文本（用于 DB 持久化）
        sentence_buffer = ""          # 当前未结束的句子缓冲
        seq = 0                        # reply_chunk 序号（per audio_id）
        sentence_seq = 0               # 句子序号
        chunk_seq = 0                  # motion_chunk 序号（per audio_id, Gate 0A 统一命名）
        total_samples_written = 0     # 全局 PCM sample 计数（替代 total_duration_ms 作为主时钟）

        # 先发 reply_start
        await _send_json(state, {
            "type": "reply_start",
            "audio_id": audio_id,
            "voice_cloned": reference.available,
            "streaming": True,
            "audio_format": "pcm_s16le",
            "sample_rate": output_sample_rate,
            "channels": 1,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        async def _flush_sentence(sentence: str) -> None:
            """v3 短句准实时：batch TTS → 对齐 → motion_chunk 先发 → PCM 后发。

            关键约束（docs §7）：
            - motion_chunk 必须在对应 PCM 可播放之前到达客户端
            - 时间轴使用 sample_index（整数），不累计 duration_ms
            - 对齐失败必须明确终止，禁止静默用等分时间轴冒充
            """
            nonlocal seq, sentence_seq, chunk_seq, total_samples_written

            clean_text, emotion = extract_emotion(sentence)
            if not clean_text.strip():
                return

            # 1. 推送 sentence_start（不含 timeline，timeline 在 motion_chunk 里）
            await _send_json(state, {
                "type": "sentence_start",
                "audio_id": audio_id,
                "sentence_seq": sentence_seq,
                "start_sample": total_samples_written,
                "emotion": emotion,
                "text": clean_text,
            })

            # 2. Batch TTS：合成完整短句 PCM（不用流式，保证对齐在发送前完成）
            try:
                from app.services.audio_utils import wav_to_pcm, pcm_to_wav

                audio_result = await synthesize_batch(
                    text=clean_text,
                    digital_human_id=state.digital_human_id,
                    emotion=emotion,
                    user_id=state.user_id,
                )
                pcm_bytes, sample_rate = wav_to_pcm(audio_result.wav_bytes)
                sentence_pcm = bytes(pcm_bytes)
                sentence_samples = len(sentence_pcm) // 2  # 16-bit mono
                if sentence_samples <= 0:
                    raise RuntimeError("TTS returned empty audio")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("ws_tts_batch_failed")
                raise RuntimeError(f"TTS batch failed: {exc}") from exc

            # 3. 对最终 PCM 执行真实 forced alignment + RMS
            try:
                timeline = await phoneme_service.align_text(
                    text=clean_text,
                    duration_ms=audio_result.duration_ms,
                    audio_id=f"{audio_id}:{sentence_seq}",
                    audio_wav=pcm_to_wav(sentence_pcm, sample_rate),
                )
            except Exception:
                logger.exception("ws_align_failed sentence_seq=%d", sentence_seq)
                # 对齐失败必须明确终止，禁止静默等分
                await _send_error(
                    state, "PHONEME_ALIGNMENT_FAILED",
                    "phoneme alignment failed; sentence skipped",
                    audio_id=audio_id, sentence_seq=sentence_seq,
                )
                # 跳过该句的 PCM 发送（不播没有口型的音频）
                sentence_seq += 1
                return

            # 4. 构建 motion_chunk（viseme + openness + sample_index）
            #    同时计算 energy envelope（per energy_hop_samples RMS）。
            energy_hop_samples = max(1, sample_rate // 50)  # 20ms hop
            segments = []
            energy_values: list[float] = []
            if timeline is not None:
                from app.core.phoneme_map import VISEME_NATURAL_OPENNESS

                raw = []
                for seg in timeline.model_dump().get("timeline", []):
                    viseme = seg.get("viseme", "mouth_closed")
                    # Openness is the viseme's ANATOMICAL aperture (a→6, i→3,
                    # m→1) — NOT alignment confidence, which is uniform ~1.0
                    # and produced a constant-level, rhythm-less mouth.
                    openness = VISEME_NATURAL_OPENNESS.get(viseme, 3)
                    # 置信度只微调幅度（±1），不再决定开合
                    strength = float(seg.get("strength", 0.5))
                    openness = min(6, max(0, openness + (1 if strength > 0.9 else 0)))
                    raw.append({
                        "phoneme": seg.get("phoneme", ""),
                        "viseme": viseme,
                        "start_sample": total_samples_written + int(seg.get("start_ms", 0) * sample_rate / 1000),
                        "peak_sample": total_samples_written + int(seg.get("peak_ms", seg.get("start_ms", 0)) * sample_rate / 1000),
                        "end_sample": total_samples_written + int(seg.get("end_ms", 0) * sample_rate / 1000),
                        "_openness": openness,
                        "blend_in_samples": int(sample_rate * 0.02),   # 20ms
                        "blend_out_samples": int(sample_rate * 0.03),  # 30ms
                    })

                # 合并同 viseme 相邻段（协同发音），并把 <60ms 的碎段并入前段
                min_samples = int(sample_rate * 0.06)
                for seg in raw:
                    if (
                        segments
                        and (seg["viseme"] == segments[-1]["viseme"]
                             or seg["end_sample"] - seg["start_sample"] < min_samples)
                    ):
                        prev = segments[-1]
                        prev["end_sample"] = max(prev["end_sample"], seg["end_sample"])
                        prev["peak_sample"] = (prev["start_sample"] + prev["end_sample"]) // 2
                        if seg["end_sample"] - seg["start_sample"] >= min_samples:
                            prev["_openness"] = seg["_openness"]
                        continue
                    segments.append(seg)

                # 协同起止等级：与相邻段的开合度取中点，口型衔接平滑而非
                # 恒定回落到固定值
                opens = [s["_openness"] for s in segments]
                for i, seg in enumerate(segments):
                    prev_open = opens[i - 1] if i > 0 else 0
                    next_open = opens[i + 1] if i < len(opens) - 1 else 0
                    seg["start_level"] = (prev_open + opens[i]) // 2
                    seg["end_level"] = (opens[i] + next_open) // 2
                    seg["peak_level"] = opens[i]
                    del seg["_openness"]
            # Compute a simple RMS energy envelope over the sentence PCM
            # so the SDK has a loudness hint per hop. Best-effort: any
            # failure leaves energy as [] which is still contract-valid.
            try:
                import struct
                for h_off in range(0, sentence_samples, energy_hop_samples):
                    h_end = min(h_off + energy_hop_samples, sentence_samples)
                    window = sentence_pcm[h_off * 2:h_end * 2]
                    if not window:
                        break
                    count = (h_end - h_off)
                    squares = sum(s * s for s in struct.unpack(f"<{count}h", window))
                    rms = (squares / count) ** 0.5 if count else 0.0
                    # normalize to [0,1] against int16 full-scale
                    energy_values.append(min(1.0, rms / 32768.0))
            except Exception:
                logger.debug("motion_energy_compute_failed", exc_info=True)
                energy_values = []

            # 5. 先发 motion_chunk（在 PCM 之前！）
            await _send_json(state, {
                "type": "motion_chunk",
                "audio_id": audio_id,
                "sentence_seq": sentence_seq,
                "chunk_seq": chunk_seq,
                "sample_rate": sample_rate,
                "valid_from_sample": total_samples_written,
                "valid_to_sample": total_samples_written + sentence_samples,
                "emotion": emotion,
                "emotion_intensity": 0.5,
                "energy_hop_samples": energy_hop_samples,
                "energy": energy_values,
                "segments": segments,
            })
            chunk_seq += 1

            # 6. 再发 PCM（分块，每块带 start_sample + sample_count）
            chunk_size = min(sentence_samples, sample_rate // 4)  # ~250ms chunks
            offset = 0
            while offset < sentence_samples:
                if (state.current_reply_task is None
                        or state.current_reply_task.cancelled()):
                    return
                end = min(offset + chunk_size, sentence_samples)
                chunk_bytes = sentence_pcm[offset * 2 : end * 2]
                this_sample_count = end - offset
                this_channels = 1
                # Gate 0A invariant: byte_length MUST equal
                # sample_count * channels * 2. Assert before sending so
                # a bug here can never ship a non-conformant reply_chunk.
                expected_bytes = this_sample_count * this_channels * 2
                if len(chunk_bytes) != expected_bytes:
                    logger.error(
                        "reply_chunk_byte_length_mismatch audio_id=%s seq=%d "
                        "got=%d expected=%d",
                        audio_id, seq, len(chunk_bytes), expected_bytes,
                    )
                    await _send_error(
                        state, "PROTOCOL_VIOLATION",
                        "reply_chunk byte_length != sample_count * channels * 2",
                        audio_id=audio_id, sentence_seq=sentence_seq, fatal=True,
                    )
                    return
                await _send_json(state, {
                    "type": "reply_chunk",
                    "seq": seq,
                    "audio_id": audio_id,
                    "sentence_seq": sentence_seq,
                    "format": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "channels": this_channels,
                    "start_sample": total_samples_written + offset,
                    "sample_count": this_sample_count,
                    "byte_length": len(chunk_bytes),
                })
                await _send_bytes(state, chunk_bytes)
                seq += 1
                offset = end

            total_samples_written += sentence_samples

            # 7. sentence_end
            await _send_json(state, {
                "type": "sentence_end",
                "audio_id": audio_id,
                "sentence_seq": sentence_seq,
                "sample_count": sentence_samples,
                "end_sample": total_samples_written,
            })
            sentence_seq += 1

        # LLM 流式
        try:
            async for token in llm_provider_service.chat_stream(
                llm_messages, provider=active_llm,
            ):
                if (state.current_reply_task is None
                        or state.current_reply_task.cancelled()):
                    return

                reply_full_text += token
                sentence_buffer += token

                # 检测句子边界
                if any(token.endswith(ch) for ch in _SENTENCE_ENDINGS):
                    sentence = sentence_buffer
                    sentence_buffer = ""
                    await _flush_sentence(sentence)
                elif len(sentence_buffer) >= _MAX_SENTENCE_CHARS:
                    # 兜底：缓冲太长强制 flush
                    sentence = sentence_buffer
                    sentence_buffer = ""
                    await _flush_sentence(sentence)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("ws_llm_stream_failed")
            await _send_error(
                state, "REPLY_STREAM_ERROR", f"reply stream error: {exc}",
                audio_id=audio_id,
            )
            return

        # flush 残余
        if sentence_buffer.strip():
            await _flush_sentence(sentence_buffer)

        # 提取最终情绪（用于 reply_end 元信息）
        _, final_emotion = extract_emotion(reply_full_text)
        # 存到 DB 的文本要去掉情绪标签
        clean_reply, _ = extract_emotion(reply_full_text)

        entitlement_service.confirm(
            user_id=state.user_id,
            key="dialogue_audio_seconds_per_month",
            request_id=usage_request_id,
            actual_quantity=max(1, (total_samples_written + output_sample_rate - 1) // output_sample_rate),
        )
        usage_confirmed = True

        # 持久化 user + assistant 消息（原子写入）
        from app.services.llm_service import SessionOwnershipError

        try:
            dialogue_service.complete_turn_and_enqueue_memory(
                session_id=state.session_id, user_id=state.user_id,
                digital_human_id=state.digital_human_id,
                user_text=user_text, assistant_text=clean_reply,
                audio_url=None,
                memory_enabled=True,
            )
        except SessionOwnershipError as exc:
            await _send_error(
                state, "SESSION_OWNERSHIP_FAILED", f"session ownership check failed: {exc}",
                audio_id=audio_id, fatal=True,
            )
            return
        try:
            dialogue_service.maybe_enqueue_session_summary(
                session_id=state.session_id, user_id=state.user_id,
                digital_human_id=state.digital_human_id,
            )
        except Exception:
            logger.exception("session_summary_enqueue_failed")

        audit_service.log(
            action="user.realtime_turn_streaming",
            actor_user_id=state.user_id,
            detail={
                "session_id": state.session_id,
                "user_text_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
                "user_text_chars": len(user_text),
                "reply_chars": len(clean_reply),
                "voice_cloned": reference.available,
                "emotion": final_emotion,
                "gpu_mode": mode,
            },
        )

        # ---- reply_end (v3: timeline already sent per-sentence via motion_chunk) ----
        await _send_json(state, {
            "type": "reply_end",
            "audio_id": audio_id,
            "reply_text": clean_reply,
            "emotion": final_emotion,
            "total_samples": total_samples_written,
            "audio_format": "pcm_s16le",
            "sample_rate": output_sample_rate,
            "channels": 1,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    except asyncio.CancelledError:
        await _send_json(state, {
            "type": "motion_interrupted",
            "audio_id": audio_id,
            "at_sample": total_samples_written,
        })
        raise
    finally:
        if usage_reserved and not usage_confirmed:
            try:
                entitlement_service.release(
                    user_id=state.user_id,
                    key="dialogue_audio_seconds_per_month",
                    request_id=usage_request_id,
                )
            except Exception:
                logger.exception("realtime_entitlement_release_failed")
        if gpu_acquired:
            try:
                await gpu_semaphore.release()
            except Exception:
                logger.debug("gpu_release_failed", exc_info=True)
        if state.current_reply_audio_id == audio_id:
            state.current_reply_audio_id = None
        if state.current_reply_task is not None and state.current_reply_task.done():
            state.current_reply_task = None


# --------------------------------------------------------------#
# ASR turn finalization (Stage B: drives the streaming session end-to-end)
# --------------------------------------------------------------#
async def _finalize_user_utterance(state: _ConnectionState) -> None:
    """Finish the ASR session, send asr_final, then kick off the reply pipeline.

    Stage B: the ASR session was created on ``start`` and fed PCM
    incrementally. ``end`` triggers session.finish() which yields the
    single final result; the partial drainer has already forwarded
    partials. Whisper batch path (no session) transcribes the buffer.
    """
    # If there is neither an active session nor buffered audio, nothing to do.
    if state.current_asr_session is None and not state.turn_buffer:
        await _send_json(state, {"type": "asr_final", "text": ""})
        return

    user_text = await _finalize_asr_turn(state)

    if not user_text:
        # asr_final already sent by _finalize_asr_turn
        return

    if state.digital_human_id:
        try:
            selected = digital_human_service.get_owned(
                state.user_id, state.digital_human_id
            )
        except DigitalHumanNotFound:
            await _send_error(
                state, "PACKAGE_NOT_READY", "digital human is unavailable",
            )
            return
        if selected.creation_status != "ready":
            await _send_error(
                state, "PACKAGE_NOT_READY", "digital human is not ready",
            )
            return

    # Cancel any previous reply (defensive — barge-in should already
    # have done this when the user started speaking).
    _cancel_current_reply(state)

    # 根据引擎 provider 选择流式 / 批量管道
    if provider_name() in {"cosyvoice", "aliyun_cosyvoice"}:
        reply_fn = _run_reply_streaming
    else:
        reply_fn = _run_reply  # legacy: fish_speech 批量
    state.current_reply_task = asyncio.create_task(reply_fn(state, user_text))


# --------------------------------------------------------------#
# WebSocket endpoint
# --------------------------------------------------------------#
@router.websocket("/stream")
async def realtime_stream(ws: WebSocket) -> None:
    """Main WS loop. Auth and digital-human selection use query parameters."""
    token = ws.query_params.get("token", "").strip()
    if not token:
        await ws.close(code=4401)
        return

    try:
        payload = decode(token)
    except JWTError:
        await ws.close(code=4401)
        return

    if payload.get("type") != "access":
        await ws.close(code=4401)
        return

    user_id = payload["sub"]
    selected_digital_human_id = (
        ws.query_params.get("digital_human_id", "").strip() or None
    )
    legacy_avatar_id: str | None = None
    if selected_digital_human_id:
        try:
            selected = digital_human_service.get_owned(
                user_id, selected_digital_human_id
            )
        except DigitalHumanNotFound:
            await ws.close(code=4404)
            return
        if selected.creation_status != "ready":
            await ws.close(code=4422)
            return
    else:
        primary = user_avatar_service.get_primary_for_user(user_id)
        if primary is not None:
            legacy_avatar_id = primary.avatar_id
            if primary.digital_human_id:
                try:
                    selected = digital_human_service.get_owned(
                        user_id, primary.digital_human_id
                    )
                    if selected.creation_status == "ready":
                        selected_digital_human_id = primary.digital_human_id
                except DigitalHumanNotFound:
                    pass

    lease_id: str | None = None
    try:
        membership = entitlement_service.get(user_id)
        concurrent_limit = int(
            membership.entitlements.get("concurrent_session_count", 1)
        )
        lease_id = await realtime_session_service.acquire(user_id, concurrent_limit)
    except Exception:
        logger.exception("realtime_membership_admission_failed")
        await ws.close(code=1013)
        return
    if lease_id is None:
        await ws.close(code=4429, reason="concurrent realtime session limit exceeded")
        return

    try:
        await ws.accept()
        state = _ConnectionState(
            ws=ws,
            user_id=user_id,
            digital_human_id=selected_digital_human_id,
        )

        # Create or reuse a dialogue session for this connection.
        active_llm = await asyncio.to_thread(
            memory_provider_service.get_provider_for_role, "chat"
        )
        session = dialogue_service.create_session(
            user_id=user_id,
            llm_provider_id=active_llm.provider_id if active_llm else None,
            avatar_id=selected_digital_human_id or legacy_avatar_id,
            digital_human_id=selected_digital_human_id,
            title="实时对话",
        )
        state.session_id = session.session_id
        await _send_json(state, {
            "type": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "session_id": state.session_id,
            "user_id": user_id,
            "digital_human_id": state.digital_human_id,
            "plan_code": membership.plan_code,
            "audio_input": {
                "format": READY_AUDIO_INPUT_FORMAT,
                "sample_rate": READY_AUDIO_INPUT_SAMPLE_RATE,
                "channels": READY_AUDIO_INPUT_CHANNELS,
            },
            "renderer": {
                "engine": READY_RENDERER_ENGINE,
                "schema_revision": READY_RENDERER_SCHEMA_REVISION,
            },
            "heartbeat_interval_ms": READY_HEARTBEAT_INTERVAL_MS,
        })
    except Exception:
        logger.exception("realtime_session_initialization_failed")
        try:
            await realtime_session_service.release(user_id, lease_id)
        except Exception:
            logger.exception("realtime_session_lease_release_failed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
        return

    async def _lease_heartbeat() -> None:
        while True:
            await asyncio.sleep(30)
            renewed = await realtime_session_service.heartbeat(user_id, lease_id)
            if not renewed:
                await ws.close(code=1013, reason="realtime session lease expired")
                return

    lease_heartbeat_task = asyncio.create_task(_lease_heartbeat())

    try:
        while True:
            msg = await ws.receive()
            renewed = await realtime_session_service.heartbeat(user_id, lease_id)
            if not renewed:
                await ws.close(code=1013, reason="realtime session lease expired")
                return

            # ---- Binary frame: audio chunk ----
            if "bytes" in msg and msg["bytes"] is not None:
                frame = msg["bytes"]
                # Stage B: feed PCM to the streaming ASR session incrementally.
                # Reject odd-byte frames (incomplete s16le sample) and oversized
                # frames (memory-attack guard).
                if len(frame) % 2 != 0:
                    await _send_error(
                        state, "PROTOCOL_VIOLATION",
                        "inbound PCM frame has odd byte count (not whole samples)",
                    )
                    continue
                if len(frame) > 64 * 1024:  # 64KB per frame cap (~0.5s at 16k mono)
                    await _send_error(
                        state, "PROTOCOL_VIOLATION",
                        f"inbound PCM frame too large: {len(frame)} bytes",
                    )
                    continue
                state.turn_buffer.extend(frame)
                state.current_input_sample_count += len(frame) // 2
                session = state.current_asr_session
                if session is not None:
                    try:
                        await session.send_pcm(bytes(frame))
                    except Exception as exc:
                        logger.debug("asr_send_pcm_failed", exc_info=True)
                        await _send_error(
                            state, "ASR_FAILED", f"ASR send_pcm failed: {exc}",
                        )
                        await _cancel_asr_turn(state)
                continue

            # ---- Text frame: JSON control ----
            text = msg.get("text")
            if text is None:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await _send_error(state, "INVALID_JSON", "inbound text frame was not valid JSON")
                continue

            mtype = data.get("type")

            if mtype == "ping":
                await _send_json(state, {"type": "pong"})
                continue

            if mtype == "interrupt":
                # Barge-in: cancel current reply AND any in-flight ASR turn,
                # then drop partial audio. User wanted to interrupt.
                _cancel_current_reply(state)
                await _cancel_asr_turn(state)
                continue

            if mtype == "end":
                # User explicitly ended their utterance.
                await _finalize_user_utterance(state)
                continue

            if mtype == "start":
                # Beginning of a new utterance. If a reply was in flight
                # this counts as barge-in: cancel reply + start fresh ASR.
                _cancel_current_reply(state)
                await _start_asr_turn(state)
                continue

            if mtype == "close":
                # Graceful client-initiated close.
                try:
                    await ws.close(code=1000)
                except Exception:
                    pass
                return

            await _send_error(
                state, "UNKNOWN_MESSAGE_TYPE", f"unknown message type: {mtype!r}",
            )

    except WebSocketDisconnect:
        logger.info("ws_disconnect user_id=%s session_id=%s", user_id, state.session_id)
    except Exception:
        logger.exception("ws_loop_crashed")
    finally:
        lease_heartbeat_task.cancel()
        try:
            await lease_heartbeat_task
        except asyncio.CancelledError:
            pass
        # Tear down any in-flight reply on disconnect.
        _cancel_current_reply(state)
        if state.current_reply_task is not None:
            try:
                await state.current_reply_task
            except asyncio.CancelledError:
                pass
        # Stage B: tear down the ASR session too.
        try:
            await _cancel_asr_turn(state)
        except Exception:
            logger.debug("asr_session_teardown_failed", exc_info=True)
        try:
            await realtime_session_service.release(user_id, lease_id)
        except Exception:
            logger.exception("realtime_session_lease_release_failed")


__all__ = ["router"]
