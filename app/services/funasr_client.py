"""FunASR Paraformer streaming ASR client.

Server: 官方 docker 镜像
    registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.13
端口：10095（WebSocket）
协议：FunASR wss-server-2pass

工作流：
  1. 客户端 WS 连接到 funasr_endpoint
  2. 发送初始 config（mode=2pass, chunk_size, etc.）
  3. 持续发送 PCM16 二进制帧
  4. 服务端返回 JSON：
       {"mode":"2pass-online","text":"...","is_final":false}  partial
       {"mode":"2pass-offline","text":"...","is_final":true}   final
  5. 客户端发 {"is_end":true} 关闭

参考：https://github.com/modelscope/FunASR/blob/main/runtime/docs/websocket.md
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import websockets

from app.core.config import settings

logger = logging.getLogger(__name__)


class FunASRError(RuntimeError):
    pass


@dataclass
class ASRResult:
    """单次识别结果。"""

    text: str
    is_final: bool          # True = offline 修正后的最终结果
    mode: str               # "online"（partial）| "offline"（final）


class FunASRClient:
    """FunASR WebSocket 流式 ASR 客户端。

    用法：
        client = FunASRClient()
        async def audio_stream():
            yield pcm_chunk_1
            yield pcm_chunk_2
            ...
        async for result in client.transcribe_stream(audio_stream()):
            if result.is_final:
                # 最终结果，可以喂给 LLM
                ...
            else:
                # partial，可以下发到客户端做实时显示
                ...
    """

    def __init__(self) -> None:
        self._endpoint = settings.funasr_endpoint
        self._timeout = float(settings.funasr_timeout_seconds)

    async def health_check(self) -> dict[str, Any]:
        """FunASR 没有 HTTP health 端点，用 WS 握手是否成功判断。"""
        try:
            async with websockets.connect(
                self._endpoint,
                open_timeout=5.0,
                close_timeout=2.0,
            ) as ws:
                # 发个空 config 看是否被服务端接受（不严格，仅测连通性）
                await ws.send(json.dumps({"is_end": True}))
            return {"status": "ok", "provider": "funasr", "endpoint": self._endpoint}
        except Exception as exc:
            return {
                "status": "error",
                "provider": "funasr",
                "endpoint": self._endpoint,
                "error": str(exc),
            }

    async def transcribe_stream(
        self,
        *,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[ASRResult]:
        """流式识别。

        输入：PCM16 16kHz mono little-endian 二进制块流
        输出：ASRResult 流（先 partial，后 final）
        """
        # 解析 ws endpoint
        parsed = urlparse(self._endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise FunASRError(f"invalid funasr_endpoint: {self._endpoint!r}")
        # websockets 库支持 ws:// 和 wss://
        ws_url = self._endpoint
        if not ws_url.startswith(("ws://", "wss://")):
            ws_url = f"ws://{ws_url}"

        try:
            async with websockets.connect(
                ws_url,
                max_size=None,            # 接受任意大小响应
                open_timeout=self._timeout,
                close_timeout=5.0,
            ) as ws:
                # ---- 1. 发送初始 config ----
                config = {
                    "mode": settings.funasr_mode,                    # "2pass"
                    "chunk_size": settings.funasr_chunk_size,         # "5,10,5"
                    "chunk_interval": settings.funasr_chunk_interval,
                    "wav_name": "realtime",
                    "is_speaking": True,
                    "hotwords": "",
                    "itn": True,
                }
                await ws.send(json.dumps(config))

                # ---- 2. 并行：发音频 + 收识别结果 ----
                audio_done = asyncio.Event()
                receive_task = asyncio.create_task(
                    self._receive_loop(ws, audio_done).__anext__()  # placeholder
                )
                # 上面这行只是为了示意；实际用下面的 async for
                receive_task.cancel()

                # 启动两个任务，用 queue 桥接
                queue: asyncio.Queue[ASRResult | None] = asyncio.Queue()

                async def _send_audio():
                    try:
                        async for chunk in audio_stream:
                            # FunASR 要求 PCM16 binary
                            if isinstance(chunk, (bytes, bytearray)):
                                await ws.send(bytes(chunk))
                            # 不要 sleep，按 chunk_size 节流由服务端控制
                        # 音频发完，发结束信号
                        await ws.send(json.dumps({"is_speaking": False}))
                    except Exception as exc:
                        logger.exception("funasr_send_failed")
                        raise

                async def _recv_results():
                    try:
                        async for msg in ws:
                            if msg is None:
                                break
                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError:
                                logger.debug("funasr_skip_invalid_msg=%r", msg[:120])
                                continue
                            text = data.get("text", "").strip()
                            if not text:
                                continue
                            mode = data.get("mode", "")
                            # 2pass-offline 是 final；2pass-online 是 partial
                            is_final = mode.endswith("offline") or data.get("is_final", False)
                            await queue.put(
                                ASRResult(text=text, is_final=is_final, mode=mode)
                            )
                    except Exception as exc:
                        logger.exception("funasr_recv_failed")
                    finally:
                        await queue.put(None)  # 信号：接收结束

                sender = asyncio.create_task(_send_audio())
                receiver = asyncio.create_task(_recv_results())

                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        yield item
                finally:
                    # 取消未完成的任务
                    if not sender.done():
                        sender.cancel()
                    if not receiver.done():
                        receiver.cancel()
                    # 等待清理
                    for task in (sender, receiver):
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass

        except websockets.exceptions.InvalidHandshake as exc:
            raise FunASRError(f"FunASR WS handshake failed: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise FunASRError(f"FunASR WS connect timeout ({self._timeout}s)") from exc

    async def _receive_loop(self, ws, event):
        """占位（被 cancel），实际逻辑在 transcribe_stream 内联。"""
        yield  # pragma: no cover


funasr_client = FunASRClient()
