"""WS 音频链路的 Opus 编解码封装（带宽扩容方案 §2.3）。

- 绑定：opuslib（ctypes 调系统 libopus，无子进程；Dockerfile 需 libopus0）
- 参数：16 kHz / 单声道 / 24 kbps / 20 ms 帧（960 样本）
- 线程模型：每 WS 连接独立实例（libopus 非线程安全）；单帧编解码 <0.1 ms，
  可直接在事件循环内同步调用
- 环境未安装 opuslib/libopus 时模块仍可导入：协商自动回退 PCM（见
  realtime_ws._negotiate_audio_codec）

帧契约（连接级协商为 opus 后）：
- 上行 binary：[0x01][单 Opus packet]；JSON 头 sample_count = 解码后样本数
- 下行 binary：同格式；不足一帧的尾样本缓存到下一包，轮次结束补静音凑帧
"""
from __future__ import annotations

from dataclasses import dataclass, field

FRAME_SAMPLES = 960          # 20 ms @ 16 kHz
SAMPLE_RATE = 16000
CHANNELS = 1
BITRATE = 24000
OPUS_FRAME_PREFIX = b"\x01"  # binary 帧类型标记（pcm 模式无前缀）


class OpusUnavailable(RuntimeError):
    """运行环境缺少 opuslib/libopus。"""


try:  # pragma: no cover - 环境相关
    import opuslib

    _OPUS_READY = True
except Exception:  # noqa: BLE001 - ImportError / OSError(加载 libopus) 均回退
    _OPUS_READY = False


def available() -> bool:
    return _OPUS_READY


@dataclass
class OpusSession:
    """单连接编解码器（上行解码 + 下行编码）。"""

    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    bitrate: int = BITRATE
    frame_samples: int = FRAME_SAMPLES
    _tx_remainder: bytearray = field(default_factory=bytearray)
    _encoder: object | None = None
    _decoder: object | None = None

    def __post_init__(self) -> None:
        if not _OPUS_READY:
            raise OpusUnavailable("opuslib/libopus not installed")
        self._encoder = opuslib.Encoder(
            self.sample_rate, self.channels, opuslib.APPLICATION_VOIP)
        try:
            # 新版属性 setter；旧版走 encode 前的 ctl（失败则用默认码率）
            self._encoder.bitrate = self.bitrate
        except Exception:  # noqa: BLE001
            pass
        self._decoder = opuslib.Decoder(self.sample_rate, self.channels)

    # ---- 下行：PCM -> Opus packets ----

    def encode_pcm(self, pcm: bytes) -> list[bytes]:
        """整段 PCM 编码为若干 Opus packet（不足帧的余量缓存到下一段）。"""
        self._tx_remainder.extend(pcm)
        packets: list[bytes] = []
        frame_bytes = self.frame_samples * 2
        while len(self._tx_remainder) >= frame_bytes:
            frame = bytes(self._tx_remainder[:frame_bytes])
            del self._tx_remainder[:frame_bytes]
            packets.append(self._encode_frame(frame))
        return packets

    def flush_tail(self) -> list[bytes]:
        """把不足一帧的残余补静音凑满一帧编码（<20 ms 静音无感）。"""
        if not self._tx_remainder:
            return []
        frame_bytes = self.frame_samples * 2
        padded = bytes(self._tx_remainder) + b"\x00" * (
            frame_bytes - len(self._tx_remainder)
        )
        self._tx_remainder.clear()
        return [self._encode_frame(padded)]

    def _encode_frame(self, frame: bytes) -> bytes:
        return bytes(self._encoder.encode(frame, self.frame_samples))

    # ---- 上行：Opus packet -> PCM ----

    def decode_packet(self, packet: bytes) -> bytes:
        """单 packet 解码为 PCM（16k s16le）；损坏包由 opuslib 抛错。"""
        return bytes(self._decoder.decode(packet, self.frame_samples))


def samples_of_pcm(pcm: bytes) -> int:
    return len(pcm) // (CHANNELS * 2)


__all__ = [
    "FRAME_SAMPLES",
    "OPUS_FRAME_PREFIX",
    "OpusSession",
    "OpusUnavailable",
    "available",
    "samples_of_pcm",
]
