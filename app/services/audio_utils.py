from __future__ import annotations
import io
import os
import subprocess
import tempfile
import wave


def wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise ValueError("invalid wav sample rate")
        return int(frames * 1000 / rate)


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def normalize_audio_to_wav(content: bytes, content_type: str, default_sample_rate: int = 22050) -> bytes:
    """任意常见音频（mp3/m4a/aac/ogg/flac/amr/wav/裸PCM）→ 单声道 s16 WAV。

    用户上传大概率不是 WAV（微信/手机录音多为 mp3/m4a/amr），统一在入口
    归一：格式按魔数判定而非 content_type（可伪造/缺失），采样率统一到
    default_sample_rate（产品链路调用方应传 16000——16kHz 全链铁律）。

    决策顺序：
      1. RIFF 且本就是目标规格的 PCM WAV → 原样返回（零拷贝快路径）
      2. content_type 显式声明裸 PCM → 加 WAV 头（历史行为保留）
      3. 其余一切（含压缩 WAV、octet-stream）→ ffmpeg 转码，失败即抛错
    """
    # 1) 已是目标规格的 PCM WAV：零拷贝
    if content[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(content), "rb") as wf:
                if (
                    wf.getnchannels() == 1
                    and wf.getsampwidth() == 2
                    and wf.getframerate() == default_sample_rate
                ):
                    return content
        except wave.Error:
            pass  # 压缩/非标 WAV → 走 ffmpeg 转码

    # 2) 显式裸 PCM 声明（audio/L16（RFC 2586）、audio/pcm-* 等）：
    #    仅此才允许当裸流，octet-stream 一律走 ffmpeg 探测
    media_type = content_type.lower()
    if ("pcm" in media_type or "l16" in media_type) and content[:4] != b"RIFF":
        return pcm_to_wav(content, default_sample_rate)

    # 3) 万物皆可 ffmpeg：mp3(ID3/0xFFEx)、m4a(ftyp)、ogg(OggS)、
    #    flac(fLaC)、amr(#!AMR)、非目标规格的 wav、octet-stream …
    #    输出用裸 PCM（-f s16le），由我们包正确 WAV 头——
    #    ffmpeg 管道输出 wav 时 size 字段是占位符（0x7FFFFFFF），
    #    导致 wave.getnframes() 读出天文数字（实测 134217727ms）
    #    输入必须走临时文件而非 stdin：m4a/mp4 的 moov atom 在文件尾，
    #    demuxer 要 seek，stdin 管道不可 seek 会报 partial file
    #    （MediaRecorder 直录的 m4a 全部如此，2026-10-21 实测）。
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as tf:
        tf.write(content)
        src_path = tf.name
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                src_path,
                "-ac",
                "1",
                "-ar",
                str(default_sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass
    if process.returncode != 0 or not process.stdout:
        error = process.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"ffmpeg audio decode failed: {error}")
    return pcm_to_wav(process.stdout, default_sample_rate)


# ---------------------------------------------------------------------------
# 录音克隆质量门禁（2026-11：从源头杜绝低品音质进入克隆链路）
# 背景：克隆忠实复刻参考音频的一切缺陷。用户手机录音常见四类硬伤——
# 削波（电流声根源）、低信噪比（底噪被复刻）、有效语音过短、过度
# 压缩/电话级带宽（闷罐声）。席位库三类音质事故（带宽/参考削波/输出
# 削波）修复后，此门禁把同样的防线前移到用户上传侧。
# ---------------------------------------------------------------------------

def _frames_rms_static(pcm: bytes, sample_rate: int) -> list[float]:
    """20ms 帧 RMS 序列（质量体检共用）。"""
    import struct as _s

    samples = _s.unpack(f"<{len(pcm)//2}h", pcm)
    win = max(1, sample_rate // 50)
    return [(sum(x * x for x in samples[i:i + win]) / win) ** 0.5
            for i in range(0, len(samples) - win, win)]


def analyze_voice_quality(wav_bytes: bytes) -> dict:
    """克隆参考音频体检。返回指标与判定（verdict: ok/reject + 中文原因）。

    指标：
      clipped        削波样本数（|s|>=32700）——已失真不可修复，只能拒绝
      snr_db         估计信噪比（低 10% 帧均值为噪声底 vs 活跃帧均值）
      speech_ratio   活跃语音占比（帧能量 > 噪声底*3）
      hf_ratio       一阶差分能量占比（过低=电话级/闷罐带宽）
      peak           峰值（0-32768）
    阈值经席位库实测校准（真人 CV 参考的分布），可在调用方覆盖。
    """
    import math
    import struct as _s
    import wave as _w

    with _w.open(io.BytesIO(wav_bytes)) as wf:
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    samples = _s.unpack(f"<{len(pcm)//2}h", pcm)
    if len(samples) < rate:  # <1s
        return {"verdict": "reject", "reason": "录音太短，请录制 10 秒以上",
                "clipped": 0, "snr_db": 0, "speech_ratio": 0, "hf_ratio": 0,
                "peak": 0, "duration_s": len(samples) / rate if rate else 0}

    clipped = sum(1 for x in samples if abs(x) >= 32700)
    peak = max(abs(x) for x in samples)
    # SNR：20ms 帧 RMS，最低 10% 为噪声底
    rms = _frames_rms_static(pcm, rate)
    if not rms:
        return {"verdict": "reject", "reason": "音频内容为空", "clipped": clipped,
                "snr_db": 0, "speech_ratio": 0, "hf_ratio": 0, "peak": peak,
                "duration_s": len(samples) / rate}
    noise_floor = sorted(rms)[:max(1, len(rms) // 10)]
    noise = sum(noise_floor) / len(noise_floor) or 1.0
    active = [r for r in rms if r > noise * 3]
    speech_ratio = len(active) / len(rms)
    snr_db = 20 * math.log10((sum(active) / len(active)) / noise) if active else 0
    # 高频活力：一阶差分能量 / 总能量
    te = sum(float(x) * x for x in samples) or 1.0
    de = sum(float(samples[i] - samples[i - 1]) ** 2 for i in range(1, len(samples)))
    hf_ratio = de / te

    dur_s = len(samples) / rate
    m = {"clipped": clipped, "snr_db": round(snr_db, 1),
         "speech_ratio": round(speech_ratio, 2), "hf_ratio": round(hf_ratio, 4),
         "peak": peak, "duration_s": round(dur_s, 1)}
    if clipped > 30:
        m.update(verdict="reject",
                 reason=f"录音音量过大已破音（{clipped} 处削波），请离麦克风稍远、调低音量后重录")
    elif snr_db < 18:
        m.update(verdict="reject",
                 reason=f"环境噪音太大（信噪比 {snr_db:.0f}dB），请到安静房间重录")
    elif speech_ratio < 0.35 or len(active) * 0.02 < 5:
        m.update(verdict="reject",
                 reason="有效说话时长不足 5 秒，请连贯地多说一会儿（聊聊家常即可）")
    elif hf_ratio < 0.004:
        m.update(verdict="reject",
                 reason="录音音质偏低（疑似电话/聊天软件转发音频），请用手机录音 App 原生录制")
    else:
        m.update(verdict="ok", reason="ok")
    return m


def enhance_voice_wav(wav_bytes: bytes) -> bytes:
    """克隆参考自动增强（可修复项）：高通 70Hz 去隆隆/直流 + 峰值归一到 0.8。
    已削波部分无法修复——由 analyze_voice_quality 在前置环节拒绝。"""
    import struct as _s
    import wave as _w

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(wav_bytes)
        src = tf.name
    try:
        out_path = src + ".enh.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", src, "-af", "highpass=f=70", "-ar", "44100",
             "-ac", "1", "-sample_fmt", "s16", out_path],
            check=True, timeout=30)
        data = open(out_path, "rb").read()
    finally:
        for p in (src, src + ".enh.wav"):
            try:
                os.unlink(p)
            except OSError:
                pass
    # 峰值归一化到 0.8（防后续链路削波）
    with _w.open(io.BytesIO(data)) as wf:
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    samples = _s.unpack(f"<{len(pcm)//2}h", pcm)
    peak = max((abs(x) for x in samples), default=1) or 1
    # 双向归一到 0.8：过响压低防削波、过轻抬升保响度一致；上限 4 倍防放大噪声底
    gain = min(4.0, int(32767 * 0.8) / peak)
    fixed = _s.pack(f"<{len(samples)}h", *(int(x * gain) for x in samples))
    return pcm_to_wav(fixed, rate)

