"""dhlive-v1 服务端参考渲染门（方案 §9.3 / A2）。

全自动服务端渲染验收：不信任 manifest 声明，直接解码包内视频帧并与几何
数据交叉验证。固定检查项：

- 黑屏/空帧亮度门（采样帧亮度与方差）
- 循环接缝门：往返契约下视频帧 0 与 N-1、1 与 N-2 为同源镜像对，
  接缝平均像素差必须低于阈值（无跳变）
- 几何越界门：采样帧对应的 json_data rect 必须落在视频画幅内
- 工件归档：采样帧拼成 contact sheet JPEG，作为 reference render 产物
  上传存储（activate_validated_dhlive_package 的
  reference_render_storage_key）

WASM 全保真渲染验收在端侧（SDK DhLivePackageValidator + 样本 App），
服务端门做内容级 fail-closed；两层互不替代。

依赖 ffmpeg/ffprobe 二进制（与构建链同工具链），不引入 cv2。
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

MIN_LUMA = 12.0          # 黑屏门：帧平均亮度下限（0-255）
MIN_STD = 8.0            # 空帧/纯色块门：帧像素标准差下限
SEAM_MAX_DIFF = 16.0     # 循环接缝门：镜像对平均绝对像素差上限（0-255）
SAMPLE_FRACTIONS = (0.05, 0.3, 0.55, 0.8)


class ReferenceRenderError(ValueError):
    """参考渲染失败（输入无法解码/结构损坏）。"""


@dataclass
class ReferenceRenderResult:
    report: dict
    artifact_jpeg: bytes


def ffprobe_json(video_path: Path) -> dict:
    """ffprobe 首视频流元数据（streams[0] + format.duration）。"""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames:format=duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise ReferenceRenderError(f"ffprobe failed: {proc.stderr[:200]}")
    return json.loads(proc.stdout)


def _extract_frames(video_path: Path, indices: list[int], out_dir: Path) -> dict[int, Path]:
    """按帧号精确抽取（select 过滤器），返回 {帧号: PNG 路径}。"""
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vf", f"select='{expr}'", "-fps_mode", "passthrough",
            "-frames:v", str(len(indices)),
            str(out_dir / "frame_%03d.png"),
        ],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise ReferenceRenderError(f"frame extraction failed: {proc.stderr[:200]}")
    files = sorted(out_dir.glob("frame_*.png"))
    if len(files) != len(indices):
        raise ReferenceRenderError(
            f"expected {len(indices)} frames, got {len(files)}"
        )
    return dict(zip(indices, files))


def _frame_stats(png_path: Path) -> tuple[float, float]:
    """(平均亮度, 像素标准差) — PIL 灰度。"""
    from PIL import Image
    import numpy as np

    with Image.open(png_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    return float(gray.mean()), float(gray.std())


def _mean_abs_diff(a: Path, b: Path) -> float:
    from PIL import Image
    import numpy as np

    ga = np.asarray(Image.open(a).convert("L"), dtype=np.float32)
    gb = np.asarray(Image.open(b).convert("L"), dtype=np.float32)
    if ga.shape != gb.shape:
        return 255.0
    return float(np.abs(ga - gb).mean())


def _contact_sheet(paths: list[Path]) -> bytes:
    """等高横排拼图 → JPEG 工件。"""
    from PIL import Image
    import io as _io

    height = 240
    imgs = []
    for p in paths:
        with Image.open(p) as img:
            ratio = height / img.height
            imgs.append(img.resize((max(1, int(img.width * ratio)), height)))
    total_w = sum(i.width for i in imgs)
    sheet = Image.new("RGB", (total_w, height), "black")
    x = 0
    for i in imgs:
        sheet.paste(i, (x, 0))
        x += i.width
    buf = _io.BytesIO()
    sheet.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def render_reference(package_bytes: bytes) -> ReferenceRenderResult:
    """对 dhlive-v1 素材包执行参考渲染门。fail-closed：任何一步不可判
    定即失败（report.passed=False 的 ReferenceRenderError 由调用方决定
    404/500 语义；本函数抛异常=结构损坏，返回 passed=False=质量不过）。"""
    with tempfile.TemporaryDirectory(prefix="dhlive_ref_") as td:
        tdp = Path(td)
        zip_path = tdp / "package.zip"
        zip_path.write_bytes(package_bytes)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tdp / "pkg")
        except zipfile.BadZipFile as exc:
            raise ReferenceRenderError(f"not a zip: {exc}") from exc

        manifest_path = tdp / "pkg" / "manifest.json"
        video_path = tdp / "pkg" / "video" / "01.mp4"
        geo_path = tdp / "pkg" / "data" / "combined_data.json.gz"
        if not (manifest_path.is_file() and video_path.is_file() and geo_path.is_file()):
            raise ReferenceRenderError("package layout is incomplete")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            geometry = json.loads(gzip.decompress(geo_path.read_bytes()))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceRenderError(f"geometry unreadable: {exc}") from exc

        stream = ffprobe_json(video_path)
        vs = (stream.get("streams") or [{}])[0]
        width, height = int(vs.get("width") or 0), int(vs.get("height") or 0)
        nb_frames = int(manifest["video"]["frame_count"])
        g_count = int(manifest["geometry"]["frame_count"])
        json_data = geometry.get("json_data") or []
        if not width or not height or nb_frames < 2 or g_count < 1:
            raise ReferenceRenderError("video metadata incomplete")
        if len(json_data) != g_count:
            raise ReferenceRenderError(
                f"json_data len {len(json_data)} != geometry frame_count {g_count}"
            )

        # 采样：常规点 + 接缝镜像对（首2帧 与 末2帧）
        sampled = sorted({max(0, int(nb_frames * f)) for f in SAMPLE_FRACTIONS})
        seam_pairs = [(0, nb_frames - 1), (1, nb_frames - 2)]
        indices = sorted({i for pair in seam_pairs for i in pair} | set(sampled))

        frames = _extract_frames(video_path, indices, tdp)
        stats = {i: _frame_stats(frames[i]) for i in indices}

        checks: dict[str, bool] = {}
        checks["no_black_frame"] = all(
            stats[i][0] >= MIN_LUMA for i in indices
        )
        checks["no_flat_frame"] = all(stats[i][1] >= MIN_STD for i in indices)
        seam_diffs = [
            round(_mean_abs_diff(frames[a], frames[b]), 2) for a, b in seam_pairs
        ]
        checks["loop_seam_continuous"] = all(d <= SEAM_MAX_DIFF for d in seam_diffs)

        # 几何越界：视频帧 j 的几何索引映射（往返契约：j<g 正放，否则镜像）
        geometry_ok = True
        for j in indices:
            gj = j if j < g_count else (2 * g_count - 1 - j)
            if not (0 <= gj < len(json_data)):
                geometry_ok = False
                break
            rect = (json_data[gj] or {}).get("rect") or []
            if len(rect) != 4:
                geometry_ok = False
                break
            x0, y0, x1, y1 = rect
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
                geometry_ok = False
                break
        checks["geometry_in_bounds"] = geometry_ok

        artifact = _contact_sheet([frames[i] for i in indices])
        report = {
            "passed": all(checks.values()),
            "engine": "dhlive_reference_render",
            "version": 1,
            "video": {
                "width": width, "height": height,
                "frame_count": nb_frames, "geometry_frame_count": g_count,
            },
            "frames_sampled": indices,
            "luminance": {str(i): round(stats[i][0], 1) for i in indices},
            "frame_std": {str(i): round(stats[i][1], 1) for i in indices},
            "seam_pair_diff": seam_diffs,
            "checks": checks,
            "artifact": {"format": "jpeg", "bytes": len(artifact)},
        }
        return ReferenceRenderResult(report=report, artifact_jpeg=artifact)
