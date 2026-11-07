"""NSFW 审查守卫：用户上传照片唯一入口的内容安全门（NudeNet）。

接入位置：photo-preflight（照片单入口后，一个卡点覆盖全部用户素材）。
检测器 NudeNet（ONNX，CPU 可跑，纯离线——人脸照片不出机房，隐私上
优于云端审查 API）；懒加载导入，未安装时按模式 fail-closed。

判定：仅 *_EXPOSED 显式标签计为违规，score ≥ 阈值（默认 0.45）拒绝；
COVERED 类（泳装等）不拦。阈值可配置。

拒绝处理（路由层）：422 通用文案（不回显标签，避免对抗性试探）+
content_safety_cases 台账（不存原图，metadata 仅标签/分数/sha——
符合 ContentSafetyService 的禁止键约束）。

未覆盖面（后续）：语音内容、运行时 ASR/LLM 文本敏感词——文本面
计划用本地词库或云内容安全 API，见方案待办。
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# NudeNet 显式标签（*_EXPOSED 才拦；COVERED 不拦）
_EXPLICIT_LABELS: frozenset[str] = frozenset({
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED_AND_MENSTRUATING",
})


class NsfwGuardUnavailable(RuntimeError):
    """检测器不可用（未安装/模型缺失）——enforce 模式下上传必须拒绝。"""


@dataclass
class NsfwCheckResult:
    passed: bool
    labels: list[dict] = field(default_factory=list)
    max_explicit_score: float = 0.0


class NsfwGuard:
    """照片 NSFW 检测。detector 可注入（测试用桩），默认懒加载 NudeNet。"""

    def __init__(self, detector=None, threshold: float | None = None):
        self._detector = detector
        self._threshold_override = threshold

    @property
    def _threshold(self) -> float:
        if self._threshold_override is not None:
            return self._threshold_override
        from app.core.config import settings

        return settings.nsfw_explicit_threshold

    def _load_detector(self):
        if self._detector is not None:
            return self._detector
        try:
            from nudenet import NudeDetector  # noqa: PLC0415（懒加载必需）
        except ImportError as exc:
            raise NsfwGuardUnavailable(
                "nudenet is not installed; image NSFW gate cannot run"
            ) from exc
        try:
            self._detector = NudeDetector()
        except Exception as exc:  # 模型文件缺失/损坏
            raise NsfwGuardUnavailable(f"NudeDetector init failed: {exc}") from exc
        return self._detector

    def check_photo(self, photo_bytes: bytes) -> NsfwCheckResult:
        if not photo_bytes:
            raise NsfwGuardUnavailable("empty photo payload")
        detector = self._load_detector()

        # NudeDetector 吃文件路径；临时文件即写即删，不落盘留存
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(photo_bytes)
            path = Path(tf.name)
        try:
            try:
                detections = detector.detect(str(path))
            except Exception as exc:
                raise NsfwGuardUnavailable(f"nsfw detection failed: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)

        labels: list[dict] = []
        max_explicit = 0.0
        for det in detections or []:
            label = str(det.get("class") or det.get("label") or "")
            score = float(det.get("score") or 0.0)
            if label in _EXPLICIT_LABELS and score >= self._threshold:
                labels.append({"label": label, "score": round(score, 3)})
                max_explicit = max(max_explicit, score)
        return NsfwCheckResult(
            passed=not labels,
            labels=sorted(labels, key=lambda d: -d["score"]),
            max_explicit_score=round(max_explicit, 3),
        )


_guard_singleton: NsfwGuard | None = None


def get_nsfw_guard() -> NsfwGuard | None:
    """按配置返回守卫；mode=disabled 返回 None（明确关闭而非静默缺失）。"""
    global _guard_singleton
    from app.core.config import settings

    if (settings.nsfw_guard_mode or "enforce").lower() == "disabled":
        return None
    if _guard_singleton is None:
        _guard_singleton = NsfwGuard()
    return _guard_singleton
