"""Emotion tag parser — extract emotion from LLM output.

让 GLM-5.1 在每句回复末尾追加 ``[EMOTION:happy|sad|...]`` 标签，
本模块解析后：
  - 清理后的文本 → 送给 TTS 合成
  - 情绪标签 → 送给在线 TTS 的情绪参数；v3 口型包保持 neutral

为什么不让 LLM 单独输出 JSON：
  - JSON 输出会拖慢 LLM 首 token（多了 schema 约束）
  - 内联标签对 LLM 更自然，质量更高
  - 解析简单（正则即可）

System prompt 示例（由 LlmProviderService 注入）：
    你是数字人助理。回复时，在最后追加一行：
    [EMOTION:happy|sad|surprised|angry|fear|neutral]

    示例：
    用户：哈哈哈太搞笑了
    你：是的我看到也笑了，这种事真的很少见呢。[EMOTION:happy]
"""
from __future__ import annotations

import re

SUPPORTED_EMOTIONS = {"neutral", "happy", "sad", "surprised", "angry", "fear"}

# 形如 [EMOTION:happy]，大小写不敏感，可能带空格；
# 兼容 LLM 偶发的变体：中文冒号（：）、全角括号（【】）
_EMOTION_PATTERN = re.compile(r"[\[【]\s*EMOTION\s*[：:]\s*(\w+)\s*[\]】]", re.IGNORECASE)
# 残片兜底：标签被分句切开（如段尾留下「[EMOTION:」或「happy]」）或使用
# 全角圆括号变体时，上面的完整匹配会漏——把任何形似情绪标签的片段从
# 送 TTS 的文本里剥掉（剥离动作不改变 emotion 判定结果）
_EMOTION_RESIDUAL = re.compile(
    r"[\[【（(]\s*EMOTION[^\]】）)]*[\]】）)]?"  # 未闭合/闭合的标签头
    r"|(?<![A-Za-z])EMOTION\s*[：:]\s*\w+"            # 裸「EMOTION:xx」
    r"|(?<![A-Za-z])happy\]|(?<![A-Za-z])sad\]|(?<![A-Za-z])warm\]",  # 掉队标签尾
    re.IGNORECASE,
)

# CosyVoice 副语言标记（<|laughter|> 等）——v2 引擎不识别会当文字读出，
# 必须在喂 TTS 前剥离（v3 支持后再恢复）
_PARALINGUISTIC_PATTERN = re.compile(r"<\|[^>]*>?", re.IGNORECASE)


def strip_paralinguistic(text: str) -> str:
    """剥离 CosyVoice 副语言标记（v2 不支持，读了会出声）。"""
    cleaned = _PARALINGUISTIC_PATTERN.sub("", text)
    # 兜底：清除残缺/未匹配到的 EMOTION 标签片段（避免被 TTS 念出来）
    cleaned = re.sub(r"[\[【]?\s*EMOTION\s*[：:]\s*\w*\s*[\]】]?", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_emotion(text: str) -> tuple[str, str]:
    """从 LLM 输出中分离文本和情绪标签。

    返回 (clean_text, emotion)。
    如果找不到标签，返回 (原文本, "neutral")。
    如果标签值不在 SUPPORTED_EMOTIONS 中，降级为 "neutral"。

    >>> extract_emotion("你好啊 [EMOTION:happy]")
    ('你好啊', 'happy')
    >>> extract_emotion("纯文本无标签")
    ('纯文本无标签', 'neutral')
    >>> extract_emotion("文本 [EMOTION:foo]")
    ('文本', 'neutral')
    """
    match = _EMOTION_PATTERN.search(text)
    if not match:
        return _EMOTION_RESIDUAL.sub("", strip_paralinguistic(text)).strip(), "neutral"

    raw_emotion = match.group(1).lower()
    emotion = raw_emotion if raw_emotion in SUPPORTED_EMOTIONS else "neutral"

    # 移除标签本身（含前后可能的空白，避免送 TTS 念出来）
    clean_text = _EMOTION_RESIDUAL.sub(
        "", strip_paralinguistic(_EMOTION_PATTERN.sub("", text).rstrip())
    ).strip()
    return clean_text, emotion


def build_emotion_system_prompt_suffix() -> str:
    """生成 system prompt 末尾的情绪标签指令。

    v2 不支持副语言标记（<|laughter|> 等会被当文字读出），也不支持
    instruction 参数（引擎 418）。当前只做情绪标签（客户端渲染用）。
    v3 升级后可恢复副语言标记。
    """
    return """

[输出格式要求]
在回复的**最末尾**追加一行情绪标签：
[EMOTION:happy|sad|surprised|angry|fear|neutral]

示例：
- "今天天气不错，适合出门走走。[EMOTION:happy]"
- "哈哈，这个确实好笑！[EMOTION:happy]"
- "我理解你的感受，别太难过了。[EMOTION:sad]"

注意：
- 情绪标签放最后
- 只能从 6 种情绪中选一个
- 不要在正文中使用任何 <|xxx|> 格式的标记
""".strip()


__all__ = [
    "extract_emotion",
    "build_emotion_system_prompt_suffix",
    "SUPPORTED_EMOTIONS",
    "EMOTION_TO_INDEXTTS_VECTOR",
    "EMOTION_TO_INDEXTTS_ALPHA",
    "build_indextts_params",
]


# LLM emotion tag → IndexTTS2 emo_vector 映射
# IndexTTS2 emo_vector 维度：[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
# 约束：sum ≤ 0.8（IndexTTS2 infer_v2.py normalize_emo_vec 强制保证）
EMOTION_TO_INDEXTTS_VECTOR: dict[str, list[float]] = {
    # 第 1 维 happy
    "happy":     [0.45, 0.0,  0.0,  0.0, 0.0, 0.0, 0.20, 0.10],  # 开心+些许惊讶
    # 第 2 维 angry
    "angry":     [0.0,  0.55, 0.0,  0.0, 0.10, 0.0, 0.10, 0.0],   # 愤怒+些许厌恶
    # 第 3 维 sad
    "sad":       [0.0,  0.0,  0.65, 0.0, 0.0, 0.10, 0.0,  0.05],  # 悲伤+忧郁
    # 第 7 维 surprised（爆发感，章北海式）
    "surprised": [0.20, 0.0,  0.0,  0.0, 0.0, 0.0, 0.50, 0.0],   # 惊讶+少许开心
    # 第 5 维 disgusted
    "fear":      [0.0,  0.0,  0.0,  0.65, 0.0, 0.10, 0.05, 0.0], # 恐惧+忧郁
    # 默认平静（低情绪强度）
    "neutral":   [0.0,  0.0,  0.0,  0.0, 0.0, 0.0,  0.0,  0.40],  # 平静主导
}

# 情绪强度（emo_alpha > 1 增强）
EMOTION_TO_INDEXTTS_ALPHA: dict[str, float] = {
    "happy":     1.3,
    "angry":     1.5,   # 愤怒要更强
    "sad":       1.2,
    "surprised": 1.5,   # 爆发感要更强
    "fear":      1.3,
    "neutral":   1.0,
}


def build_indextts_params(emotion: str) -> dict:
    """LLM emotion tag → IndexTTS2 调用参数。

    返回：{emo_vector, emo_alpha} 可直接塞进 /tts_cached 请求体。

    配合 IndexTTS2 normalize_emo_vec：
      - sum > 0.8 会自动缩放
      - emo_bias 已经在 normalize 里施加（偏向 calm/sad，降低 surprised）
    """
    vec = EMOTION_TO_INDEXTTS_VECTOR.get(emotion) or EMOTION_TO_INDEXTTS_VECTOR["neutral"]
    alpha = EMOTION_TO_INDEXTTS_ALPHA.get(emotion, 1.0)
    return {
        "emo_vector": vec,
        "emo_alpha": alpha,
    }
