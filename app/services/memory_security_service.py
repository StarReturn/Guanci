"""Sensitive-memory detection, credential rejection and prompt-injection defense.

This module is the single chokepoint that decides:

* whether raw candidate text gets classified ``sensitive`` (and therefore
  encrypted, never indexed, never auto-recalled); or
* whether it is rejected outright as a credential / verification code that
  must never be persisted in any form; or
* whether it is normal text that flows into the plaintext retrieval path.

It also exposes ``sanitize_for_prompt()`` which wraps any untrusted text
(remembered summary, relationship profile) in a delimited, escaped envelope
so a malicious memory cannot override the system prompt or trigger tool
calls.

The patterns are deliberately conservative — when in doubt, classify as
sensitive. The cost of false-positive sensitive is a slight retrieval
recall hit on that single row; the cost of false-negative sensitive is a
privacy leak. So the regex list is reviewed on every change.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Credential / verification / PII patterns
# ---------------------------------------------------------------------------
# Each pattern is anchored on shape rather than keyword so we catch both
# Chinese and English inputs. The redactor returns ``None`` to signal
# "reject this candidate entirely" (do not persist in any form).
_HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 密码 / 支付密码 / 验证码 / 安全码 / CVV + 4..64 chars
    re.compile(
        r"(?:密码|支付密码|登录密码|安全码|验证码|动态码|短信码|CVV|CVC|PIN)\s*[:：是为]?\s*[^\s，。；,;]{4,64}",
        re.I,
    ),
    # 身份证号
    re.compile(r"(?:身份证|证件号|身份号|ID\s*number)\s*[:：是为]?\s*[0-9Xx]{8,18}"),
    # 银行卡 / 信用卡（12..19 digits）
    re.compile(r"(?:银行卡|信用卡|储蓄卡|卡号|账号)\s*[:：是为]?\s*\d{12,19}"),
    # private key / PEM blobs
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----", re.I),
    # bearer tokens / long alnum tokens prefixed by token-like keywords
    re.compile(
        r"(?:access[_ ]?token|api[_ ]?key|bearer|authorization|secret|私钥|密钥)\s*[:：=]?\s*[A-Za-z0-9_\-]{16,}",
        re.I,
    ),
    # JWT shape (three base64 segments)
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
)

# Sensitive-topic classifier. Hits are encrypted + isolated, but not rejected
# outright. Tuned for Chinese conversational content; English equivalents
# appended as plain substrings so the regex stays readable.
_SENSITIVE_PATTERN = re.compile(
    "癌|肿瘤|白血病|重病|绝症|艾滋病|HIV|精神分裂|抑郁|焦虑|自杀|自残|"
    "住院|手术|化疗|放疗|ICU|"
    "死亡|去世|离世|死因|"
    "家暴|虐待|性侵|离婚|出轨|婚变|"
    "破产|欠债|高利贷|"
    "病历|诊断|病情"
)

# Prompt-injection patterns. Hits are kept as data (we still store the memory)
# but the prompt builder escapes the boundary markers so they cannot break
# out of the bounded envelope.
_INJECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:system|assistant|user|tool)\s*[:：]",
    re.I,
)
_INJECTION_INSTRUCTION_PATTERN = re.compile(
    r"(忽略|无视|跳过|忽略以上|忽略前面|忽略上面|"
    r"ignore (?:all |previous |above |)instructions|"
    r"pretend you are|act as|you are now|从现在起你是|"
    r"</?system>|</?prompt>|</?instruction>)",
    re.I,
)


@dataclass(frozen=True)
class Classification:
    """Decision returned by ``classify``."""

    sensitivity: str  # "normal" | "sensitive"
    reject_reason: str | None  # set when the text must NOT be persisted at all
    has_injection_marker: bool


def contains_high_risk_credentials(text: str) -> str | None:
    """Return the matched substring if ``text`` carries verbatim credentials.

    Caller MUST drop the candidate entirely. Storing a redacted version is
    NOT acceptable because the original transcript might already be cached
    upstream; we treat any credential leak as a hard stop.
    """
    if not text:
        return None
    for pattern in _HIGH_RISK_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def looks_sensitive(text: str) -> bool:
    if not text:
        return False
    return bool(_SENSITIVE_PATTERN.search(text))


def has_prompt_injection_marker(text: str) -> bool:
    if not text:
        return False
    return bool(_INJECTION_PATTERN.search(text) or _INJECTION_INSTRUCTION_PATTERN.search(text))


def classify(text: str, *, category: str = "") -> Classification:
    """Classify raw candidate text.

    Decision rules:

    1. ``contains_high_risk_credentials`` -> reject (must not persist).
    2. ``looks_sensitive`` -> sensitive (encrypted, not indexed/recalled).
    3. ``category == 'health'`` -> sensitive (defensive default).
    4. otherwise normal.
    """
    if not text or not text.strip():
        return Classification(
            sensitivity="normal",
            reject_reason="empty",
            has_injection_marker=False,
        )
    if contains_high_risk_credentials(text):
        return Classification(
            sensitivity="sensitive",
            reject_reason="credential",
            has_injection_marker=has_prompt_injection_marker(text),
        )
    if looks_sensitive(text) or (category or "").lower() == "health":
        return Classification(
            sensitivity="sensitive",
            reject_reason=None,
            has_injection_marker=has_prompt_injection_marker(text),
        )
    return Classification(
        sensitivity="normal",
        reject_reason=None,
        has_injection_marker=has_prompt_injection_marker(text),
    )


def sanitize_for_prompt(untrusted: str, *, label: str) -> str:
    """Render untrusted memory text so it cannot break out of the envelope.

    Steps:

    * HTML-escape (also neutralises ``</system>`` style markers).
    * Replace newlines with spaces so a "system:" prefix on its own line
      cannot be smuggled.
    * Strip ASCII control chars.
    * Wrap in ``<<<label>>> ... <<</label>>>>`` so the model sees an
      explicit boundary, and the system prompt explicitly tells it these
      blocks are reference data, not instructions.
    """
    if untrusted is None:
        return ""
    flattened = untrusted.replace("\r", " ").replace("\n", " ")
    flattened = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", flattened)
    escaped = html.escape(flattened, quote=False)
    # Remove any stray delimiter the user might have typed.
    safe_label = re.sub(r"[^A-Za-z0-9_]", "", label) or "memory"
    return f"<<<{safe_label}>>> {escaped} <<<{safe_label}_end>>>"


__all__ = [
    "Classification",
    "classify",
    "contains_high_risk_credentials",
    "has_prompt_injection_marker",
    "looks_sensitive",
    "sanitize_for_prompt",
]
