"""开源版配额桩（Guanci 发布版）。

完整版的对话/构建配额由会员体系（membership entitlement ledger）管理，
该体系不属于本开源仓库。此桩提供同名接口：一律放行、无记账。
生产部署请自行实现配额/风控，或联系作者获取完整版。

接口契约与完整版一致：
- reserve(user_id, key, quantity, request_id, ...) -> reservation（恒成功）
- check_limit(user_id, key, quantity) -> None（恒通过，超限抛异常的语义不存在）
- confirm(user_id, key, request_id, actual_quantity) -> 空操作
- release(user_id, key, request_id) -> 空操作
- release_stale_reservations() -> 空操作
- get(user_id) -> MinimalEntitlements（全量默认值）
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EntitlementDenied(RuntimeError):
    """开源版恒不抛出；保留以兼容调用方的 except 分支。"""


class MembershipValidationError(RuntimeError):
    """占位（开源版恒不抛出）。"""


class MembershipError(RuntimeError):
    """完整版会员域错误的占位；开源版恒不抛出，保留以兼容 except 分支。"""


class MinimalEntitlements:
    """get() 返回对象的最小实现：常用键给全量默认，其余属性兜底。"""

    _defaults = {
        "dialogue_audio_seconds_per_month": 10**9,
        "tts_characters_per_month": 10**9,
        "cross_session_memory": True,
        "memory_items_per_avatar": 10**6,
        "memory_retention_days": 10**6,
        "storage_bytes": 10**12,
        "digital_humans": 10**6,
    }

    def __getattr__(self, name: str) -> Any:
        return self._defaults.get(name, 10**9)


class EntitlementServiceStub:
    def reserve(self, **kwargs) -> dict:
        return {"request_id": kwargs.get("request_id"), "stub": True}

    def check_limit(self, *args, **kwargs) -> None:
        return None

    def confirm(self, *args, **kwargs) -> None:
        return None

    def release(self, *args, **kwargs) -> None:
        return None

    def release_stale_reservations(self, *args, **kwargs) -> None:
        return None

    def get(self, user_id: str) -> MinimalEntitlements:
        return MinimalEntitlements()


entitlement_service = EntitlementServiceStub()


class EffectiveMembership:
    """完整版 EffectiveMembershipSnapshot 的开源占位：全量默认权益。"""

    def __init__(self, *args, **kwargs) -> None:
        self.entitlements = MinimalEntitlements()

    def __getattr__(self, name: str):
        return getattr(self.entitlements, name)
