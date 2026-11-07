"""API 路由装配（Guanci 开源版）。

完整版还包含 admin 管理面、支付（payments）、会员（membership）、
多账号矩阵（admin_dashscope_accounts）等运营与商业化模块——均不在
本仓库分发范围内，故此处仅挂载客户端路由。
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import (
    appmeta,
    auth,
    backgrounds,
    dialogue,
    dhlive_build,
    dhlive_chat,
    dhlive_sharing,
    digital_humans,
    health,
    keyframes,
    llm_providers,
    memories,
    phonemes,
    preset_voices,
    realtime,
    realtime_ws,
    tts,
    tts_providers,
    users,
)
from app.api.v2.routes import packages as packages_v2
from app.api.v2.routes import realtime_ws as realtime_ws_v2

client_router = APIRouter()

client_router.include_router(auth.router, prefix="/auth", tags=["client/auth"])
client_router.include_router(auth.me_router, prefix="/me", tags=["client/me"])
client_router.include_router(users.router, prefix="/users", tags=["client/users"])
client_router.include_router(
    digital_humans.router,
    prefix="/me/digital-humans",
    tags=["client/digital-humans"],
)
client_router.include_router(
    dhlive_build.router,
    prefix="/me/digital-humans",
    tags=["client/dhlive-build"],
)
client_router.include_router(
    dhlive_chat.router,
    prefix="/me/digital-humans",
    tags=["client/dhlive-chat"],
)
client_router.include_router(
    dhlive_sharing.router,
    prefix="/me/sharing",
    tags=["client/sharing"],
)
client_router.include_router(
    preset_voices.router, prefix="/preset-voices", tags=["client/preset-voices"]
)
client_router.include_router(dialogue.router, prefix="/dialogue", tags=["client/dialogue"])
client_router.include_router(
    realtime.router, prefix="/realtime", tags=["client/realtime"]
)
client_router.include_router(
    realtime_ws.router, prefix="/realtime", tags=["client/realtime-ws"]
)
client_router.include_router(
    memories.router,
    prefix="/me/digital-humans",
    tags=["client/memories"],
)
client_router.include_router(
    backgrounds.router, prefix="/me/backgrounds", tags=["client/backgrounds"]
)
client_router.include_router(
    keyframes.router, prefix="/keyframes", tags=["client/keyframes"]
)
client_router.include_router(tts.router, prefix="/tts", tags=["client/tts"])
client_router.include_router(phonemes.router, prefix="/phonemes", tags=["client/phonemes"])
client_router.include_router(
    tts_providers.router, prefix="/tts-providers", tags=["client/tts-providers"]
)
client_router.include_router(
    llm_providers.router, prefix="/llm-providers", tags=["client/llm-providers"]
)
client_router.include_router(appmeta.router, prefix="/app", tags=["client/app"])
client_router.include_router(health.router, prefix="/health", tags=["client/health"])

client_v2_router = APIRouter()
client_v2_router.include_router(packages_v2.router, tags=["client-v2/packages"])
client_v2_router.include_router(realtime_ws_v2.router, tags=["client-v2/realtime-ws"])

__all__ = ["client_router", "client_v2_router"]
