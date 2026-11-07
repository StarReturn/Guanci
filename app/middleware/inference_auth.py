"""推理容器 API 认证中间件。

业务机调用推理机时带 Authorization: Bearer <token>。
Token 通过环境变量 INFERENCE_API_TOKEN 共享。
未设置 token 时跳过校验（开发模式）。
"""
from __future__ import annotations

import os
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class InferenceAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = os.environ.get("INFERENCE_API_TOKEN", "")
        if not token:
            # 开发模式：未配置 token 则不校验
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {token}":
            return await call_next(request)

        # 健康检查端点免认证
        if request.url.path in ("/health", "/healthz", "/"):
            return await call_next(request)

        return Response(
            content='{"detail":"unauthorized"}',
            status_code=401,
            media_type="application/json",
        )
