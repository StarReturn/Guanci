from __future__ import annotations

import time
from uuid import uuid4

from app.services.redis_client import redis_client


class RealtimeSessionService:
    _acquire_script = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local expires = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local session_id = ARGV[4]
    redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
    if redis.call('ZCARD', key) >= limit then
        return 0
    end
    redis.call('ZADD', key, expires, session_id)
    redis.call('EXPIRE', key, math.ceil((expires - now) / 1000) + 60)
    return 1
    """
    _heartbeat_script = """
    local key = KEYS[1]
    local expires = tonumber(ARGV[1])
    local session_id = ARGV[2]
    local ttl = tonumber(ARGV[3])
    if redis.call('ZSCORE', key, session_id) == false then
        return 0
    end
    redis.call('ZADD', key, expires, session_id)
    redis.call('EXPIRE', key, ttl)
    return 1
    """

    def __init__(self, lease_seconds: int = 90) -> None:
        self.lease_seconds = lease_seconds

    @staticmethod
    def _key(user_id: str) -> str:
        return f"membership:realtime_sessions:{user_id}"

    async def acquire(self, user_id: str, limit: int) -> str | None:
        session_id = str(uuid4())
        now_ms = int(time.time() * 1000)
        expires_ms = now_ms + self.lease_seconds * 1000
        accepted = await redis_client.client.eval(
            self._acquire_script,
            1,
            self._key(user_id),
            now_ms,
            expires_ms,
            limit,
            session_id,
        )
        return session_id if int(accepted) == 1 else None

    async def heartbeat(self, user_id: str, session_id: str) -> bool:
        expires_ms = int(time.time() * 1000) + self.lease_seconds * 1000
        renewed = await redis_client.client.eval(
            self._heartbeat_script,
            1,
            self._key(user_id),
            expires_ms,
            session_id,
            self.lease_seconds + 60,
        )
        return int(renewed) == 1

    async def release(self, user_id: str, session_id: str) -> None:
        await redis_client.client.zrem(self._key(user_id), session_id)


realtime_session_service = RealtimeSessionService()
