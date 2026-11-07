import redis.asyncio as redis

from app.core.config import settings


class RedisClient:
    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(settings.redis_url, decode_responses=True)
        return self._client

    async def health_check(self) -> dict:
        try:
            pong = await self.client.ping()
            return {"status": "ok" if pong else "error"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


redis_client = RedisClient()
