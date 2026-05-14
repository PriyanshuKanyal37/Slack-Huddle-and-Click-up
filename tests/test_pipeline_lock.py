import asyncio

from main import acquire_pipeline_lock, release_pipeline_lock


class FakeRedis:
    def __init__(self, set_result=True):
        self.set_result = set_result
        self.set_calls = []
        self.deleted = []

    async def set(self, key, value, nx=None, ex=None):
        self.set_calls.append({"key": key, "value": value, "nx": nx, "ex": ex})
        return self.set_result

    async def delete(self, *keys):
        self.deleted.extend(keys)


def test_acquire_pipeline_lock_uses_atomic_redis_set_nx():
    fake = FakeRedis(set_result=True)

    acquired = asyncio.run(acquire_pipeline_lock("bot-1", redis_client=fake))

    assert acquired is True
    assert fake.set_calls == [
        {"key": "bot_lock:bot-1", "value": "1", "nx": True, "ex": 21600}
    ]


def test_acquire_pipeline_lock_returns_false_when_existing_lock_present():
    fake = FakeRedis(set_result=None)

    acquired = asyncio.run(acquire_pipeline_lock("bot-1", redis_client=fake))

    assert acquired is False


def test_release_pipeline_lock_deletes_lock_key():
    fake = FakeRedis()

    asyncio.run(release_pipeline_lock("bot-1", redis_client=fake))

    assert fake.deleted == ["bot_lock:bot-1"]
