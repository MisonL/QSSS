"""Cache failure contract tests."""

import pytest

from qsss.cache.manager import CacheDeserializationError, CacheManager
from qsss.distributed.cache import (
    CacheDeserializationError as DistributedCacheDeserializationError,
)
from qsss.distributed.cache import (
    DistributedCache,
)


class _BrokenRedis:
    def get(self, cache_key):
        return b"\xff\xfe\x00broken"


class _BrokenWritableRedis(_BrokenRedis):
    def __init__(self):
        self.writes = []

    def setex(self, cache_key, ttl, value):
        self.writes.append((cache_key, ttl, value))
        return True


def test_cache_manager_raises_on_corrupted_redis_value():
    assert DistributedCacheDeserializationError is CacheDeserializationError
    cache = CacheManager(redis_client=_BrokenRedis())

    with pytest.raises(CacheDeserializationError):
        cache.get("corrupted")


def test_distributed_cache_raises_on_corrupted_redis_value():
    cache = DistributedCache(redis_client=_BrokenRedis())

    with pytest.raises(DistributedCacheDeserializationError):
        cache.get("corrupted")


def test_cache_memoize_treats_corrupted_cached_value_as_miss():
    redis_client = _BrokenWritableRedis()
    cache = CacheManager(redis_client=redis_client)
    calls = {"count": 0}

    @cache.cache_memoize(ttl=60, prefix="memoize")
    def compute_value():
        calls["count"] += 1
        return {"ok": True}

    assert compute_value() == {"ok": True}
    assert calls["count"] == 1
    assert redis_client.writes
