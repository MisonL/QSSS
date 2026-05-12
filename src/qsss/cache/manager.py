"""高性能缓存管理器"""

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]

import hashlib
import json
import pickle
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from loguru import logger


class CacheManager:
    """高性能缓存管理器"""

    def __init__(
        self,
        redis_client: Any | None = None,
        default_ttl: int = 3600,
        max_memory_items: int = 1000,
    ) -> None:
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        # 将 redis_client 视为 Any，避免第三方类型细节干扰核心逻辑类型检查
        self.redis_client: Any | None = redis_client
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }
        self._stats_lock = threading.Lock()

        if self.redis_client is None and REDIS_AVAILABLE:
            try:
                # 一些较旧版本的 redis-py 不支持 connection_pool_kwargs 等高级参数，
                # 为了兼容性这里只使用通用参数；如需自定义连接池，可在外部构造 redis_client 传入。
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=3,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                    retry_on_timeout=True,
                )
                self.redis_client.ping()
                self.use_redis = True
                logger.info("Redis缓存连接成功")
            except Exception as e:
                logger.warning(f"Redis连接失败或不兼容当前参数: {e}，使用内存缓存")
                self.redis_client = None
                self.use_redis = False
        elif self.redis_client is not None:
            # 外部传入的 redis_client 统一视为 Any
            self.use_redis = True
        else:
            self.redis_client = None
            self.use_redis = False

    def _generate_key(self, key: str, prefix: str = "qsss") -> str:
        """生成缓存键"""
        if len(key) > 200:  # 避免键过长
            key_hash = hashlib.md5(key.encode()).hexdigest()
            key = f"{key[:100]}_{key_hash}"
        return f"{prefix}:{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """序列化值"""
        try:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Pickle序列化失败: {e}，使用JSON序列化")
            return json.dumps(value, default=str).encode("utf-8")

    def _deserialize_value(self, value: bytes) -> Any:
        """反序列化值"""
        try:
            return pickle.loads(value)
        except Exception:
            try:
                return json.loads(value.decode("utf-8"))
            except Exception as e:
                logger.error(f"反序列化失败: {e}")
                return None

    def get(self, key: str, prefix: str = "qsss") -> Optional[Any]:
        """获取缓存值"""
        try:
            cache_key = self._generate_key(key, prefix)

            if self.use_redis:
                assert self.redis_client is not None
                value = self.redis_client.get(cache_key)
                if value:
                    with self._stats_lock:
                        self._cache_stats["hits"] += 1
                    return self._deserialize_value(value)
            else:
                if cache_key in self._memory_cache:
                    item = self._memory_cache[cache_key]
                    if (
                        item.get("expire_time") is None
                        or time.time() < item["expire_time"]
                    ):
                        with self._stats_lock:
                            self._cache_stats["hits"] += 1
                        return item["value"]
                    else:
                        # 过期，删除
                        del self._memory_cache[cache_key]

            with self._stats_lock:
                self._cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None, prefix: str = "qsss"
    ) -> bool:
        """设置缓存值"""
        try:
            cache_key = self._generate_key(key, prefix)
            ttl = ttl or self.default_ttl

            if self.use_redis:
                assert self.redis_client is not None
                serialized_value = self._serialize_value(value)
                result = self.redis_client.setex(cache_key, ttl, serialized_value)
                if result:
                    with self._stats_lock:
                        self._cache_stats["sets"] += 1
                    return True
            else:
                # 内存缓存LRU策略
                if len(self._memory_cache) >= self.max_memory_items:
                    # 移除最旧的项
                    oldest_key = min(
                        self._memory_cache.keys(),
                        key=lambda k: self._memory_cache[k].get("created_at", 0),
                    )
                    del self._memory_cache[oldest_key]
                    with self._stats_lock:
                        self._cache_stats["evictions"] += 1

                self._memory_cache[cache_key] = {
                    "value": value,
                    "created_at": time.time(),
                    "expire_time": time.time() + ttl if ttl else None,
                }

                with self._stats_lock:
                    self._cache_stats["sets"] += 1
                return True

            # 走到这里说明设置失败
            return False

        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False

    def delete(self, key: str, prefix: str = "qsss") -> bool:
        """删除缓存"""
        try:
            cache_key = self._generate_key(key, prefix)

            if self.use_redis:
                assert self.redis_client is not None
                result = self.redis_client.delete(cache_key)
                if result:
                    with self._stats_lock:
                        self._cache_stats["deletes"] += 1
                    return True
            else:
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    with self._stats_lock:
                        self._cache_stats["deletes"] += 1
                    return True

            return False

        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False

    def exists(self, key: str, prefix: str = "qsss") -> bool:
        """检查缓存是否存在"""
        try:
            cache_key = self._generate_key(key, prefix)

            if self.use_redis:
                assert self.redis_client is not None
                return bool(self.redis_client.exists(cache_key))
            else:
                if cache_key in self._memory_cache:
                    item = self._memory_cache[cache_key]
                    if (
                        item.get("expire_time") is None
                        or time.time() < item["expire_time"]
                    ):
                        return True
                    else:
                        # 过期，删除
                        del self._memory_cache[cache_key]
                return False

        except Exception as e:
            logger.error(f"检查缓存存在失败: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        try:
            if self.use_redis:
                assert self.redis_client is not None
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    deleted_int = int(deleted)
                    with self._stats_lock:
                        self._cache_stats["deletes"] += deleted_int
                    return deleted_int
                # Redis 可用但没有匹配的 key
                return 0
            else:
                count = 0
                keys_to_delete = [
                    key
                    for key in self._memory_cache.keys()
                    if pattern.replace("*", "") in key
                ]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    count += 1

                with self._stats_lock:
                    self._cache_stats["deletes"] += count
                return count

        except Exception as e:
            logger.error(f"清除缓存模式失败: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_rate = self._cache_stats["hits"] / max(1, total_requests)

            stats = {
                "hits": self._cache_stats["hits"],
                "misses": self._cache_stats["misses"],
                "sets": self._cache_stats["sets"],
                "deletes": self._cache_stats["deletes"],
                "evictions": self._cache_stats["evictions"],
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "memory_items": len(self._memory_cache) if not self.use_redis else 0,
            }

            if self.use_redis and self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    stats.update(
                        {
                            "redis_memory": redis_info.get(
                                "used_memory_human", "Unknown"
                            ),
                            "redis_connections": redis_info.get("connected_clients", 0),
                            "redis_hits": redis_info.get("keyspace_hits", 0),
                            "redis_misses": redis_info.get("keyspace_misses", 0),
                            "redis_hit_rate": redis_info.get("keyspace_hits", 0)
                            / max(
                                1,
                                redis_info.get("keyspace_hits", 0)
                                + redis_info.get("keyspace_misses", 0),
                            ),
                        }
                    )
                except Exception as e:
                    logger.error(f"获取Redis统计信息失败: {e}")

            return stats

        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}

    def cache_clear(self) -> None:
        """清空所有缓存"""
        try:
            if self.use_redis:
                assert self.redis_client is not None
                self.redis_client.flushdb()
            else:
                self._memory_cache.clear()

            logger.info("缓存已清空")

        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    def cache_memoize(
        self, ttl: Optional[int] = None, prefix: str = "memoize"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """缓存装饰器"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # 生成缓存键
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

                # 尝试从缓存获取
                cached_result = self.get(cache_key, prefix)
                if cached_result is not None:
                    return cached_result

                # 执行函数
                result = func(*args, **kwargs)

                # 缓存结果
                self.set(cache_key, result, ttl, prefix)

                return result

            return wrapper

        return decorator


# 全局缓存管理器实例
cache_manager = CacheManager()
