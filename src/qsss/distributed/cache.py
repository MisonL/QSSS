"""分布式缓存实现"""

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]

import json
import pickle
import time
from typing import Any, Dict, Optional

from loguru import logger

from qsss.cache.errors import CacheDeserializationError

from ..config.settings import settings


class DistributedCache:
    """分布式缓存管理器"""

    def __init__(self, redis_client: Any | None = None) -> None:
        # 统一将 redis_client 视为 Any，避免第三方类型细节干扰
        self.redis_client: Any | None = redis_client
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        if self.redis_client is None and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=1,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                    retry_on_timeout=True,
                )
                # 测试连接
                self.redis_client.ping()
            except redis.ConnectionError:
                logger.warning("Redis连接失败，使用内存缓存")
                self.redis_client = None
        elif self.redis_client is None:
            self.redis_client = None

    def _generate_key(self, key: str, prefix: str = "qsss") -> str:
        """生成缓存键"""
        return f"{prefix}:{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """序列化值"""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"序列化失败: {e}")
            return json.dumps(value).encode("utf-8")

    def _deserialize_value(self, value: bytes) -> Any:
        """反序列化值"""
        try:
            return pickle.loads(value)
        except Exception as pickle_error:
            try:
                return json.loads(value.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise CacheDeserializationError(
                    "分布式缓存反序列化失败，缓存内容可能已损坏"
                ) from pickle_error

    def get(self, key: str, prefix: str = "qsss") -> Optional[Any]:
        """获取缓存值"""
        try:
            cache_key = self._generate_key(key, prefix)

            if self.redis_client:
                value = self.redis_client.get(cache_key)
                if value:
                    return self._deserialize_value(value)
            else:
                return self._memory_cache.get(cache_key)

        except CacheDeserializationError:
            raise
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")

        return None

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None, prefix: str = "qsss"
    ) -> bool:
        """设置缓存值"""
        try:
            cache_key = self._generate_key(key, prefix)
            ttl = ttl or settings.cache_ttl

            serialized_value = self._serialize_value(value)

            if self.redis_client:
                return bool(self.redis_client.setex(cache_key, ttl, serialized_value))
            else:
                self._memory_cache[cache_key] = {
                    "value": value,
                    "expire_time": time.time() + ttl if ttl else None,
                }
                return True

        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False

    def delete(self, key: str, prefix: str = "qsss") -> bool:
        """删除缓存"""
        try:
            cache_key = self._generate_key(key, prefix)

            if self.redis_client:
                return bool(self.redis_client.delete(cache_key))
            else:
                return self._memory_cache.pop(cache_key, None) is not None

        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False

    def exists(self, key: str, prefix: str = "qsss") -> bool:
        """检查缓存是否存在"""
        try:
            cache_key = self._generate_key(key, prefix)

            if self.redis_client:
                return bool(self.redis_client.exists(cache_key))
            else:
                if cache_key in self._memory_cache:
                    item = self._memory_cache[cache_key]
                    if item.get("expire_time") and time.time() > item["expire_time"]:
                        del self._memory_cache[cache_key]
                        return False
                    return True
                return False

        except Exception as e:
            logger.error(f"检查缓存存在失败: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return int(self.redis_client.delete(*keys))
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
                return count

        except Exception as e:
            logger.error(f"清除缓存模式失败: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    "used_memory": info.get("used_memory_human", "Unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "hit_rate": info.get("keyspace_hits", 0)
                    / max(
                        1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)
                    ),
                }
            else:
                memory_usage = sum(len(str(v)) for v in self._memory_cache.values())
                return {
                    "memory_items": len(self._memory_cache),
                    "memory_usage": f"{memory_usage} bytes",
                }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}


# 全局缓存实例
cache = DistributedCache()
