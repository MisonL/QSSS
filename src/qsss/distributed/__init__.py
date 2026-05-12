"""分布式计算模块"""

from .cache import DistributedCache
from .scheduler import TaskScheduler

__all__ = ["DistributedWorker", "TaskScheduler", "DistributedCache"]


def __getattr__(name):
    """Lazy-load worker to avoid importing strategy during package import."""
    if name == "DistributedWorker":
        from .worker import DistributedWorker

        return DistributedWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
