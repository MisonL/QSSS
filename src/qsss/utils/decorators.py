"""装饰器工具"""

import functools
import time
from typing import Any, Callable

from loguru import logger


def retry_on_exception(
    retries: int = 3, delay: float = 1.0, backoff: float = 2.0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """重试装饰器，支持指数退避"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:  # 最后一次重试
                        logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
                        raise e

                    # 针对网络请求和连接错误的特殊处理
                    import requests

                    if isinstance(
                        e,
                        (
                            requests.exceptions.RequestException,
                            requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout,
                        ),
                    ):
                        logger.warning(
                            f"网络请求失败，重试 {func.__name__} ({i + 1}/{retries})，"
                            f"延迟{current_delay}s: {str(e)}"
                        )
                    elif "连接" in str(e) or "connection" in str(e).lower():
                        logger.warning(
                            f"连接失败，重试 {func.__name__} ({i + 1}/{retries})，"
                            f"延迟{current_delay}s: {str(e)}"
                        )
                    else:
                        logger.warning(f"重试 {func.__name__} ({i + 1}/{retries})...")

                    time.sleep(current_delay)
                    current_delay *= backoff  # 指数退避
            return None

        return wrapper

    return decorator


def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """计时装饰器"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"函数 {func.__name__} 执行耗时: {end_time - start_time:.2f}s")
        return result

    return wrapper


def log_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    """异常日志装饰器"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"函数 {func.__name__} 发生异常: {str(e)}")
            raise

    return wrapper
