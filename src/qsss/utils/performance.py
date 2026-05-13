"""性能监控工具"""

import time
from typing import Any, Dict

import psutil
from loguru import logger


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.stats: Dict[str, Any] = {}

    def start_monitoring(self) -> None:
        """开始监控"""
        self.start_time = time.time()
        self.stats = {
            "start_time": self.start_time,
            "processed_stocks": 0,
            "success_count": 0,
            "failed_count": 0,
            "total_retries": 0,
            "avg_process_time": 0,
            "peak_memory": 0,
            "peak_cpu": 0,
        }

    def get_optimal_thread_count(
        self,
        min_workers: int,
        max_workers: int,
        cpu_threshold: float,
        memory_threshold: float,
    ) -> int:
        """获取最优线程数"""
        try:
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            swap = psutil.swap_memory()
            swap_percent = swap.percent

            # 更新峰值统计
            self.stats["peak_cpu"] = max(self.stats.get("peak_cpu", 0), cpu_percent)
            self.stats["peak_memory"] = max(
                self.stats.get("peak_memory", 0), memory_percent
            )

            # 计算综合负载
            system_load = max(
                cpu_percent / 100, memory_percent / 100, swap_percent / 100
            )

            # 根据负载调整线程数
            if system_load > 0.8:
                optimal_threads = min_workers
            elif system_load > 0.6:
                optimal_threads = int(
                    min_workers
                    + (max_workers - min_workers) * (0.8 - system_load) / 0.2
                )
            else:
                optimal_threads = int(max_workers * (0.9 - system_load * 0.5))

            # 确保在合理范围内
            optimal_threads = max(min(optimal_threads, max_workers), min_workers)

            logger.info(
                f"系统状态 - CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%, "
                f"交换空间: {swap_percent:.1f}%, 线程数: {optimal_threads}"
            )

            return optimal_threads

        except Exception as e:
            logger.error(f"获取最优线程数失败: {e}")
            return min_workers

    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控并返回统计信息"""
        if self.start_time is None:
            return {}

        end_time = time.time()
        self.stats["end_time"] = end_time
        self.stats["total_time"] = end_time - self.start_time

        success_rate = (
            self.stats["success_count"] / max(1, self.stats["processed_stocks"]) * 100
        )
        logger.info(
            f"性能统计 - 总耗时: {self.stats['total_time']:.0f}s, "
            f"处理股票: {self.stats['processed_stocks']}, 成功率: {success_rate:.1f}%"
        )

        return self.stats.copy()

    def update_stats(self, success: bool, process_time: float) -> None:
        """更新统计信息"""
        self.stats["processed_stocks"] += 1
        if success:
            self.stats["success_count"] += 1
        else:
            self.stats["failed_count"] += 1

        # 更新平均处理时间
        current_count = self.stats["processed_stocks"]
        current_avg = self.stats["avg_process_time"]
        self.stats["avg_process_time"] = (
            current_avg * (current_count - 1) + process_time
        ) / current_count
