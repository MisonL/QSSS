"""分布式工作节点"""

import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import psutil
from loguru import logger

from ..core.strategy import QuantStrategy
from .scheduler import TaskScheduler

PUBLIC_BATCH_ERROR = "批次任务处理失败，请查看日志。"
PUBLIC_WORKER_STATUS_ERROR = "获取工作节点状态失败，请查看日志。"
PUBLIC_TASK_ERROR = "任务处理失败，请查看日志。"


class DistributedWorker:
    """分布式工作节点"""

    def __init__(self, worker_id: Optional[str] = None, max_workers: int = 4) -> None:
        self.worker_id = worker_id or f"worker_{int(time.time())}"
        self.max_workers = max_workers
        self.scheduler = TaskScheduler()
        self.strategy = QuantStrategy()
        self.running = False
        self.processed_tasks = 0
        self.success_tasks = 0
        self.failed_tasks = 0

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """信号处理"""
        logger.info(f"收到信号 {signum}，正在关闭工作节点...")
        self.running = False
        sys.exit(0)

    def _process_stock_analysis_batch(
        self, task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理股票分析批次任务"""
        try:
            stocks = task_data.get("stocks", [])
            batch_id = task_data.get("batch_id", "unknown")

            logger.info(
                f"工作节点 {self.worker_id} 开始处理批次 {batch_id}，共 {len(stocks)} 只股票"
            )

            results = []
            failed_stocks = []

            # 使用线程池并行处理批次内的股票
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_stock = {
                    executor.submit(self._analyze_single_stock, stock): stock
                    for stock in stocks
                }

                for future in as_completed(future_to_stock):
                    stock = future_to_stock[future]
                    try:
                        result = future.result(timeout=60)  # 60秒超时
                        if result:
                            results.append(result)
                        else:
                            failed_stocks.append(stock)
                    except Exception as e:
                        logger.error(
                            f"分析股票 {stock.get('symbol', 'unknown')} 失败: {e}"
                        )
                        failed_stocks.append(stock)

            logger.info(
                f"批次 {batch_id} 处理完成 - 成功: {len(results)}, 失败: {len(failed_stocks)}"
            )

            return {
                "batch_id": batch_id,
                "results": results,
                "failed_stocks": failed_stocks,
                "processed_count": len(stocks),
                "success_count": len(results),
                "failure_count": len(failed_stocks),
            }

        except Exception as e:
            logger.error(f"处理批次任务失败: {e}")
            return {
                "batch_id": batch_id,
                "results": [],
                "failed_stocks": stocks,
                "error": PUBLIC_BATCH_ERROR,
            }

    def _analyze_single_stock(
        self, stock_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """分析单个股票"""
        try:
            symbol = stock_info["symbol"]
            name = stock_info["name"]

            # 使用策略分析
            result = self.strategy.analyze_single_stock(stock_info)

            if result:
                logger.debug(f"股票 {symbol}({name}) 分析成功")
                return result
            else:
                logger.warning(f"股票 {symbol}({name}) 分析返回空结果")
                return None

        except Exception as e:
            logger.error(f"分析股票 {stock_info.get('symbol', 'unknown')} 失败: {e}")
            return None

    def _get_system_stats(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()

            return {
                "worker_id": self.worker_id,
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "processed_tasks": self.processed_tasks,
                "success_tasks": self.success_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": self.success_tasks / max(1, self.processed_tasks),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"worker_id": self.worker_id, "error": PUBLIC_WORKER_STATUS_ERROR}

    def _should_accept_task(self) -> bool:
        """判断是否应接受新任务"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # 如果系统资源使用率过高，暂停接受新任务
            if cpu_percent > 85 or memory.percent > 90:
                return False

            return True
        except Exception as e:
            logger.error(f"检查系统资源失败: {e}")
            return True

    def start(self) -> None:
        """启动工作节点"""
        logger.info(f"启动分布式工作节点: {self.worker_id}")
        self.running = True

        while self.running:
            try:
                # 检查系统资源
                if not self._should_accept_task():
                    logger.warning("系统资源使用率过高，暂停处理任务")
                    time.sleep(10)
                    continue

                # 获取待处理任务
                pending_tasks = self.scheduler.get_pending_tasks(limit=1)

                if not pending_tasks:
                    # 没有任务，等待一段时间
                    time.sleep(2)
                    continue

                task = pending_tasks[0]
                task_id = task["id"]
                task_type = task["type"]

                logger.info(
                    f"工作节点 {self.worker_id} 开始处理任务: {task_id}, 类型: {task_type}"
                )

                # 更新任务状态
                self.scheduler.update_task_status(task_id, "running")

                # 处理任务
                result = None
                error = None

                try:
                    if task_type == "stock_analysis_batch":
                        result = self._process_stock_analysis_batch(task["payload"])
                    else:
                        error = f"不支持的任务类型: {task_type}"

                except Exception as e:
                    error = PUBLIC_TASK_ERROR
                    logger.error(f"处理任务 {task_id} 失败: {e}")

                # 更新任务状态和结果
                if error:
                    self.scheduler.update_task_status(task_id, "failed", error=error)
                    self.failed_tasks += 1
                else:
                    self.scheduler.update_task_status(
                        task_id, "completed", result=result
                    )
                    self.success_tasks += 1

                self.processed_tasks += 1

                # 定期报告状态
                if self.processed_tasks % 10 == 0:
                    stats = self._get_system_stats()
                    logger.info(f"工作节点状态: {json.dumps(stats, indent=2)}")

                # 短暂休息，避免CPU过载
                time.sleep(0.5)

            except KeyboardInterrupt:
                logger.info("收到键盘中断，正在关闭...")
                break
            except Exception as e:
                logger.error(f"工作节点主循环错误: {e}")
                time.sleep(5)

        logger.info(f"工作节点 {self.worker_id} 已停止")
        logger.info(
            "处理任务统计 - 总计: "
            f"{self.processed_tasks}, 成功: {self.success_tasks}, 失败: {self.failed_tasks}"
        )

    def stop(self) -> None:
        """停止工作节点"""
        logger.info(f"停止工作节点: {self.worker_id}")
        self.running = False


# 任务处理函数（供Celery调用）
# 检查TaskScheduler类是否有celery_app属性
if hasattr(TaskScheduler, "celery_app") and TaskScheduler.celery_app is not None:

    @TaskScheduler.celery_app.task(bind=True, name="qsss.distributed.stock_analysis")
    def process_stock_analysis_task(  # type: ignore[no-untyped-def]
        self, task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Celery任务 - 处理股票分析"""
        try:
            worker_id = f"celery_worker_{self.request.id}"
            worker = DistributedWorker(worker_id=worker_id, max_workers=2)

            # 更新任务进度
            self.update_state(
                state="PROGRESS",
                meta={"status": "正在处理股票分析批次", "worker_id": worker_id},
            )

            result = worker._process_stock_analysis_batch(task_data)

            return {"status": "completed", "result": result, "worker_id": worker_id}

        except Exception as e:
            logger.error(f"Celery任务处理失败: {e}")
            self.update_state(state="FAILURE", meta={"error": PUBLIC_TASK_ERROR})
            raise
