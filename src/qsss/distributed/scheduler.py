"""任务调度器"""

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]

try:
    from celery import Celery

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from loguru import logger


class TaskScheduler:
    """分布式任务调度器"""

    def __init__(self, redis_client: Any | None = None) -> None:
        # 统一将 redis_client 视为 Any
        self.redis_client: Any | None = redis_client
        self._task_queue: List[Dict[str, Any]] = []
        self._task_results: Dict[str, Dict[str, Any]] = {}

        if self.redis_client is None and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=2,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                )
                self.redis_client.ping()
            except redis.ConnectionError:
                logger.warning("Redis连接失败，使用内存任务队列")
                self.redis_client = None
        elif self.redis_client is not None:
            # 外部传入 redis_client
            pass
        else:
            self.redis_client = None

        # 初始化Celery
        if CELERY_AVAILABLE:
            self.celery_app = Celery(
                "qsss",
                broker="redis://localhost:6379/0",
                backend="redis://localhost:6379/0",
            )

            self.celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="Asia/Shanghai",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=30 * 60,  # 30分钟
                task_soft_time_limit=25 * 60,  # 25分钟
                worker_prefetch_multiplier=1,
                worker_max_tasks_per_child=1000,
            )
        else:
            self.celery_app = None

    def create_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        delay: Optional[int] = None,
    ) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
        }

        if delay:
            task_data["execute_at"] = (
                datetime.now() + timedelta(seconds=delay)
            ).isoformat()

        if self.redis_client:
            self.redis_client.hset(f"task:{task_id}", mapping=task_data)
            # 添加到优先级队列
            self.redis_client.zadd("task_queue", {task_id: priority}, nx=True)
        else:
            self._task_queue.append(task_data)
            # 按优先级排序
            self._task_queue.sort(key=lambda x: x["priority"], reverse=True)

        logger.info(f"创建任务: {task_id}, 类型: {task_type}, 优先级: {priority}")
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        try:
            if self.redis_client:
                task_data = self.redis_client.hgetall(f"task:{task_id}")
                return dict(task_data) if task_data else None
            else:
                for task in self._task_queue + list(self._task_results.values()):
                    if task["id"] == task_id:
                        return task
                return None
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Any | None = None,
        error: Optional[str] = None,
    ) -> None:
        """更新任务状态"""
        try:
            if self.redis_client:
                updates = {"status": status}
                if status == "running":
                    updates["started_at"] = datetime.now().isoformat()
                elif status in ["completed", "failed"]:
                    updates["completed_at"] = datetime.now().isoformat()
                    if result is not None:
                        updates["result"] = json.dumps(result)
                    if error:
                        updates["error"] = error

                self.redis_client.hset(f"task:{task_id}", mapping=updates)
            else:
                task = self.get_task(task_id)
                if task:
                    task["status"] = status
                    if status == "running":
                        task["started_at"] = datetime.now().isoformat()
                    elif status in ["completed", "failed"]:
                        task["completed_at"] = datetime.now().isoformat()
                        task["result"] = result
                        task["error"] = error
                        # 移到结果存储
                        self._task_results[task_id] = task
                        if task in self._task_queue:
                            self._task_queue.remove(task)
        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")

    def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取待处理任务"""
        try:
            if self.redis_client:
                # 从优先级队列获取任务ID
                task_ids = self.redis_client.zrange("task_queue", 0, limit - 1)
                tasks = []

                for task_id in task_ids:
                    task_data = self.redis_client.hgetall(f"task:{task_id}")
                    if task_data and task_data.get("status") == "pending":
                        # 检查延迟执行时间
                        execute_at = task_data.get("execute_at")
                        if execute_at:
                            if datetime.now() < datetime.fromisoformat(execute_at):
                                continue

                        tasks.append(dict(task_data))
                        # 从队列中移除
                        self.redis_client.zrem("task_queue", task_id)

                return tasks
            else:
                current_time = datetime.now()
                available_tasks = []

                for task in self._task_queue[:limit]:
                    execute_at = task.get("execute_at")
                    if execute_at:
                        if current_time < datetime.fromisoformat(execute_at):
                            continue

                    if task["status"] == "pending":
                        available_tasks.append(task)
                        task["status"] = "queued"

                return available_tasks

        except Exception as e:
            logger.error(f"获取待处理任务失败: {e}")
            return []

    def distribute_stock_analysis(
        self, stock_list: List[Dict[str, Any]], batch_size: int = 50
    ) -> List[str]:
        """分发股票分析任务"""
        task_ids = []

        # 将股票列表分批
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i : i + batch_size]
            task_id = self.create_task(
                "stock_analysis_batch",
                {
                    "stocks": batch,
                    "batch_id": f"batch_{i//batch_size}",
                    "start_index": i,
                },
                priority=8,
            )
            task_ids.append(task_id)

        logger.info(
            f"分发了 {len(task_ids)} 个股票分析任务，共 {len(stock_list)} 只股票"
        )
        return task_ids

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        try:
            if self.redis_client:
                # 获取队列长度
                queue_length = self.redis_client.zcard("task_queue")

                # 获取正在运行的任务数
                running_tasks = len(
                    [
                        key
                        for key in self.redis_client.keys("task:*")
                        if self.redis_client.hget(key, "status") == b"running"
                    ]
                )

                # 获取系统资源使用情况
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory = psutil.virtual_memory()

                return {
                    "queue_length": queue_length,
                    "running_tasks": running_tasks,
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "memory_available": memory.available,
                    "cluster_healthy": cpu_percent < 90 and memory.percent < 90,
                }
            else:
                return {
                    "queue_length": len(self._task_queue),
                    "running_tasks": len(
                        [t for t in self._task_queue if t["status"] == "running"]
                    ),
                    "cpu_usage": psutil.cpu_percent(interval=0.5),
                    "memory_usage": psutil.virtual_memory().percent,
                    "cluster_healthy": True,
                }

        except Exception as e:
            logger.error(f"获取集群状态失败: {e}")
            return {"error": str(e)}

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """清理完成的任务"""
        try:
            if self.redis_client:
                # 获取过期的已完成任务
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                completed_keys = []

                for key in self.redis_client.keys("task:*"):
                    completed_at = self.redis_client.hget(key, "completed_at")
                    if completed_at:
                        if datetime.fromisoformat(completed_at) < cutoff_time:
                            completed_keys.append(key)

                # 删除过期任务
                if completed_keys:
                    return int(self.redis_client.delete(*completed_keys))
                return 0
            else:
                # 内存模式清理
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                removed_count = 0

                keys_to_remove = []
                for task_id, task in self._task_results.items():
                    if task.get("completed_at"):
                        if datetime.fromisoformat(task["completed_at"]) < cutoff_time:
                            keys_to_remove.append(task_id)

                for key in keys_to_remove:
                    del self._task_results[key]
                    removed_count += 1

                return removed_count

        except Exception as e:
            logger.error(f"清理任务失败: {e}")
            return 0
