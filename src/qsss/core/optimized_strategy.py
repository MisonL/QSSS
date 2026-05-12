"""高性能核心策略类 - 集成缓存和分布式计算"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from types import TracebackType
from typing import Any, Dict, Optional, Type, TypedDict, cast

import pandas as pd
from loguru import logger

from ..cache import cache_manager
from ..config.settings import settings
from ..data.manager import data_manager
from ..database.optimizer import DatabaseOptimizer
from ..distributed import DistributedCache, TaskScheduler
from ..strategies.short_term import ShortTermAnalyzer
from ..strategies.technical import TechnicalAnalyzer
from ..utils.decorators import retry_on_exception
from ..utils.performance import PerformanceMonitor


class AnalysisStats(TypedDict):
    insufficient_data: list[str]
    invalid_data: list[str]
    failed_stocks: list[str]
    cache_hits: int
    cache_misses: int
    distributed_tasks: int


class OptimizedQuantStrategy:
    """高性能量化策略核心类"""

    def __init__(self, use_distributed: bool = True, use_cache: bool = True):
        self._ml_model: Optional[Any] = None
        self.technical_analyzer = TechnicalAnalyzer()
        self.short_term_analyzer = ShortTermAnalyzer()
        self.performance_monitor = PerformanceMonitor()

        # 分布式和缓存配置
        self.use_distributed = use_distributed
        self.use_cache = use_cache

        if use_distributed:
            self.task_scheduler = TaskScheduler()
            self.distributed_cache = DistributedCache()

        if use_cache:
            self.cache = cache_manager

        # 统计信息
        self.analysis_stats: AnalysisStats = {
            "insufficient_data": [],
            "invalid_data": [],
            "failed_stocks": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "distributed_tasks": 0,
        }

        # 数据库优化器
        self.db_optimizer = DatabaseOptimizer()

        # 处理池
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None

    @property
    def ml_model(self) -> Any:
        """按需加载机器学习模型，避免导入优化策略时依赖 LightGBM 运行库。"""
        if self._ml_model is None:
            from ..models.ml_model import MLModel

            self._ml_model = MLModel()
        return self._ml_model

    def __enter__(self) -> "OptimizedQuantStrategy":
        """上下文管理器进入"""
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.thread_pool = ThreadPoolExecutor(max_workers=16)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """上下文管理器退出"""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

    @retry_on_exception(retries=3, delay=1)
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表 - 带缓存"""
        cache_key = "stock_list"

        if self.use_cache:
            cached_data = self.cache.get(cache_key, "stock_list")
            if cached_data is not None:
                self.analysis_stats["cache_hits"] += 1
                logger.debug("从缓存获取股票列表")
                return cached_data
            else:
                self.analysis_stats["cache_misses"] += 1

        # 从数据源获取
        stocks = data_manager.get_stock_list()

        if self.use_cache and not stocks.empty:
            self.cache.set(
                cache_key, stocks, ttl=3600, prefix="stock_list"
            )  # 1小时缓存

        return stocks

    @retry_on_exception(retries=3, delay=1)
    def get_stock_data(self, symbol: str, start_date: str = "20220101") -> pd.DataFrame:
        """获取股票数据 - 多级缓存"""
        cache_key = f"stock_data_{symbol}_{start_date}"

        # L1缓存 - 内存缓存
        if self.use_cache:
            cached_data = self.cache.get(cache_key, "stock_data")
            if cached_data is not None:
                self.analysis_stats["cache_hits"] += 1
                logger.debug(f"从L1缓存获取股票数据: {symbol}")
                return cached_data

        # L2缓存 - 分布式缓存
        if self.use_distributed:
            cached_data = self.distributed_cache.get(cache_key, "stock_data")
            if cached_data is not None:
                self.analysis_stats["cache_hits"] += 1
                logger.debug(f"从L2缓存获取股票数据: {symbol}")
                # 回填L1缓存
                if self.use_cache:
                    self.cache.set(
                        cache_key, cached_data, ttl=1800, prefix="stock_data"
                    )  # 30分钟
                return cached_data
            else:
                self.analysis_stats["cache_misses"] += 1

        # 从数据源获取
        df = data_manager.get_daily_data(symbol, start_date)

        # 缓存数据
        if not df.empty:
            if self.use_cache:
                self.cache.set(cache_key, df, ttl=1800, prefix="stock_data")  # 30分钟
            if self.use_distributed:
                self.distributed_cache.set(
                    cache_key, df, ttl=7200, prefix="stock_data"
                )  # 2小时

        return df

    def analyze_single_stock_optimized(
        self, stock_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """优化的单股票分析"""
        try:
            symbol = stock_info["symbol"]
            name = stock_info["name"]
            market = stock_info["market"]

            # 检查缓存
            cache_key = f"analysis_{symbol}"
            if self.use_cache:
                cached_result = cast(
                    Optional[Dict[str, Any]],
                    self.cache.get(cache_key, "analysis"),
                )
                if cached_result is not None:
                    self.analysis_stats["cache_hits"] += 1
                    logger.debug(f"从缓存获取分析结果: {symbol}")
                    return cached_result

            # 获取数据
            df = self.get_stock_data(symbol)
            if df.empty or len(df) < settings.min_data_days:
                self.analysis_stats["insufficient_data"].append(f"{name}({symbol})")
                return None

            # 计算技术指标
            df = self.technical_analyzer.calculate_all_indicators(df)
            if df.empty:
                self.analysis_stats["invalid_data"].append(f"{name}({symbol})")
                return None

            # 训练模型
            model, scaler = self.ml_model.train_model(df)
            if model is None:
                return None

            # 获取预测结果
            prediction = self.ml_model.predict(df, model, scaler)
            if prediction is None:
                return None

            # 计算超短线爆发潜力
            explosion_score = self.short_term_analyzer.calculate_explosion_score(df)

            # 获取MACD信号
            macd_signal = self.technical_analyzer.get_macd_signal(df)

            # 获取最新数据
            latest = df.iloc[-1]

            result = {
                "symbol": symbol,
                "name": name,
                "market": market,
                "prediction": prediction,
                "momentum_score": latest.get("momentum_1m", 0),
                "rsi": latest.get("rsi", 50),
                "volatility": latest.get("volatility", 0.5),
                "macd": latest.get("macd", 0),
                "macd_status": macd_signal,
                "close": latest["close"],
                "volume": latest["volume"],
                "turn": latest.get("turn", 0),
                "explosion_score": explosion_score,
                "analysis_timestamp": time.time(),
            }

            # 缓存结果
            if self.use_cache:
                self.cache.set(cache_key, result, ttl=900, prefix="analysis")  # 15分钟

            return result

        except Exception as e:
            logger.error(f"分析股票{stock_info['symbol']}失败: {e}")
            self.analysis_stats["failed_stocks"].append(
                f"{stock_info['name']}({stock_info['symbol']}): {str(e)}"
            )
            return None

    def get_optimal_worker_count(self) -> int:
        """获取最优工作进程数"""
        return self.performance_monitor.get_optimal_thread_count(
            min_workers=settings.min_workers,
            max_workers=settings.max_workers,
            cpu_threshold=settings.cpu_threshold,
            memory_threshold=settings.memory_threshold,
        )

    def run_distributed_analysis(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """运行分布式分析"""
        logger.info("开始分布式量化分析...")
        self.performance_monitor.start_monitoring()

        if stock_list.empty:
            logger.error("股票列表为空")
            return pd.DataFrame()

        logger.info(f"共获取到 {len(stock_list)} 只股票")

        # 使用分布式任务调度
        if self.use_distributed:
            return self._run_distributed_batch_analysis(stock_list)
        else:
            return self._run_local_parallel_analysis(stock_list)

    def _run_distributed_batch_analysis(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """分布式批量分析"""
        try:
            # 将股票列表转换为字典列表
            stocks_data = stock_list.to_dict("records")

            # 分发任务
            task_ids = self.task_scheduler.distribute_stock_analysis(
                stocks_data, batch_size=20  # 每批20只股票
            )

            self.analysis_stats["distributed_tasks"] = len(task_ids)
            logger.info(f"分发了 {len(task_ids)} 个分布式任务")

            # 等待所有任务完成
            results = []
            completed_tasks = 0

            while completed_tasks < len(task_ids):
                for task_id in task_ids[:]:
                    task = self.task_scheduler.get_task(task_id)
                    if task and task.get("status") in ["completed", "failed"]:
                        if task["status"] == "completed" and task.get("result"):
                            task_result = json.loads(task.get("result", "{}"))
                            if "results" in task_result:
                                results.extend(task_result["results"])

                        task_ids.remove(task_id)
                        completed_tasks += 1

                        if completed_tasks % 5 == 0:
                            logger.info(f"任务进度: {completed_tasks}/{len(task_ids)}")

                if task_ids:  # 还有未完成的任务
                    time.sleep(2)  # 等待2秒再检查

            logger.info(f"分布式分析完成，共获得 {len(results)} 个结果")

            # 转换为DataFrame并筛选
            if results:
                results_df = pd.DataFrame(results)
                return self.apply_filters(results_df)
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"分布式分析失败: {e}")
            return self._run_local_parallel_analysis(stock_list)

    def _run_local_parallel_analysis(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """本地并行分析"""
        logger.info("开始本地并行分析...")

        results: list[Dict[str, Any]] = []

        # 使用线程池并行处理
        def analyze_stock_wrapper(stock: Any) -> Optional[Dict[str, Any]]:
            return self.analyze_single_stock_optimized(stock)

        # 获取最优线程数
        worker_count = self.get_optimal_worker_count()

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # 提交任务
            future_to_stock = {
                executor.submit(analyze_stock_wrapper, stock): stock
                for _, stock in stock_list.iterrows()
            }

            # 处理结果
            for future in as_completed(future_to_stock):
                result = future.result()
                if result:
                    results.append(result)

        # 更新性能统计
        self.performance_monitor.stop_monitoring()

        if not results:
            logger.warning("没有符合条件的股票")
            return pd.DataFrame()

        # 转换为DataFrame并筛选
        results_df = pd.DataFrame(results)
        return self.apply_filters(results_df)

    def run_analysis(self) -> pd.DataFrame:
        """运行完整分析（兼容原接口）"""
        logger.info("开始量化分析...")
        self.performance_monitor.start_monitoring()

        # 获取股票列表
        stocks = self.get_stock_list()
        if stocks.empty:
            logger.error("未能获取股票列表")
            return pd.DataFrame()

        # 使用优化的分析方法
        return self.run_distributed_analysis(stocks)

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用筛选条件"""
        if df.empty:
            return df

        # 计算综合得分
        df["total_score"] = (
            df["prediction"] * 0.3
            + df["momentum_score"] * 0.2
            + df["explosion_score"] * 0.35
            + (1 - df["volatility"]) * 0.15
        )

        # 应用筛选条件
        mask = (
            (df["prediction"] >= settings.min_prediction_threshold)
            & (df["momentum_score"] >= settings.min_momentum_score)
            & (df["rsi"] >= settings.rsi_range[0])
            & (df["rsi"] <= settings.rsi_range[1])
            & (df["volatility"] <= settings.max_volatility)
            & (df["volume"] >= settings.min_volume)
            & (df["close"] >= settings.min_price)
        )

        filtered_df = df[mask].sort_values("total_score", ascending=False)

        logger.info(f"筛选完成，从 {len(df)} 只股票中选出 {len(filtered_df)} 只")
        return filtered_df

    def calculate_ma15(self, symbol: str) -> Optional[float]:
        """计算15日均线 - 带缓存"""
        cache_key = f"ma15_{symbol}"

        # 检查缓存
        if self.use_cache:
            cached_result = cast(Optional[float], self.cache.get(cache_key, "ma15"))
            if cached_result is not None:
                return cached_result

        try:
            df = self.get_stock_data(symbol)
            if df.empty:
                return None

            ma15 = df["close"].rolling(window=15).mean().iloc[-1]

            # 缓存结果
            if self.use_cache:
                self.cache.set(cache_key, float(ma15), ttl=3600, prefix="ma15")  # 1小时

            return float(ma15)

        except Exception as e:
            logger.error(f"计算{symbol}15日均线失败: {e}")
            return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        # 获取性能统计信息
        try:
            performance_stats = self.performance_monitor.stop_monitoring()
        except Exception:
            performance_stats = {}

        # 获取缓存统计信息
        try:
            cache_stats = self.cache.get_stats() if self.use_cache else {}
        except Exception:
            cache_stats = {}

        return {
            "performance_stats": performance_stats,
            "analysis_stats": self.analysis_stats,
            "cache_stats": cache_stats,
            "distributed_enabled": self.use_distributed,
            "cache_enabled": self.use_cache,
        }
