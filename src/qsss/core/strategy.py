"""核心策略类"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, Optional

import pandas as pd
from loguru import logger

from ..config.settings import settings
from ..data.manager import data_manager
from ..strategies.short_term import ShortTermAnalyzer
from ..strategies.technical import TechnicalAnalyzer
from ..utils.decorators import retry_on_exception
from ..utils.performance import PerformanceMonitor


class QuantStrategy:
    """量化策略核心类"""

    def __init__(self) -> None:
        self._ml_model: Optional[Any] = None
        self.technical_analyzer = TechnicalAnalyzer()
        self.short_term_analyzer = ShortTermAnalyzer()
        self.performance_monitor = PerformanceMonitor()

        # 缓存
        self.stock_data_cache: Dict[str, pd.DataFrame] = {}
        self.ma15_cache: Dict[str, float] = {}

        # 锁
        self.print_lock = Lock()
        self.data_lock = Lock()
        self.cache_lock = Lock()

        # 统计信息
        self.analysis_stats: Dict[str, list[str]] = {
            "insufficient_data": [],
            "invalid_data": [],
            "failed_stocks": [],
        }

    @property
    def ml_model(self) -> Any:
        """按需加载机器学习模型，避免纯导入路径依赖 LightGBM 运行库。"""
        if self._ml_model is None:
            from ..models.ml_model import MLModel

            self._ml_model = MLModel()
        return self._ml_model

    @retry_on_exception(retries=3, delay=1)
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表，过滤掉问题较多的北交所和新三板股票"""
        stocks = data_manager.get_stock_list()
        if stocks.empty:
            return stocks

        # 过滤股票，只保留主板、科创板、创业板
        # 排除北交所(8开头)和新三板(4开头、83开头等)
        filtered_stocks = stocks[
            stocks["symbol"].str.match(r"^(60|68|00|30)")  # 只保留这些开头的股票
        ].copy()

        logger.info(f"过滤后剩余 {len(filtered_stocks)} 只股票（排除了北交所和新三板）")
        return filtered_stocks

    @retry_on_exception(retries=3, delay=1)
    def get_stock_data(self, symbol: str, start_date: str = "20220101") -> pd.DataFrame:
        """获取股票数据"""
        cache_key = f"{symbol}_{start_date}"

        with self.cache_lock:
            if cache_key in self.stock_data_cache:
                return self.stock_data_cache[cache_key]

        try:
            df = data_manager.get_daily_data(symbol, start_date)

            if not df.empty:
                with self.cache_lock:
                    self.stock_data_cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"获取{symbol}数据失败: {e}")
            return pd.DataFrame()

    def analyze_single_stock(
        self, stock_info: Dict[str, Any], start_date: str = "20220101"
    ) -> Optional[Dict[str, Any]]:
        """分析单个股票"""
        try:
            symbol = stock_info["symbol"]
            name = stock_info["name"]
            market = stock_info["market"]

            # 快速过滤明显不符合条件的股票
            if symbol.startswith(("8", "4")):  # 北交所、新三板
                return None

            # 获取数据
            df = self.get_stock_data(symbol, start_date=start_date)
            if df.empty or len(df) < settings.min_data_days:
                with self.data_lock:
                    self.analysis_stats["insufficient_data"].append(f"{name}({symbol})")
                return None

            # 检查数据质量
            if df["close"].isna().any() or (df["close"] <= 0).any():
                with self.data_lock:
                    self.analysis_stats["invalid_data"].append(f"{name}({symbol})")
                return None

            # 计算技术指标
            df = self.technical_analyzer.calculate_all_indicators(df)
            if df.empty or df.isna().all().all():
                with self.data_lock:
                    self.analysis_stats["invalid_data"].append(f"{name}({symbol})")
                return None

            # 训练模型
            model, scaler = self.ml_model.train_model(df)
            if model is None:
                return None

            # 获取预测结果
            prediction = self.ml_model.predict(df, model, scaler)
            if prediction is None or prediction < 0 or prediction > 1:
                return None

            # 计算超短线爆发潜力
            explosion_score = self.short_term_analyzer.calculate_explosion_score(df)

            # 获取MACD信号
            macd_signal = self.technical_analyzer.get_macd_signal(df)

            # 获取最新数据
            latest = df.iloc[-1]

            # 检查关键指标是否合理
            if latest.get("rsi", 50) < 0 or latest.get("rsi", 50) > 100:
                return None

            return {
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
            }

        except Exception as e:
            with self.data_lock:
                self.analysis_stats["failed_stocks"].append(
                    f"{stock_info['name']}({stock_info['symbol']}): {str(e)}"
                )
            logger.error(f"分析股票{stock_info['symbol']}失败: {e}")
            return None

    def get_optimal_thread_count(self) -> int:
        """获取最优线程数"""
        return int(
            self.performance_monitor.get_optimal_thread_count(
                min_workers=settings.min_workers,
                max_workers=settings.max_workers,
                cpu_threshold=settings.cpu_threshold,
                memory_threshold=settings.memory_threshold,
            )
        )

    def run_analysis(
        self,
        start_date: str = "20220101",
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> pd.DataFrame:
        """运行完整分析

        progress_callback: 可选回调，用于上报进度和日志到上层（例如 Web UI）。
        回调会接收到形如 {"event": str, "processed": int, "total": int, "log": str} 的字典。
        """
        logger.info("开始量化分析...")
        self.performance_monitor.start_monitoring()

        # 获取股票列表
        stocks = self.get_stock_list()
        if stocks.empty:
            logger.error("未能获取股票列表")
            if progress_callback:
                progress_callback(
                    {
                        "event": "error",
                        "processed": 0,
                        "total": 0,
                        "log": "未能获取股票列表",
                    }
                )
            return pd.DataFrame()

        if limit is not None:
            stocks = stocks.head(limit)

        total_stocks = len(stocks)
        logger.info(f"共获取到 {total_stocks} 只股票")
        if progress_callback:
            progress_callback(
                {
                    "event": "start",
                    "processed": 0,
                    "total": total_stocks,
                    "log": f"开始量化分析，共 {total_stocks} 只股票",
                }
            )

        results = []
        processed_count = 0
        error_count = 0

        def analyze_stock_wrapper(stock: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """股票分析包装函数"""
            try:
                return self.analyze_single_stock(stock, start_date=start_date)
            except KeyboardInterrupt:
                raise  # 重新抛出键盘中断
            except Exception as e:
                logger.debug(f"分析股票 {stock.get('symbol', 'Unknown')} 失败: {e}")
                return None

        # 使用线程池并行处理，限制最大线程数为4以避免过度消耗资源
        thread_count = min(self.get_optimal_thread_count(), 4)

        try:
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                # 提交任务，限制并发任务数
                stock_list = list(stocks.iterrows())
                max_concurrent = min(50, len(stock_list))  # 限制并发任务数

                # 分批处理股票
                batch_size = max_concurrent
                for i in range(0, len(stock_list), batch_size):
                    batch = stock_list[i : i + batch_size]

                    # 提交批次任务
                    future_to_stock = {
                        executor.submit(analyze_stock_wrapper, stock): stock
                        for _, stock in batch
                    }

                    # 处理批次结果
                    for future in as_completed(future_to_stock):
                        try:
                            result = future.result(timeout=30)  # 添加超时控制
                            if result:
                                results.append(result)
                            processed_count += 1

                            # 每处理一定数量的股票更新一次进度（真实百分比）
                            if progress_callback and total_stocks > 0:
                                progress_callback(
                                    {
                                        "event": "progress",
                                        "processed": processed_count,
                                        "total": total_stocks,
                                        "log": (
                                            f"已处理 {processed_count}/{total_stocks} "
                                            f"只股票"
                                        ),
                                    }
                                )

                            # 每处理50只股票写一条日志
                            if processed_count % 50 == 0:
                                logger.info(
                                    f"处理进度: {processed_count}/{total_stocks} "
                                    f"({processed_count/total_stocks*100:.1f}%)"
                                )

                        except Exception as e:
                            error_count += 1
                            processed_count += 1
                            if error_count % 25 == 0:  # 每25个错误记录一次
                                logger.warning(
                                    f"处理错误累计: {error_count}, 最近错误: {e}"
                                )
                            if progress_callback and total_stocks > 0:
                                progress_callback(
                                    {
                                        "event": "error",
                                        "processed": processed_count,
                                        "total": total_stocks,
                                        "log": f"处理错误累计: {error_count}, 最近错误: {e}",
                                    }
                                )

        except KeyboardInterrupt:
            logger.info("用户中断处理，正在清理...")
            executor.shutdown(wait=False)
            raise

        logger.info(
            f"处理完成: 成功{len(results)}只, 失败{error_count}只, 总计{processed_count}只"
        )
        if progress_callback and total_stocks > 0:
            progress_callback(
                {
                    "event": "finished",
                    "processed": processed_count,
                    "total": total_stocks,
                    "log": (
                        f"处理完成: 成功{len(results)}只, 失败{error_count}只, "
                        f"总计{processed_count}只"
                    ),
                }
            )

        # 更新性能统计
        self.performance_monitor.stop_monitoring()

        if not results:
            logger.warning("没有符合条件的股票")
            if progress_callback:
                progress_callback(
                    {
                        "event": "empty",
                        "processed": processed_count,
                        "total": total_stocks,
                        "log": "没有符合条件的股票",
                    }
                )
            return pd.DataFrame()

        # 转换为DataFrame并筛选
        results_df = pd.DataFrame(results)

        # 应用筛选条件
        filtered_df = self.apply_filters(results_df)

        # 排序
        filtered_df = filtered_df.sort_values("total_score", ascending=False)

        logger.info(f"分析完成，共筛选出 {len(filtered_df)} 只股票")
        return filtered_df

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

        return df[mask]

    def calculate_ma15(
        self, symbol: str, start_date: str = "20220101"
    ) -> Optional[float]:
        """计算15日均线"""
        try:
            cache_key = f"ma15_{symbol}_{start_date}"

            with self.cache_lock:
                if cache_key in self.ma15_cache:
                    cached = self.ma15_cache[cache_key]
                    return float(cached)

            df = self.get_stock_data(symbol, start_date=start_date)
            if df.empty:
                return None

            ma15 = df["close"].rolling(window=15).mean().iloc[-1]

            with self.cache_lock:
                self.ma15_cache[cache_key] = ma15

            return float(ma15)

        except Exception as e:
            logger.error(f"计算{symbol}15日均线失败: {e}")
            return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return {
            "performance_stats": self.performance_monitor.stop_monitoring(),
            "analysis_stats": self.analysis_stats,
            "cache_size": len(self.stock_data_cache),
        }
