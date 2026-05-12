"""数据库优化模块"""

try:
    from sqlalchemy import Index, text
    from sqlalchemy.orm import scoped_session, sessionmaker
    from sqlalchemy.pool import QueuePool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Index = None  # type: ignore
    text = None  # type: ignore
    sessionmaker = None  # type: ignore
    scoped_session = None  # type: ignore
    QueuePool = None  # type: ignore

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

# 可选导入数据库模型
try:
    from ..web.models import AnalysisResult, Backtest, Stock, db

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    db = Stock = AnalysisResult = Backtest = None  # type: ignore[assignment, misc]


class DatabaseOptimizer:
    """数据库查询优化器"""

    def __init__(self, app: Optional[Any] = None) -> None:
        self.app: Optional[Any] = (
            app if SQLALCHEMY_AVAILABLE and MODELS_AVAILABLE else None
        )
        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.slow_query_threshold: float = 1.0  # 1秒

        if not SQLALCHEMY_AVAILABLE or not MODELS_AVAILABLE:
            logger.warning("数据库优化器依赖未安装，功能将受限")
            return

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Any) -> None:
        """初始化应用"""
        if not SQLALCHEMY_AVAILABLE or not MODELS_AVAILABLE:
            logger.warning("数据库优化器依赖未安装，无法初始化")
            return

        self.app = app

        # 配置数据库连接池
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "pool_size": 20,
            "pool_recycle": 3600,
            "pool_pre_ping": True,
            "max_overflow": 30,
            "pool_timeout": 30,
            "poolclass": QueuePool,
            "echo": False,  # 生产环境关闭SQL日志
            "execution_options": {"isolation_level": "READ_COMMITTED"},
        }

        # 创建数据库索引
        self._create_indexes()

        # 启用查询缓存
        self._setup_query_cache()

    def _create_indexes(self) -> None:
        """创建数据库索引"""
        if self.app is None:
            logger.warning("DatabaseOptimizer.app 未配置，跳过索引创建")
            return
        try:
            with self.app.app_context():
                # 股票表索引
                db.session.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_stock_code ON stock(code);
                    CREATE INDEX IF NOT EXISTS idx_stock_market ON stock(market);
                    CREATE INDEX IF NOT EXISTS idx_stock_name ON stock(name);
                    CREATE INDEX IF NOT EXISTS idx_stock_created_at
                        ON stock(created_at);
                """
                    )
                )

                # 分析结果表索引
                db.session.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_analysis_stock_id
                        ON analysis_result(stock_id);
                    CREATE INDEX IF NOT EXISTS idx_analysis_date
                        ON analysis_result(analysis_date);
                    CREATE INDEX IF NOT EXISTS idx_analysis_created_at
                        ON analysis_result(created_at);
                    CREATE INDEX IF NOT EXISTS idx_analysis_strategy
                        ON analysis_result(strategy_type);
                    CREATE INDEX IF NOT EXISTS idx_analysis_stock_date
                        ON analysis_result(stock_id, analysis_date);
                """
                    )
                )

                # 回测表索引
                db.session.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_backtest_stock_id
                        ON backtest(stock_id);
                    CREATE INDEX IF NOT EXISTS idx_backtest_strategy_id
                        ON backtest(strategy_id);
                    CREATE INDEX IF NOT EXISTS idx_backtest_status
                        ON backtest(status);
                    CREATE INDEX IF NOT EXISTS idx_backtest_created_at
                        ON backtest(created_at);
                    CREATE INDEX IF NOT EXISTS idx_backtest_date_range
                        ON backtest(start_date, end_date);
                """
                    )
                )

                db.session.commit()
                logger.info("数据库索引创建完成")

        except Exception as e:
            logger.error(f"创建数据库索引失败: {e}")
            db.session.rollback()

    def _setup_query_cache(self) -> None:
        """设置查询缓存"""
        # 这里可以集成Redis缓存
        pass

    @contextmanager
    def timed_query(self, query_name: str) -> Iterator[None]:
        """查询性能监控上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            if execution_time > self.slow_query_threshold:
                logger.warning(f"慢查询警告: {query_name} 耗时 {execution_time:.2f}秒")
            else:
                logger.debug(f"查询执行: {query_name} 耗时 {execution_time:.3f}秒")

    def get_stocks_with_analysis(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """高效获取带分析结果的股票列表"""
        cache_key = f"stocks_with_analysis_{limit}_{offset}"

        # 检查缓存
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        with self.timed_query("get_stocks_with_analysis"):
            try:
                # 使用单次查询获取所有数据
                query = text(
                    """
                    SELECT
                        s.id, s.code, s.name, s.market, s.created_at,
                        ar.analysis_date, ar.strategy_type, ar.prediction_score,
                        ar.momentum_score, ar.rsi, ar.volatility
                    FROM stock s
                    LEFT JOIN analysis_result ar ON s.id = ar.stock_id
                    AND ar.id = (
                        SELECT id FROM analysis_result
                        WHERE stock_id = s.id
                        ORDER BY analysis_date DESC
                        LIMIT 1
                    )
                    ORDER BY s.code
                    LIMIT :limit OFFSET :offset
                """
                )

                result = db.session.execute(query, {"limit": limit, "offset": offset})

                stocks = []
                for row in result:
                    stock_data = {
                        "id": row[0],
                        "code": row[1],
                        "name": row[2],
                        "market": row[3],
                        "created_at": row[4],
                        "latest_analysis": (
                            {
                                "analysis_date": row[5],
                                "strategy_type": row[6],
                                "prediction_score": row[7],
                                "momentum_score": row[8],
                                "rsi": row[9],
                                "volatility": row[10],
                            }
                            if row[5]
                            else None
                        ),
                    }
                    stocks.append(stock_data)

                # 缓存结果（5分钟）
                self.query_cache[cache_key] = stocks

                # 设置缓存过期
                def clear_cache() -> None:
                    if cache_key in self.query_cache:
                        del self.query_cache[cache_key]

                # 使用定时器清除缓存
                import threading

                timer = threading.Timer(300, clear_cache)  # 5分钟
                timer.start()

                return stocks

            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
                return []

    def get_analysis_history(
        self, stock_id: int, days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取股票分析历史"""
        cache_key = f"analysis_history_{stock_id}_{days}"

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        with self.timed_query("get_analysis_history"):
            try:
                query = text(
                    """
                    SELECT
                        analysis_date, strategy_type, prediction_score,
                        momentum_score, rsi, volatility, created_at
                    FROM analysis_result
                    WHERE stock_id = :stock_id
                    AND analysis_date >= DATE('now', '-' || :days || ' days')
                    ORDER BY analysis_date DESC
                """
                )

                result = db.session.execute(query, {"stock_id": stock_id, "days": days})

                history = []
                for row in result:
                    history.append(
                        {
                            "analysis_date": row[0],
                            "strategy_type": row[1],
                            "prediction_score": row[2],
                            "momentum_score": row[3],
                            "rsi": row[4],
                            "volatility": row[5],
                            "created_at": row[6],
                        }
                    )

                self.query_cache[cache_key] = history

                # 设置缓存过期
                def clear_cache() -> None:
                    if cache_key in self.query_cache:
                        del self.query_cache[cache_key]

                import threading

                timer = threading.Timer(180, clear_cache)  # 3分钟
                timer.start()

                return history

            except Exception as e:
                logger.error(f"获取分析历史失败: {e}")
                return []

    def bulk_insert_analysis_results(self, results: List[Dict[str, Any]]) -> int:
        """批量插入分析结果"""
        if not results:
            return 0

        with self.timed_query("bulk_insert_analysis_results"):
            try:
                # 使用批量插入
                insert_query = text(
                    """
                    INSERT INTO analysis_result (
                        stock_id, analysis_date, strategy_type, prediction_score,
                        momentum_score, rsi, volatility, created_at
                    ) VALUES (
                        :stock_id, :analysis_date, :strategy_type, :prediction_score,
                        :momentum_score, :rsi, :volatility, :created_at
                    )
                """
                )

                # 准备数据
                insert_data = []
                for result in results:
                    insert_data.append(
                        {
                            "stock_id": result["stock_id"],
                            "analysis_date": result.get(
                                "analysis_date", datetime.now().date()
                            ),
                            "strategy_type": result.get("strategy_type", "technical"),
                            "prediction_score": result.get("prediction_score", 0.5),
                            "momentum_score": result.get("momentum_score", 0),
                            "rsi": result.get("rsi", 50),
                            "volatility": result.get("volatility", 0.5),
                            "created_at": datetime.now(),
                        }
                    )

                # 执行批量插入
                db.session.execute(insert_query, insert_data)
                db.session.commit()

                logger.info(f"批量插入 {len(results)} 条分析结果成功")
                return len(results)

            except Exception as e:
                logger.error(f"批量插入分析结果失败: {e}")
                db.session.rollback()
                return 0

    def get_top_performing_stocks(
        self, limit: int = 20, days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取表现最佳的股票"""
        cache_key = f"top_performing_stocks_{limit}_{days}"

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        with self.timed_query("get_top_performing_stocks"):
            try:
                query = text(
                    """
                    SELECT
                        s.code, s.name, s.market,
                        AVG(ar.prediction_score) as avg_prediction,
                        AVG(ar.momentum_score) as avg_momentum,
                        AVG(ar.rsi) as avg_rsi,
                        COUNT(ar.id) as analysis_count,
                        MAX(ar.analysis_date) as latest_analysis
                    FROM stock s
                    JOIN analysis_result ar ON s.id = ar.stock_id
                    WHERE ar.analysis_date >= DATE('now', '-' || :days || ' days')
                    GROUP BY s.id, s.code, s.name, s.market
                    HAVING COUNT(ar.id) >= 3
                    ORDER BY avg_prediction DESC, avg_momentum DESC
                    LIMIT :limit
                """
                )

                result = db.session.execute(query, {"limit": limit, "days": days})

                stocks = []
                for row in result:
                    stocks.append(
                        {
                            "code": row[0],
                            "name": row[1],
                            "market": row[2],
                            "avg_prediction": row[3],
                            "avg_momentum": row[4],
                            "avg_rsi": row[5],
                            "analysis_count": row[6],
                            "latest_analysis": row[7],
                        }
                    )

                self.query_cache[cache_key] = stocks

                # 设置缓存过期
                import threading

                timer = threading.Timer(600, clear_cache)  # 10分钟
                timer.start()

                return stocks

            except Exception as e:
                logger.error(f"获取表现最佳股票失败: {e}")
                return []

    def vacuum_database(self) -> None:
        """清理和优化数据库"""
        if self.app is None:
            logger.warning("DatabaseOptimizer.app 未配置，跳过数据库清理")
            return
        try:
            with self.app.app_context():
                logger.info("开始数据库清理和优化...")

                # 清理过期的分析结果（保留90天）
                deleted_count = db.session.execute(
                    text(
                        """
                    DELETE FROM analysis_result
                    WHERE analysis_date < DATE('now', '-90 days')
                """
                    )
                )

                # 清理过期的回测结果（保留180天）
                deleted_backtests = db.session.execute(
                    text(
                        """
                    DELETE FROM backtest
                    WHERE created_at < DATE('now', '-180 days')
                    AND status = 'completed'
                """
                    )
                )

                db.session.commit()

                logger.info(
                    "数据库清理完成 - 删除过期分析结果: "
                    f"{deleted_count.rowcount}, 过期回测: {deleted_backtests.rowcount}"
                )

                # 执行VACUUM优化（SQLite）
                try:
                    db.session.execute(text("VACUUM"))
                    logger.info("数据库VACUUM优化完成")
                except Exception:
                    logger.warning("VACUUM优化失败（可能不是SQLite数据库）")

        except Exception as e:
            logger.error(f"数据库清理失败: {e}")
            db.session.rollback()


def clear_cache() -> None:
    """清除缓存函数"""
    pass  # 这个函数会在定时器中调用
