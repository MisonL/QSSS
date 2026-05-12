import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

from web.app import celery, db
from web.models import Backtest

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qsss.core.strategy import QuantStrategy  # noqa: E402
from src.qsss.data.manager import DataManager  # noqa: E402


@celery.task(bind=True)
def run_backtest_task(self: Any, backtest_id: int) -> dict:
    """异步回测任务"""
    backtest = Backtest.query.get(backtest_id)
    if not backtest:
        return {"error": "Backtest not found"}

    try:
        # 更新状态为运行中
        backtest.status = "running"
        db.session.commit()

        # 获取股票和策略信息
        stock = backtest.stock
        strategy = backtest.strategy

        # 初始化数据管理器
        data_manager = DataManager()

        # 获取历史数据
        start_date = backtest.start_date.strftime("%Y-%m-%d")
        end_date = backtest.end_date.strftime("%Y-%m-%d")

        # 获取股票数据
        stock_data = data_manager.get_daily_data(
            stock.code, start_date=start_date, end_date=end_date
        )

        if stock_data.empty:
            raise ValueError("无法获取股票数据")

        # 初始化策略
        quant_strategy = QuantStrategy()

        # 执行回测
        # 注意：核心策略暂未实现回测方法，这里仅做占位处理
        results: Dict[str, Any] = {}
        if hasattr(quant_strategy, "backtest"):
            results = quant_strategy.backtest(  # type: ignore
                stock_data=stock_data,
                strategy_type=strategy.type,
                parameters=strategy.parameters or {},
            )
        else:
            results = {"error": "Backtest method not implemented in core strategy"}

        # 更新回测结果
        backtest.total_return = results.get("total_return", 0)
        backtest.annual_return = results.get("annual_return", 0)
        backtest.max_drawdown = results.get("max_drawdown", 0)
        backtest.sharpe_ratio = results.get("sharpe_ratio", 0)
        backtest.win_rate = results.get("win_rate", 0)
        backtest.total_trades = results.get("total_trades", 0)
        backtest.results_data = results
        backtest.status = "completed"
        backtest.completed_at = datetime.utcnow()

        db.session.commit()

        return {"status": "completed", "backtest_id": backtest_id, "results": results}

    except Exception as e:
        # 更新错误状态
        backtest.status = "failed"
        db.session.commit()

        return {"status": "failed", "error": str(e), "backtest_id": backtest_id}


@celery.task(bind=True)
def run_analysis_task(
    self: Any,
    stock_code: str,
    analysis_type: str,
    analysis_date: Optional[date] = None,
) -> dict:
    """异步分析任务"""
    try:
        if not analysis_date:
            analysis_date = datetime.now().date()

        # 初始化数据管理器
        data_manager = DataManager()

        # 获取股票数据
        stock_data = data_manager.get_daily_data(
            stock_code,
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            end_date=analysis_date.strftime("%Y-%m-%d") if analysis_date else None,
        )

        if stock_data.empty:
            raise ValueError("无法获取股票数据")

        # 执行分析
        quant_strategy = QuantStrategy()
        # 注意：核心策略暂未实现analyze方法，这里仅做占位处理
        analysis_result: Dict[str, Any] = {}
        if hasattr(quant_strategy, "analyze"):
            analysis_result = quant_strategy.analyze(  # type: ignore
                stock_data=stock_data, analysis_type=analysis_type
            )
        else:
            # 尝试使用 analyze_single_stock 作为替代
            stock_info = {"symbol": stock_code, "name": stock_code, "market": "Unknown"}
            single_result = quant_strategy.analyze_single_stock(stock_info)
            if single_result:
                analysis_result = single_result
            else:
                analysis_result = {"error": "Analysis failed or not implemented"}

        return {
            "status": "completed",
            "stock_code": stock_code,
            "analysis_type": analysis_type,
            "results": analysis_result,
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "stock_code": stock_code,
            "analysis_type": analysis_type,
        }
