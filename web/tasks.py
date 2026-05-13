import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

from web.app import app, celery, db
from web.models import Backtest

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qsss.core.strategy import QuantStrategy  # noqa: E402
from src.qsss.data.manager import DataManager  # noqa: E402

PUBLIC_TASK_ERROR = "任务执行失败，请查看服务端日志。"
PUBLIC_BACKTEST_ERROR = "回测任务失败，请查看服务端日志。"


def _resolve_stock_info(data_manager: Any, stock_code: str) -> Dict[str, str]:
    stock_info = {"symbol": stock_code, "name": stock_code, "market": "Unknown"}
    try:
        stock_list = data_manager.get_stock_list()
    except Exception as e:
        app.logger.warning("获取股票元数据失败，使用默认信息: %s", e)
        return stock_info

    if stock_list is None or stock_list.empty:
        return stock_info

    code_column = "symbol" if "symbol" in stock_list.columns else "code"
    if code_column not in stock_list.columns:
        return stock_info

    matched = stock_list[stock_list[code_column].astype(str) == str(stock_code)]
    if matched.empty:
        return stock_info

    row = matched.iloc[0]
    stock_info["name"] = str(row.get("name") or stock_code)
    stock_info["market"] = str(row.get("market") or "Unknown")
    return stock_info


@celery.task(bind=True)
def run_backtest_task(self: Any, backtest_id: int) -> dict:
    """异步回测任务"""
    from web.routes import (
        _mark_backtest_failed,
        _run_backtest_core,
        _save_backtest_result,
    )

    backtest = db.session.get(Backtest, backtest_id)
    if not backtest:
        app.logger.error("回测任务不存在: backtest_id=%s", backtest_id)
        return {"status": "failed", "error": PUBLIC_BACKTEST_ERROR}

    try:
        # 更新状态为运行中
        backtest.status = "running"
        db.session.commit()
        results = _run_backtest_core(backtest)
        if results.get("error"):
            raise RuntimeError(str(results["error"]))
        _save_backtest_result(backtest, results)

        return {"status": "completed", "backtest_id": backtest_id, "results": results}

    except Exception as e:
        _mark_backtest_failed(backtest_id)
        app.logger.exception(
            "回测任务执行失败: backtest_id=%s error=%s", backtest_id, e
        )

        return {
            "status": "failed",
            "error": PUBLIC_BACKTEST_ERROR,
            "backtest_id": backtest_id,
        }


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

        quant_strategy = QuantStrategy()
        stock_info = _resolve_stock_info(data_manager, stock_code)
        analysis_result: Dict[str, Any] = quant_strategy.analyze_single_stock(
            stock_info
        )
        if not analysis_result:
            raise ValueError("策略未返回有效分析结果")
        if analysis_result.get("error"):
            raise RuntimeError(str(analysis_result["error"]))

        return {
            "status": "completed",
            "stock_code": stock_code,
            "analysis_type": analysis_type,
            "results": analysis_result,
        }

    except Exception as e:
        app.logger.exception(
            "分析任务执行失败: stock_code=%s analysis_type=%s error=%s",
            stock_code,
            analysis_type,
            e,
        )
        return {
            "status": "failed",
            "error": PUBLIC_TASK_ERROR,
            "stock_code": stock_code,
            "analysis_type": analysis_type,
        }
