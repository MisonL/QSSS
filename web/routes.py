import math
import os
import re
import sys
import threading
import time
import uuid
from datetime import date, datetime
from typing import Any, Dict

from flask import jsonify, render_template, request

from web.app import app, celery, db
from web.models import AnalysisResult, Backtest, Stock, Strategy

# Add src to path for importing QSSS modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.qsss.core.strategy import QuantStrategy  # noqa: E402
from src.qsss.data.manager import data_manager  # noqa: E402

# 简单的内存级全市场分析任务管理（避免依赖 Redis/Celery 也能展示进度和日志）
_market_jobs_lock = threading.Lock()
_market_jobs: Dict[str, Dict[str, Any]] = {}
_boards_cache_lock = threading.Lock()
_boards_cache: Dict[str, Dict[str, Any]] = {}
CELERY_TASK_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
INDEX_SYMBOLS = ["000001", "399001", "399006"]
DEFAULT_WATCHLIST_SYMBOLS = ["600519", "300750", "601318", "000651", "600036"]
AI_TECH_SYMBOLS = ["300033", "000977", "603986", "688256", "300308"]
BOARD_CACHE_TTL_SECONDS = 60
BOARD_TABLE_LIMIT = 50
PUBLIC_DATA_SOURCE_ERROR = "数据源请求失败，请查看服务端日志。"
PUBLIC_ANALYSIS_ERROR = "分析失败，请查看服务端日志。"
PUBLIC_BACKTEST_ERROR = "回测失败，请查看服务端日志。"


@app.route("/")
def home() -> str:
    """首页 - 显示股票和最新分析概览"""
    # 股票总数
    stocks = Stock.query.order_by(Stock.code).all()
    total_stocks = len(stocks)

    # 策略数量
    total_strategies = Strategy.query.filter_by(is_active=True).count()

    # 最近分析与回测
    recent_analyses = (
        AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(10).all()
    )
    recent_backtests = (
        Backtest.query.order_by(Backtest.created_at.desc()).limit(5).all()
    )

    return render_template(
        "index.html",
        total_stocks=total_stocks,
        total_strategies=total_strategies,
        recent_backtests=recent_backtests,
        analyses=recent_analyses,
    )


@app.route("/stocks")
def stocks() -> str:
    """股票列表页面"""
    page = request.args.get("page", 1, type=int)
    per_page = 20

    stocks = Stock.query.paginate(page=page, per_page=per_page, error_out=False)

    return render_template("stocks.html", stocks=stocks)


@app.route("/stock/<int:stock_id>")
def stock_detail(stock_id: int) -> str:
    """股票详情页面"""
    stock = Stock.query.get_or_404(stock_id)
    analyses = (
        AnalysisResult.query.filter_by(stock_id=stock_id)
        .order_by(AnalysisResult.analysis_date.desc())
        .all()
    )

    backtests = (
        Backtest.query.filter_by(stock_id=stock_id)
        .order_by(Backtest.created_at.desc())
        .all()
    )

    return render_template(
        "stock_detail.html",
        stock=stock.to_dict(),
        analyses=[a.to_dict() for a in analyses],
        backtests=[b.to_dict() for b in backtests],
    )


@app.route("/strategies")
def strategies() -> str:
    """策略列表页面"""
    strategies = Strategy.query.filter_by(is_active=True).all()
    return render_template(
        "strategies.html", strategies=[s.to_dict() for s in strategies]
    )


@app.route("/backtests")
def backtests() -> str:
    """回测列表页面"""
    page = request.args.get("page", 1, type=int)
    per_page = 20

    backtests = Backtest.query.order_by(Backtest.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return render_template("backtests.html", backtests=backtests)


@app.route("/analysis")
def analysis() -> str:
    """分析页面"""
    stocks = Stock.query.order_by(Stock.code).all()
    strategies = Strategy.query.filter_by(is_active=True).all()

    return render_template(
        "analysis.html",
        stocks=[s.to_dict() for s in stocks],
        strategies=[s.to_dict() for s in strategies],
    )


@app.route("/market")
def market_overview() -> str:
    """行情总览页面，展示指数和默认自选池实时行情。"""
    index_rows = []
    watchlist_rows = []
    error = None
    try:
        index_rows = _records(data_manager.get_realtime_data(INDEX_SYMBOLS))
        watchlist_rows = _records(data_manager.get_realtime_data(DEFAULT_WATCHLIST_SYMBOLS))
    except Exception as exc:
        app.logger.exception("行情总览数据源请求失败: %s", exc)
        error = PUBLIC_DATA_SOURCE_ERROR
    return render_template(
        "market.html",
        index_symbols=INDEX_SYMBOLS,
        watchlist_symbols=DEFAULT_WATCHLIST_SYMBOLS,
        index_rows=index_rows,
        watchlist_rows=watchlist_rows,
        error=error,
    )


@app.route("/boards")
def board_rotation() -> str:
    """板块轮动页面，展示 AKShare 板块和资金流。"""
    boards = []
    flows = []
    strength_rows = []
    error = None
    try:
        payload = _get_board_rotation_payload()
        boards = payload["boards"]
        flows = payload["flows"]
        strength_rows = payload["strength_rows"]
    except Exception as exc:
        app.logger.exception("板块轮动数据源请求失败: %s", exc)
        error = PUBLIC_DATA_SOURCE_ERROR
    return render_template(
        "boards.html",
        boards=boards,
        flows=flows,
        strength_rows=strength_rows,
        error=error,
    )


@app.route("/watchlists/ai-tech")
def ai_tech_watchlist() -> str:
    """AI 科技观察池页面，展示固定观察池的真实行情请求结果。"""
    rows = []
    error = None
    try:
        rows = _records(data_manager.get_realtime_data(AI_TECH_SYMBOLS))
    except Exception as exc:
        app.logger.exception("AI 科技观察池数据源请求失败: %s", exc)
        error = PUBLIC_DATA_SOURCE_ERROR
    return render_template(
        "ai_watchlist.html",
        symbols=AI_TECH_SYMBOLS,
        rows=rows,
        error=error,
    )


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    records = frame.to_dict("records")
    return [dict(row) for row in records]


def _get_board_rotation_payload() -> dict[str, list[dict[str, Any]]]:
    cache_key = "concept"
    now = time.monotonic()
    with _boards_cache_lock:
        cached = _boards_cache.get(cache_key)
        if cached and now - cached["created_at"] <= BOARD_CACHE_TTL_SECONDS:
            return cached["payload"]
        boards = _records(data_manager.get_boards(board_type="concept"))[
            :BOARD_TABLE_LIMIT
        ]
        flows = _records(data_manager.get_board_flows(board_type="concept"))[
            :BOARD_TABLE_LIMIT
        ]
        payload = {
            "boards": boards,
            "flows": flows,
            "strength_rows": _build_board_strength_rows(boards, flows),
        }
        _boards_cache[cache_key] = {"created_at": now, "payload": payload}
        return payload


def _build_board_strength_rows(
    boards: list[dict[str, Any]], flows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    flow_by_code = {row.get("board_code"): row for row in flows if row.get("board_code")}
    rows = []
    for board in boards:
        board_code = board.get("board_code")
        flow = flow_by_code.get(board_code, {})
        pct_chg = _to_float_or_zero(board.get("pct_chg"))
        net_inflow = _to_float_or_zero(flow.get("net_inflow"))
        amount = _to_float_or_zero(board.get("amount"))
        strength = pct_chg * 0.6 + _scaled_flow_score(net_inflow, amount) * 0.4
        rows.append(
            {
                "board_code": board_code,
                "board_name": board.get("board_name"),
                "pct_chg": pct_chg,
                "net_inflow": net_inflow,
                "strength": round(strength, 2),
                "source": board.get("source") or flow.get("source") or "-",
            }
        )
    return sorted(rows, key=lambda row: row["strength"], reverse=True)[:20]


def _to_float_or_zero(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _scaled_flow_score(net_inflow: float, amount: float) -> float:
    if amount <= 0:
        return 0.0
    return max(-10.0, min(10.0, net_inflow / amount * 100.0))


@app.route("/api/analyze", methods=["POST"])
def analyze_stock() -> Any:
    """分析单只股票的 API（仍保留给高级用户使用）"""
    data = request.get_json()
    stock_code = data.get("stock_code")
    strategy_type = data.get("strategy_type", "technical")

    if not stock_code:
        return jsonify({"error": "股票代码不能为空"}), 400

    try:
        # 获取或创建股票记录
        stock = Stock.query.filter_by(code=stock_code).first()
        if not stock:
            # 这里应该从数据源获取股票信息
            stock = Stock(code=stock_code, name=stock_code, market="SH")
            db.session.add(stock)
            db.session.commit()

        # 执行分析
        result = perform_analysis.delay(stock_code, strategy_type)

        return jsonify(
            {"task_id": result.id, "message": "分析任务已提交，请稍后查看结果"}
        )

    except Exception as e:
        app.logger.exception("提交单股分析任务失败: %s", e)
        return jsonify({"error": PUBLIC_ANALYSIS_ERROR}), 500


@app.route("/api/analyze_market", methods=["POST"])
def analyze_market() -> Any:
    """触发全市场自动选股分析的 API（后台线程执行，前端可查看真实进度和日志）。"""
    data = request.get_json(silent=True) or {}
    limit = data.get("limit", 50)

    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        limit_int = 50

    job_id = uuid.uuid4().hex
    with _market_jobs_lock:
        _market_jobs[job_id] = {
            "state": "PENDING",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "progress": 0.0,
            "processed": 0,
            "total": 0,
            "logs": [],
            "error": None,
            "result": None,
        }

    thread = threading.Thread(
        target=_market_analysis_worker, args=(job_id, limit_int), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/backtest", methods=["POST"])
def create_backtest() -> Any:
    """创建回测任务API（当前版本暂未实现回测逻辑）"""
    # 为避免误用，直接返回友好的错误提示
    return jsonify({"error": "回测功能尚未实现，当前版本仅支持选股分析"}), 501


@app.route("/api/market_status/<job_id>")
def get_market_status(job_id: str) -> Any:
    """查询全市场选股任务的实时进度和日志。"""
    with _market_jobs_lock:
        job = _market_jobs.get(job_id)
        if not job:
            return jsonify({"error": "任务不存在"}), 404

        data = {
            "job_id": job_id,
            "state": job["state"],
            "progress": float(job.get("progress") or 0.0),
            "processed": int(job.get("processed") or 0),
            "total": int(job.get("total") or 0),
            "logs": list(job.get("logs") or []),
            "error": job.get("error"),
        }
        # 任务完成时附带结果
        if job["state"] == "SUCCESS" and job.get("result") is not None:
            data["result"] = job["result"]

    return jsonify(data)


@app.route("/api/task/<task_id>")
def get_task_status(task_id: str) -> Any:
    """查询 Celery 异步任务状态。"""
    if not CELERY_TASK_ID_PATTERN.fullmatch(task_id):
        return jsonify({"error": "任务 ID 格式无效"}), 400
    task = celery.AsyncResult(task_id)
    if task.state == "PENDING":
        return jsonify({"state": task.state, "status": "任务等待执行"})
    if task.state == "PROGRESS":
        meta = task.info if isinstance(task.info, dict) else {}
        return jsonify({"state": task.state, "status": meta.get("status", "正在处理")})
    if task.state == "SUCCESS":
        result = task.result
        if isinstance(result, dict) and result.get("error"):
            return jsonify({"state": "FAILURE", "error": result["error"]})
        return jsonify({"state": task.state, "result": result})
    if task.state == "FAILURE":
        app.logger.error("Celery 任务失败: task_id=%s info=%s", task_id, task.info)
        return jsonify({"state": task.state, "error": PUBLIC_ANALYSIS_ERROR})
    return jsonify({"state": task.state, "status": str(task.info or task.state)})


@celery.task(bind=True)
def perform_analysis(self: Any, stock_code: str, strategy_type: str) -> dict:
    """异步执行单只股票分析（使用当前 QuantStrategy 和 data_manager）"""
    try:
        self.update_state(state="PROGRESS", meta={"status": "正在获取数据..."})

        # 获取或创建股票记录
        stock = Stock.query.filter_by(code=stock_code).first()
        if not stock:
            stock_name = stock_code
            stock_market = "未知"
            try:
                stocks_df = data_manager.get_stock_list()
                if stocks_df is not None and not stocks_df.empty:
                    row = stocks_df[stocks_df["symbol"] == stock_code]
                    if not row.empty:
                        stock_name = row.iloc[0]["name"]
                        stock_market = row.iloc[0]["market"]
            except Exception as e:
                app.logger.warning("获取股票元数据失败，使用股票代码作为名称: %s", e)

            stock = Stock(code=stock_code, name=stock_name, market=stock_market)
            db.session.add(stock)
            db.session.commit()

        # 组装策略所需的股票信息
        stock_info = {
            "symbol": stock.code,
            "name": stock.name,
            "market": stock.market,
        }

        self.update_state(state="PROGRESS", meta={"status": "正在分析..."})

        strategy = QuantStrategy()
        result = strategy.analyze_single_stock(stock_info)
        if not result:
            raise ValueError("策略未返回有效分析结果")

        # 计算与 CLI 一致的综合得分
        total_score = (
            result["prediction"] * 0.3
            + result["momentum_score"] * 0.2
            + result["explosion_score"] * 0.35
            + (1 - result["volatility"]) * 0.15
        )

        # 简单推荐逻辑
        if total_score >= 0.8:
            recommendation = "buy"
        elif total_score >= 0.6:
            recommendation = "hold"
        else:
            recommendation = "sell"

        # 保存分析结果到数据库
        analysis = AnalysisResult(
            stock_id=stock.id,
            analysis_date=date.today(),
            strategy_type=strategy_type,
            rsi=result.get("rsi"),
            macd=result.get("macd"),
            macd_signal=None,
            bollinger_upper=None,
            bollinger_lower=None,
            bollinger_middle=None,
            ml_prediction=result.get("prediction"),
            ml_confidence=result.get("prediction"),
            explosion_potential=result.get("explosion_score"),
            momentum_1m=None,
            momentum_3m=None,
            momentum_6m=None,
            volatility=result.get("volatility"),
            volume_ratio=None,
            total_score=total_score,
            recommendation=recommendation,
        )
        db.session.add(analysis)
        db.session.commit()

        # 返回给前端使用的结果
        def _to_float_safe(value: Any) -> float | None:
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        return {
            "stock_code": stock.code,
            "stock_name": stock.name,
            "strategy_type": strategy_type,
            "total_score": float(total_score),
            "recommendation": recommendation,
            "rsi": _to_float_safe(result.get("rsi")),
            "macd": _to_float_safe(result.get("macd")),
            "explosion_potential": _to_float_safe(result.get("explosion_score")),
            "volatility": _to_float_safe(result.get("volatility")),
            "analysis_id": analysis.id,
        }

    except Exception as e:
        db.session.rollback()
        app.logger.exception("单股分析任务执行失败: %s", e)
        return {"error": PUBLIC_ANALYSIS_ERROR}


def _update_market_job(
    job_id: str,
    *,
    state: str | None = None,
    processed: int | None = None,
    total: int | None = None,
    log: str | None = None,
    error: str | None = None,
    result: dict | None = None,
) -> None:
    """线程安全地更新全市场任务的状态与日志。"""
    with _market_jobs_lock:
        job = _market_jobs.get(job_id)
        if not job:
            return

        if state is not None:
            job["state"] = state
            if state == "RUNNING" and job.get("started_at") is None:
                job["started_at"] = datetime.utcnow().isoformat()
            if state in {"SUCCESS", "FAILURE"}:
                job["finished_at"] = datetime.utcnow().isoformat()

        if total is not None:
            job["total"] = int(total)
        if processed is not None:
            job["processed"] = int(processed)

        # 根据 processed/total 计算 progress
        t = int(job.get("total") or 0)
        p = int(job.get("processed") or 0)
        job["progress"] = float(p / t) if t > 0 else 0.0

        if log:
            logs = job.setdefault("logs", [])
            logs.append(log)
            # 只保留最近 200 条日志，避免无限增长
            if len(logs) > 200:
                job["logs"] = logs[-200:]

        if error is not None:
            job["error"] = error
        if result is not None:
            job["result"] = result


def _market_analysis_worker(job_id: str, limit: int) -> None:
    """在后台线程中执行全市场选股分析。"""
    with app.app_context():
        _update_market_job(job_id, state="RUNNING", log="全市场分析任务已启动...")

        def progress_callback(event: Dict[str, Any]) -> None:
            _update_market_job(
                job_id,
                processed=event.get("processed"),
                total=event.get("total"),
                log=event.get("log"),
            )

        result = _run_market_analysis_core(limit, progress_callback=progress_callback)

        if result.get("error"):
            _update_market_job(
                job_id,
                state="FAILURE",
                error=PUBLIC_ANALYSIS_ERROR,
                log="任务失败，请查看服务端日志。",
            )
        else:
            _update_market_job(
                job_id,
                state="SUCCESS",
                result=result,
                log="任务完成，已生成选股结果。",
            )


def _run_market_analysis_core(
    limit: int = 50, progress_callback: Any = None
) -> dict:
    """在当前进程中执行全市场量化选股分析的核心逻辑。"""
    try:
        strategy = QuantStrategy()
        selected_df = strategy.run_analysis(progress_callback=progress_callback)
        if selected_df is None or selected_df.empty:
            return {"error": "未找到符合条件的股票"}

        # 限制返回数量
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            limit_int = 50
        if limit_int > 0:
            selected_df = selected_df.head(limit_int)

        today = date.today()
        results_payload = []

        def _to_float_safe(value: Any) -> float | None:
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        for _, row in selected_df.iterrows():
            code = str(row.get("symbol") or "").strip()
            if not code:
                continue

            stock = Stock.query.filter_by(code=code).first()
            if not stock:
                stock = Stock(
                    code=code,
                    name=row.get("name") or code,
                    market=row.get("market") or "未知",
                )
                db.session.add(stock)
                db.session.flush()  # 立即获得 ID

            total_score_raw = row.get("total_score")
            total_score = _to_float_safe(total_score_raw) or 0.0

            if total_score >= 0.8:
                recommendation = "buy"
            elif total_score >= 0.6:
                recommendation = "hold"
            else:
                recommendation = "sell"

            analysis = AnalysisResult(
                stock_id=stock.id,
                analysis_date=today,
                strategy_type="quant_strategy",
                rsi=_to_float_safe(row.get("rsi")),
                macd=_to_float_safe(row.get("macd")),
                macd_signal=None,
                bollinger_upper=None,
                bollinger_lower=None,
                bollinger_middle=None,
                ml_prediction=_to_float_safe(row.get("prediction")),
                ml_confidence=_to_float_safe(row.get("prediction")),
                explosion_potential=_to_float_safe(row.get("explosion_score")),
                momentum_1m=_to_float_safe(row.get("momentum_1m")),
                momentum_3m=_to_float_safe(row.get("momentum_3m")),
                momentum_6m=_to_float_safe(row.get("momentum_6m")),
                volatility=_to_float_safe(row.get("volatility")),
                volume_ratio=None,
                total_score=total_score,
                recommendation=recommendation,
            )
            db.session.add(analysis)

            results_payload.append(
                {
                    "stock_code": code,
                    "stock_name": stock.name,
                    "market": stock.market,
                    "total_score": total_score,
                    "recommendation": recommendation,
                    "prediction": _to_float_safe(row.get("prediction")),
                    "momentum_score": _to_float_safe(row.get("momentum_score")),
                    "explosion_score": _to_float_safe(row.get("explosion_score")),
                    "volatility": _to_float_safe(row.get("volatility")),
                }
            )

        db.session.commit()
        return {"count": len(results_payload), "results": results_payload}

    except Exception as e:
        db.session.rollback()
        app.logger.exception("全市场分析任务执行失败: %s", e)
        return {"error": PUBLIC_ANALYSIS_ERROR}


@celery.task(bind=True)
def perform_market_analysis(self: Any, limit: int = 50) -> dict:
    """全市场自动选股分析任务（通过 Celery 调度，复用核心逻辑）"""
    self.update_state(state="PROGRESS", meta={"status": "正在运行全市场分析..."})
    return _run_market_analysis_core(limit)


@celery.task(bind=True)
def perform_backtest(self: Any, backtest_id: int) -> dict:
    """异步执行回测（占位实现，当前版本未提供具体回测逻辑）"""
    # 直接返回错误信息，避免误用旧接口
    return {"error": PUBLIC_BACKTEST_ERROR}
