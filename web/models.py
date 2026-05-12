from datetime import datetime

from web.app import db


class Stock(db.Model):
    __tablename__ = "stocks"

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False, index=True)
    name = db.Column(db.String(50), nullable=False)
    market = db.Column(db.String(10), nullable=False)  # SH, SZ
    industry = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    backtests = db.relationship("Backtest", backref="stock", lazy=True)
    analyses = db.relationship("AnalysisResult", backref="stock", lazy=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "market": self.market,
            "industry": self.industry,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Strategy(db.Model):
    __tablename__ = "strategies"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    type = db.Column(db.String(50), nullable=False)  # technical, ml, fundamental
    parameters = db.Column(db.JSON)  # Store strategy parameters as JSON
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    backtests = db.relationship("Backtest", backref="strategy", lazy=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "parameters": self.parameters,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Backtest(db.Model):
    __tablename__ = "backtests"

    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey("stocks.id"), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategies.id"), nullable=False)

    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    initial_capital = db.Column(db.Float, default=100000.0)

    # Results
    total_return = db.Column(db.Float)
    annual_return = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    total_trades = db.Column(db.Integer)

    # Detailed results as JSON
    results = db.Column(db.JSON)

    status = db.Column(
        db.String(20), default="pending"
    )  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stock_id": self.stock_id,
            "strategy_id": self.strategy_id,
            "stock": self.stock.to_dict() if self.stock else None,
            "strategy": self.strategy.to_dict() if self.strategy else None,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "results": self.results,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey("stocks.id"), nullable=False)

    analysis_date = db.Column(db.Date, nullable=False)
    strategy_type = db.Column(db.String(50), nullable=False)

    # Technical indicators
    rsi = db.Column(db.Float)
    macd = db.Column(db.Float)
    macd_signal = db.Column(db.Float)
    bollinger_upper = db.Column(db.Float)
    bollinger_lower = db.Column(db.Float)
    bollinger_middle = db.Column(db.Float)

    # ML predictions
    ml_prediction = db.Column(db.Float)
    ml_confidence = db.Column(db.Float)

    # Short-term signals
    explosion_potential = db.Column(db.Float)
    momentum_1m = db.Column(db.Float)
    momentum_3m = db.Column(db.Float)
    momentum_6m = db.Column(db.Float)

    # Risk metrics
    volatility = db.Column(db.Float)
    volume_ratio = db.Column(db.Float)

    # Overall score
    total_score = db.Column(db.Float)
    recommendation = db.Column(db.String(20))  # buy, hold, sell

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stock_id": self.stock_id,
            "stock": self.stock.to_dict() if self.stock else None,
            "analysis_date": self.analysis_date.isoformat(),
            "strategy_type": self.strategy_type,
            "rsi": self.rsi,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_lower": self.bollinger_lower,
            "bollinger_middle": self.bollinger_middle,
            "ml_prediction": self.ml_prediction,
            "ml_confidence": self.ml_confidence,
            "explosion_potential": self.explosion_potential,
            "momentum_1m": self.momentum_1m,
            "momentum_3m": self.momentum_3m,
            "momentum_6m": self.momentum_6m,
            "volatility": self.volatility,
            "volume_ratio": self.volume_ratio,
            "total_score": self.total_score,
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
        }
