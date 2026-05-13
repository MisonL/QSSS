#!/usr/bin/env python3
"""
QSSS Web Application Runner
量化选股系统Web应用启动脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from web.config import apply_config, resolve_config_name  # noqa: E402


def init_database(app, db, stock_model, strategy_model) -> None:
    """初始化数据库"""
    with app.app_context():
        db.create_all()

        # 初始化策略数据
        if strategy_model.query.count() == 0:
            strategies = [
                strategy_model(
                    name="技术分析策略",
                    description="基于RSI、MACD、布林带等技术指标的综合分析",
                    type="technical",
                    parameters={
                        "rsi_period": 14,
                        "macd_fast": 12,
                        "macd_slow": 26,
                        "macd_signal": 9,
                        "bollinger_period": 20,
                        "bollinger_std": 2,
                    },
                ),
                strategy_model(
                    name="机器学习策略",
                    description="使用LightGBM模型预测未来5日收益率",
                    type="ml",
                    parameters={
                        "model_type": "lightgbm",
                        "prediction_days": 5,
                        "features": [
                            "rsi",
                            "macd",
                            "bollinger",
                            "volume_ratio",
                            "momentum",
                        ],
                        "confidence_threshold": 0.7,
                    },
                ),
                strategy_model(
                    name="短线爆发策略",
                    description="识别短线爆发潜力，结合量价关系分析",
                    type="short_term",
                    parameters={
                        "volume_threshold": 1.5,
                        "price_change_threshold": 0.05,
                        "momentum_period": 5,
                        "volatility_threshold": 0.02,
                    },
                ),
            ]

            for strategy in strategies:
                db.session.add(strategy)

            db.session.commit()
            print(" 策略数据初始化完成")

        # 初始化股票数据（示例）
        if stock_model.query.count() == 0:
            sample_stocks = [
                stock_model(
                    code="000001", name="平安银行", market="SZ", industry="银行"
                ),
                stock_model(
                    code="000002", name="万科A", market="SZ", industry="房地产"
                ),
                stock_model(
                    code="000858", name="五粮液", market="SZ", industry="白酒"
                ),
                stock_model(
                    code="600000", name="浦发银行", market="SH", industry="银行"
                ),
                stock_model(
                    code="600519", name="贵州茅台", market="SH", industry="白酒"
                ),
                stock_model(
                    code="601318", name="中国平安", market="SH", industry="保险"
                ),
                stock_model(
                    code="002415", name="海康威视", market="SZ", industry="安防"
                ),
                stock_model(
                    code="300750", name="宁德时代", market="SZ", industry="新能源"
                ),
                stock_model(
                    code="600036", name="招商银行", market="SH", industry="银行"
                ),
                stock_model(
                    code="000651", name="格力电器", market="SZ", industry="家电"
                ),
            ]

            for stock in sample_stocks:
                db.session.add(stock)

            db.session.commit()
            print(" 股票数据初始化完成")


def create_directories() -> None:
    """创建必要的目录"""
    directories = ["web/logs", "web/uploads", "web/static/cache"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(" 目录结构创建完成")


def main() -> None:
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="QSSS Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--init-db", action="store_true", help="初始化数据库")
    parser.add_argument(
        "--config",
        choices=["development", "testing", "production"],
        default="development",
        help="配置环境（完整名称：development/testing/production）",
    )
    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod"],
        help="环境别名（与 --config 等价）：dev=development, test=testing, prod=production",
    )

    args = parser.parse_args()
    config_name = resolve_config_name(args.env or args.config)
    os.environ["QSSS_WEB_ENV"] = config_name

    from web.app import app, celery  # noqa: E402
    from web.app import db  # noqa: E402
    from web.models import Stock, Strategy  # noqa: E402

    # 设置配置
    config_name = apply_config(app, config_name)
    celery.conf.update(app.config)

    # 确保在启动前导入模型和路由，完成 ORM 映射和路由注册
    import web.models  # noqa: F401
    import web.routes  # noqa: F401

    # 创建目录
    create_directories()

    # 初始化数据库
    if args.init_db:
        init_database(app, db, Stock, Strategy)
        return

    # 自动初始化数据库
    init_database(app, db, Stock, Strategy)

    # 启动应用
    if config_name == "production":
        # 生产环境使用Waitress
        try:
            from waitress import serve  # type: ignore[import-untyped]

            print(f" 启动生产服务器: http://{args.host}:{args.port}")
            serve(app, host=args.host, port=args.port)
        except ImportError:
            print("生产环境需要安装waitress: pip install waitress")
            sys.exit(1)
    else:
        # 开发环境使用Flask内置服务器
        print(f" 启动开发服务器: http://{args.host}:{args.port}")
        print("开发服务器仅用于开发环境")
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
