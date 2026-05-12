"""命令行接口"""

import time
from pathlib import Path

import click
from loguru import logger

from .config.settings import settings
from .core.strategy import QuantStrategy
from .data.manager import data_manager
from .interactive_cli import main as interactive_main
from .simple_interactive import main as simple_interactive_main


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """QSSS - 量化选股系统命令行工具

    无参数时启动交互式界面
    """
    if ctx.invoked_subcommand is None:
        # 无参数时启动简化交互式界面
        click.echo("正在启动QSSS交互式界面...")
        simple_interactive_main()
    else:
        # 有子命令时执行对应命令
        pass


@cli.command()
def interactive() -> None:
    """启动交互式菜单界面"""
    click.echo("正在启动QSSS交互式界面...")
    interactive_main()


@cli.command()
@click.option("--start-date", default="20220101", help="开始日期 (YYYYMMDD)")
@click.option("--output", "-o", help="输出文件路径")
@click.option("--limit", default=50, help="最大输出数量")
def analyze(start_date: str, output: str | None, limit: int) -> None:
    """运行量化分析"""
    logger.info("开始量化分析...")

    strategy = QuantStrategy()
    selected_stocks = strategy.run_analysis(start_date=start_date, limit=limit)

    if selected_stocks.empty:
        click.echo("未找到符合条件的股票")
        return

    # 计算15日均线
    selected_stocks["ma15"] = selected_stocks["symbol"].apply(
        lambda x: strategy.calculate_ma15(x, start_date=start_date)
    )

    # 重命名列
    columns_map = {
        "name": "股票名称",
        "symbol": "股票代码",
        "market": "交易所-板块",
        "prediction": "上涨概率",
        "momentum_score": "动量得分",
        "rsi": "RSI指标",
        "close": "收盘价",
        "ma15": "15日均线价格",
        "explosion_score": "爆发潜力值",
        "macd_status": "MACD状态",
    }

    display_df = selected_stocks[list(columns_map.keys())].copy()
    display_df.columns = [columns_map[col] for col in display_df.columns]

    # 显示结果
    click.echo("\n=== 选出的标的（前20名） ===")
    click.echo(display_df.head(limit).to_string(index=False))

    # 筛选15日均线在15元以内的股票
    low_price_stocks = display_df[display_df["15日均线价格"] <= 15]
    if not low_price_stocks.empty:
        click.echo("\n=== 15日均线在15元以内的标的 ===")
        click.echo(low_price_stocks.to_string(index=False))

    # 超短线爆发潜力股票
    explosion_stocks = selected_stocks[
        selected_stocks["explosion_score"] > 1.5
    ].sort_values("explosion_score", ascending=False)

    if not explosion_stocks.empty:
        explosion_display = explosion_stocks[list(columns_map.keys())].copy()
        explosion_display.columns = [
            columns_map[col] for col in explosion_display.columns
        ]

        click.echo("\n=== 超短线爆发潜力股票（前20名） ===")
        click.echo(explosion_display.head(limit).to_string(index=False))

        # 筛选15日均线在15元以内的爆发潜力股票
        low_price_explosion = explosion_display[explosion_display["15日均线价格"] <= 15]
        if not low_price_explosion.empty:
            click.echo("\n=== 15日均线在15元以内的爆发潜力股票 ===")
            click.echo(low_price_explosion.to_string(index=False))

    # 保存结果
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_stocks.to_csv(output_path, index=False, encoding="utf-8-sig")
        click.echo(f"\n结果已保存到: {output_path}")


@cli.command()
def version() -> None:
    """显示版本信息"""
    from . import __version__

    click.echo(f"QSSS版本: {__version__}")


@cli.command()
def config() -> None:
    """显示当前配置"""
    click.echo("当前配置:")
    click.echo(f"  主数据源: {settings.primary_data_source}")
    click.echo(f"  备用数据源: {settings.backup_data_source or '未配置'}")
    click.echo(f"  最小线程数: {settings.min_workers}")
    click.echo(f"  最大线程数: {settings.max_workers}")
    click.echo(f"  上涨概率阈值: {settings.min_prediction_threshold}")
    click.echo(f"  最小数据天数: {settings.min_data_days}")
    click.echo(f"  数据源降级阈值: {settings.datasource_failure_threshold} 次连续失败")
    click.echo(f"  数据源冷却时间: {settings.datasource_cooldown_seconds} 秒")


@cli.command(name="sources")
def show_sources() -> None:
    """显示已注册数据源及其健康状态"""
    click.echo("已注册数据源:")
    sources = data_manager.get_available_sources()
    if not sources:
        click.echo("  (无可用数据源)")
        return

    for name in sources:
        click.echo(f"  - {name}")

    health = data_manager.get_source_health()
    if not health:
        click.echo("\n尚无健康状态记录(尚未执行过数据请求)。")
        return

    click.echo("\n数据源健康状态:")
    now = time.time()
    for name in sources:
        info = health.get(name) or {}
        failures = info.get("consecutive_failures", 0)
        degraded_until = float(info.get("degraded_until") or 0.0)
        is_degraded = degraded_until > now
        status = "降级中" if is_degraded else "正常"
        click.echo(
            f"  - {name}: 状态={status}, 连续失败={failures}, "
            f"降级截止={degraded_until if degraded_until else 'N/A'}"
        )


def main() -> None:
    """CLI入口点"""
    cli()


if __name__ == "__main__":
    main()
