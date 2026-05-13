#!/usr/bin/env python3
"""QSSS交互式演示 - 最终简化版"""

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


def main():
    """主函数 - 直接运行演示"""

    # 显示欢迎信息
    if console:
        console.print(
            "\n[bold cyan]+--======================================================"
            "======+[/bold cyan]"
        )
        console.print(
            "[bold cyan]|                    [/bold cyan]"
            "[bold yellow]QSSS 量化选股系统[/bold yellow]"
            "[bold cyan]                      |[/bold cyan]"
        )
        console.print(
            "[bold cyan]|              [/bold cyan]"
            "[bold green]Quantitative Stock Selection System v2.0[/bold green]"
            "[bold cyan]        |[/bold cyan]"
        )
        console.print(
            "[bold cyan]+--======================================================"
            "======+[/bold cyan]"
        )
    else:
        print("\n" + "=" * 70)
        print("                  QSSS 量化选股系统")
        print("            Quantitative Stock Selection System v2.0")
        print("=" * 70)

    print("\n 欢迎使用QSSS交互式量化选股系统！")
    print("\n 系统功能:")
    print("- AI驱动选股 - LightGBM机器学习预测5日上涨概率")
    print("- 多因子评分 - 综合上涨概率、动量、爆发潜力、风险控制")
    print("- 技术指标分析 - RSI、MACD、布林带、成交量综合分析")
    print("- 智能筛选 - 自动识别优质标的和爆发潜力股")

    print("\n 快速开始:")
    print("1. 直接运行: qsss")
    print("2. 传统模式: qsss analyze")
    print("3. 保存结果: qsss analyze --output results.csv")
    print("4. 查看配置: qsss config")
    print("5. 查看版本: qsss version")

    print("\n  重要提醒:")
    print("- 首次运行需要连接股票数据源，可能需要一些时间")
    print("- 分析结果基于历史数据，不构成投资建议")
    print("- 投资有风险，决策需谨慎")

    if console:
        # 显示模拟数据表格
        console.print("\n[bold cyan] 示例分析结果:[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("股票代码", style="cyan")
        table.add_column("股票名称", style="cyan")
        table.add_column("上涨概率", style="green")
        table.add_column("动量得分", style="yellow")
        table.add_column("爆发潜力", style="red")
        table.add_column("收盘价", style="blue")

        # 模拟数据
        sample_data = [
            ("000001", "平安银行", 0.72, 0.65, 2.1, 12.45),
            ("000002", "万科A", 0.68, 0.58, 1.8, 15.20),
            ("600036", "招商银行", 0.75, 0.72, 2.3, 35.60),
            ("000858", "五粮液", 0.69, 0.61, 1.9, 128.50),
            ("002415", "海康威视", 0.71, 0.67, 2.2, 28.90),
        ]

        for code, name, prob, momentum, explosion, price in sample_data:
            table.add_row(
                code,
                name,
                f"{prob:.3f}",
                f"{momentum:.3f}",
                f"{explosion:.1f}",
                f"CNY {price:.2f}",
            )

        console.print(table)

        console.print("\n[bold green] 交互式菜单已创建！[/bold green]")
        console.print("[yellow] 使用方法:[/yellow]")
        console.print(
            "  - 运行 '.venv/bin/python scripts/interactive_qsss.py' 使用完整交互式界面"
        )
        console.print("- 运行 'qsss analyze' 使用传统命令行模式")
        console.print("- 运行 'qsss' 启动交互式菜单 (需要修复数据源连接)")

        console.print("\n[bold cyan] 系统已就绪，开始您的量化投资之旅！[/bold cyan]")
    else:
        print("\n 示例分析结果:")
        print("股票代码  股票名称    上涨概率  动量得分  爆发潜力  收盘价")
        print("-" * 55)
        print("000001    平安银行    0.72      0.65      2.1       CNY 12.45")
        print("000002    万科A      0.68      0.58      1.8       CNY 15.20")
        print("600036    招商银行    0.75      0.72      2.3       CNY 35.60")
        print("000858    五粮液     0.69      0.61      1.9       CNY 128.50")
        print("002415    海康威视    0.71      0.67      2.2       CNY 28.90")

        print("\n 交互式菜单已创建！")
        print(" 使用方法:")
        print(
            "  - 运行 '.venv/bin/python scripts/interactive_qsss.py' 使用完整交互式界面"
        )
        print("- 运行 'qsss analyze' 使用传统命令行模式")
        print("- 运行 'qsss' 启动交互式菜单 (需要修复数据源连接)")

        print("\n 系统已就绪，开始您的量化投资之旅！")


if __name__ == "__main__":
    main()
