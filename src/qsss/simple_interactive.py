#!/usr/bin/env python3
"""QSSS交互式命令行界面 - 简化版"""

import sys
from pathlib import Path
from typing import Any, Optional

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from qsss.config.settings import settings
    from qsss.core.strategy import QuantStrategy

    QSSS_AVAILABLE = True
except ImportError as e:
    print(f"导入QSSS模块失败: {e}")
    QSSS_AVAILABLE = False
    sys.exit(1)


class SimpleInteractiveCLI:
    """简化的交互式CLI"""

    def __init__(self) -> None:
        self.strategy: Optional[QuantStrategy] = None
        self.last_results = None
        self.console: Optional[Console] = Console() if RICH_AVAILABLE else None

    def print_banner(self) -> None:
        """显示欢迎横幅"""
        banner = """
+--============================================================+
|                    QSSS 量化选股系统                         |
|              Quantitative Stock Selection System v2.0        |
+--============================================================+
        """
        if self.console:
            self.console.print(banner, style="cyan")
        else:
            print(banner)

    def print_menu(self) -> None:
        """显示主菜单"""
        menu_text = """
[bold cyan]=== 主菜单 ===[/bold cyan]
  [1]  运行选股分析
  [2]  查看分析结果
  [3]  系统配置
  [4]  帮助信息
  [0]  退出系统
        """

        if self.console:
            self.console.print(menu_text)
        else:
            print("\n=== QSSS 主菜单 ===")
            print("1. 运行选股分析")
            print("2. 查看分析结果")
            print("3. 系统配置")
            print("4. 帮助信息")
            print("0. 退出系统")

    def get_choice(self) -> str:
        """获取用户选择"""
        if self.console:
            return Prompt.ask(
                "\n请选择操作", choices=["0", "1", "2", "3", "4"], default="1"
            )
        else:
            return input("\n请选择操作 (0-4): ").strip()

    def init_strategy(self) -> bool:
        """初始化策略"""
        try:
            if self.console:
                self.console.print("[yellow]正在初始化数据源...[/yellow]")
            else:
                print("正在初始化数据源...")

            self.strategy = QuantStrategy()

            if self.console:
                self.console.print("[green] 数据源初始化成功！[/green]")
            else:
                print(" 数据源初始化成功！")

            return True

        except Exception as e:
            if self.console:
                self.console.print(f"[red] 初始化失败: {e}[/red]")
                self.console.print(
                    "[yellow]提示: 这可能是由于数据源连接问题，请确保网络连接正常[/yellow]"
                )
            else:
                print(f" 初始化失败: {e}")
                print("提示: 这可能是由于数据源连接问题，请确保网络连接正常")
            return False

    def run_analysis(self) -> None:
        """运行分析"""
        if not self.strategy and not self.init_strategy():
            return

        if self.console:
            self.console.print("\n[bold cyan]=== 选股分析 ===[/bold cyan]")
            limit = IntPrompt.ask("最大输出股票数量", default=50)
            save_file = Confirm.ask("是否保存结果到文件", default=True)
        else:
            print("\n=== 选股分析 ===")
            try:
                limit = int(input("最大输出股票数量 (默认50): ") or "50")
            except ValueError:
                limit = 50
            save_input = input("是否保存结果到文件? (y/N): ").strip().lower()
            save_file = save_input in ["y", "yes"]

        if save_file:
            from datetime import datetime

            default_filename = (
                f"qsss_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            if self.console:
                filename = Prompt.ask("文件名", default=default_filename)
            else:
                filename = (
                    input(f"文件名 (默认: {default_filename}): ").strip()
                    or default_filename
                )
        else:
            filename = None

        # 开始分析
        if self.console:
            self.console.print("\n[green]开始运行选股分析...[/green]")
            self.console.print("[yellow]正在获取市场数据，请稍候...[/yellow]")
        else:
            print("\n开始运行选股分析...")
            print("正在获取市场数据，请稍候...")

        try:
            assert self.strategy is not None
            results = self.strategy.run_analysis()

            if results.empty:
                if self.console:
                    self.console.print("[yellow]未找到符合条件的股票[/yellow]")
                else:
                    print("未找到符合条件的股票")
                return

            self.last_results = results

            # 显示结果
            self.display_results(results, limit)

            # 保存结果
            if filename:
                self.save_results(results, filename)

        except Exception as e:
            if self.console:
                self.console.print(f"[red]分析过程出错: {e}[/red]")
                self.console.print(
                    "[yellow]提示: 部分股票数据可能不可用，这是正常现象[/yellow]"
                )
                self.console.print("[yellow]系统会继续处理其他股票数据[/yellow]")
            else:
                print(f"分析过程出错: {e}")
                print("提示: 部分股票数据可能不可用，这是正常现象")
                print("系统会继续处理其他股票数据")

    def display_results(self, results: Any, limit: int) -> None:
        """显示结果"""
        if self.console:
            self.console.print(
                f"\n[bold green]分析完成！找到 {len(results)} 只符合条件的股票[/bold green]"
            )

            # 综合评分前N名
            self.console.print(f"\n[cyan]=== 综合评分前{min(limit, 20)}名 ===[/cyan]")
            self.display_stock_table(results.head(min(limit, 20)))

            # 15元以内股票
            if "ma15" in results.columns:
                low_price = results[results["ma15"] <= 15]
                if not low_price.empty:
                    self.console.print(
                        f"\n[cyan]=== 15元以内优质股 (共{len(low_price)}只) ===[/cyan]"
                    )
                    self.display_stock_table(low_price.head(min(limit, 10)))

            # 爆发潜力股
            if "explosion_score" in results.columns:
                explosion = results[results["explosion_score"] > 1.5].sort_values(
                    "explosion_score", ascending=False
                )
                if not explosion.empty:
                    self.console.print(
                        f"\n[cyan]=== 爆发潜力股 (共{len(explosion)}只) ===[/cyan]"
                    )
                    self.display_stock_table(explosion.head(min(limit, 10)))
        else:
            print(f"\n分析完成！找到 {len(results)} 只符合条件的股票")
            print(f"\n=== 综合评分前{min(limit, 20)}名 ===")
            self.print_simple_results(results.head(min(limit, 20)))

    def display_stock_table(self, data: Any) -> None:
        """显示股票表格（Rich版）"""
        if data.empty:
            assert self.console is not None
            self.console.print("[yellow]暂无数据[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta")

        # 定义显示列
        display_columns = {
            "symbol": "股票代码",
            "name": "股票名称",
            "prediction": "上涨概率",
            "momentum_score": "动量得分",
            "explosion_score": "爆发潜力",
            "close": "收盘价",
            "ma15": "15日均线",
        }

        # 添加表头
        for col, name in display_columns.items():
            if col in data.columns:
                table.add_column(name, style="cyan", no_wrap=True)

        # 添加数据行
        for _, row in data.iterrows():
            row_data = []
            for col, _ in display_columns.items():
                if col in row:
                    if col in ["prediction", "momentum_score", "explosion_score"]:
                        row_data.append(f"{row[col]:.3f}")
                    elif col in ["close", "ma15"]:
                        row_data.append(f"{row[col]:.2f}")
                    else:
                        row_data.append(str(row[col]))
                else:
                    row_data.append("")
            table.add_row(*row_data)

        assert self.console is not None
        self.console.print(table)

    def print_simple_results(self, data: Any) -> None:
        """简单显示结果（无Rich版）"""
        if data.empty:
            print("暂无数据")
            return

        # 显示关键列
        display_cols = ["symbol", "name", "prediction", "momentum_score"]
        if "explosion_score" in data.columns:
            display_cols.append("explosion_score")
        if "close" in data.columns:
            display_cols.append("close")

        # 打印表头
        headers = []
        for col in display_cols:
            if col == "symbol":
                headers.append("股票代码")
            elif col == "name":
                headers.append("股票名称")
            elif col == "prediction":
                headers.append("上涨概率")
            elif col == "momentum_score":
                headers.append("动量得分")
            elif col == "explosion_score":
                headers.append("爆发潜力")
            elif col == "close":
                headers.append("收盘价")

        print("".join(f"{h:<12}" for h in headers))
        print("-" * (len(headers) * 13))

        # 打印数据
        for _, row in data.iterrows():
            row_data = []
            for col in display_cols:
                if col in row:
                    if col in ["prediction", "momentum_score", "explosion_score"]:
                        row_data.append(f"{row[col]:<12.3f}")
                    else:
                        row_data.append(f"{str(row[col]):<12}")
                else:
                    row_data.append(" " * 12)
            print("".join(row_data))

    def save_results(self, results: Any, filename: str) -> None:
        """保存结果"""
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results.to_csv(output_path, index=False, encoding="utf-8-sig")

            if self.console:
                self.console.print(
                    f"[green]结果已保存到: {output_path.absolute()}[/green]"
                )
            else:
                print(f"结果已保存到: {output_path.absolute()}")

        except Exception as e:
            if self.console:
                self.console.print(f"[red]保存文件失败: {e}[/red]")
            else:
                print(f"保存文件失败: {e}")

    def show_config(self) -> None:
        """显示配置和数据源状态"""
        if self.console:
            from qsss.data.manager import data_manager

            self.console.print("\n[bold cyan]=== 系统配置 ===[/bold cyan]")
            self.console.print(f"主数据源: {settings.primary_data_source}")
            self.console.print(f"备用数据源: {settings.backup_data_source or '未配置'}")
            self.console.print(f"最小线程数: {settings.min_workers}")
            self.console.print(f"最大线程数: {settings.max_workers}")
            self.console.print(f"上涨概率阈值: {settings.min_prediction_threshold}")
            self.console.print(f"最小数据天数: {settings.min_data_days}")
            self.console.print(
                f"数据源降级阈值: {settings.datasource_failure_threshold} 次连续失败"
            )
            self.console.print(
                f"数据源冷却时间: {settings.datasource_cooldown_seconds} 秒"
            )

            # 简单展示数据源列表
            sources = data_manager.get_available_sources()
            self.console.print(
                "\n[cyan]已注册数据源:[/cyan] " + ", ".join(sources)
                if sources
                else "(无)"
            )
        else:
            print("\n=== 系统配置 ===")
            print(f"主数据源: {settings.primary_data_source}")
            print(f"备用数据源: {settings.backup_data_source or '未配置'}")
            print(f"最小线程数: {settings.min_workers}")
            print(f"最大线程数: {settings.max_workers}")
            print(f"上涨概率阈值: {settings.min_prediction_threshold}")
            print(f"最小数据天数: {settings.min_data_days}")

    def show_help(self) -> None:
        """显示帮助"""
        help_text = """
QSSS 量化选股系统 - 使用帮助

系统功能:
- AI驱动选股 - 使用LightGBM机器学习模型预测5日上涨概率
- 多因子评分 - 综合上涨概率、动量、爆发潜力、风险控制
- 技术指标分析 - RSI、MACD、布林带、成交量综合分析
- 智能筛选 - 自动识别优质标的和爆发潜力股

使用流程:
1. 选择"运行选股分析"开始分析
2. 设置分析参数(股票数量、是否保存等)
3. 查看分析结果和推荐标的
4. 可选择保存结果到CSV文件

结果说明:
- 上涨概率: AI模型预测的未来5日上涨概率(0-1)
- 动量得分: 综合1M/3M/6M动量指标(-1到1)
- 爆发潜力: 超短线爆发可能性评分(0-3)
- 15日均线: 技术面支撑位参考

注意事项:
- 首次运行需要连接数据源，可能需要一些时间
- 分析结果基于历史数据，不构成投资建议
- 投资有风险，决策需谨慎
        """

        if self.console:
            self.console.print(Panel(help_text, title="帮助信息", border_style="blue"))
        else:
            print(help_text)

    def view_last_results(self) -> None:
        """查看上次结果"""
        if self.last_results is None:
            if self.console:
                self.console.print("[yellow]暂无分析结果，请先运行分析[/yellow]")
            else:
                print("暂无分析结果，请先运行分析")
            return

        if self.console:
            self.console.print("\n[bold cyan]=== 上次分析结果 ===[/bold cyan]")
        else:
            print("\n=== 上次分析结果 ===")
        self.display_results(self.last_results, 50)

    def run(self) -> None:
        """运行交互式CLI"""
        try:
            self.print_banner()

            while True:
                self.print_menu()
                choice = self.get_choice()

                if choice == "0":
                    if self.console:
                        self.console.print(
                            "\n[green]感谢使用QSSS量化选股系统！[/green]"
                        )
                    else:
                        print("\n感谢使用QSSS量化选股系统！")
                    break
                elif choice == "1":
                    self.run_analysis()
                elif choice == "2":
                    self.view_last_results()
                elif choice == "3":
                    self.show_config()
                elif choice == "4":
                    self.show_help()

                if choice != "0":
                    if self.console:
                        Prompt.ask("\n按回车键继续...")
                    else:
                        input("\n按回车键继续...")

        except KeyboardInterrupt:
            if self.console:
                self.console.print("\n\n[yellow]程序被用户中断[/yellow]")
            else:
                print("\n\n程序被用户中断")
        except Exception as e:
            if self.console:
                self.console.print(f"\n[red]程序运行出错: {e}[/red]")
            else:
                print(f"\n程序运行出错: {e}")


def main() -> None:
    """主入口点"""
    cli = SimpleInteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
