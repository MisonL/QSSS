#!/usr/bin/env python3
"""QSSS交互式命令行界面 - 完全独立版"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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


class StandaloneInteractiveCLI:
    """完全独立的交互式CLI"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.last_results_file = None

    def print_banner(self):
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

    def print_menu(self):
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

    def get_choice(self):
        """获取用户选择"""
        if self.console and sys.stdin.isatty():
            return Prompt.ask(
                "\n请选择操作", choices=["0", "1", "2", "3", "4"], default="1"
            )
        else:
            # 非交互模式，直接返回1
            return "1"

    def run_analysis(self):
        """运行分析"""
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
            # 使用原有的qsss analyze命令
            cmd = [".venv/bin/qsss", "analyze", "--limit", str(limit)]
            if filename:
                cmd.extend(["--output", filename])
                self.last_results_file = filename

            # 运行命令
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="/Volumes/Work/code/QSSS"
            )

            if result.returncode == 0:
                if self.console:
                    self.console.print("\n[green] 分析完成！[/green]")
                    self.console.print("\n[bold cyan]=== 分析结果 ===[/bold cyan]")
                    self.console.print(result.stdout)
                else:
                    print("\n 分析完成！")
                    print("\n=== 分析结果 ===")
                    print(result.stdout)

                if filename and os.path.exists(filename):
                    if self.console:
                        self.console.print(
                            f"\n[green]结果已保存到: {Path(filename).absolute()}[/green]"
                        )
                    else:
                        print(f"\n结果已保存到: {Path(filename).absolute()}")
            else:
                if self.console:
                    self.console.print(f"[red]分析失败: {result.stderr}[/red]")
                else:
                    print(f"分析失败: {result.stderr}")

        except Exception as e:
            if self.console:
                self.console.print(f"[red]运行分析出错: {e}[/red]")
            else:
                print(f"运行分析出错: {e}")

    def view_last_results(self):
        """查看上次分析结果"""
        if not self.last_results_file or not os.path.exists(self.last_results_file):
            if self.console:
                self.console.print("[yellow]暂无分析结果文件，请先运行分析[/yellow]")
            else:
                print("暂无分析结果文件，请先运行分析")
            return

        try:
            # 读取CSV文件显示结果
            import pandas as pd

            results = pd.read_csv(self.last_results_file)

            if self.console:
                self.console.print(
                    f"\n[bold cyan]=== 分析结果 ({self.last_results_file}) ===[/bold cyan]"
                )
                self.console.print(f"[green]共找到 {len(results)} 只股票[/green]")

                # 显示前几行
                if len(results) > 0:
                    table = Table(show_header=True, header_style="bold magenta")

                    # 添加列
                    for col in results.columns[:6]:  # 只显示前6列
                        table.add_column(str(col), style="cyan")

                    # 添加数据行
                    for _, row in results.head(10).iterrows():
                        row_data = [str(val)[:20] for val in row[:6]]  # 限制长度
                        table.add_row(*row_data)

                    self.console.print(table)

                    if len(results) > 10:
                        self.console.print(
                            f"[yellow]... 还有 {len(results) - 10} 只股票[/yellow]"
                        )
            else:
                print(f"\n=== 分析结果 ({self.last_results_file}) ===")
                print(f"共找到 {len(results)} 只股票")
                print(results.head(10).to_string())

        except Exception as e:
            if self.console:
                self.console.print(f"[red]读取结果文件失败: {e}[/red]")
            else:
                print(f"读取结果文件失败: {e}")

    def show_config(self):
        """显示配置"""
        if self.console:
            self.console.print("\n[bold cyan]=== 系统配置 ===[/bold cyan]")
        else:
            print("\n=== 系统配置 ===")

        try:
            # 使用qsss config命令
            result = subprocess.run(
                [".venv/bin/qsss", "config"],
                capture_output=True,
                text=True,
                cwd="/Volumes/Work/code/QSSS",
            )

            if result.returncode == 0:
                if self.console:
                    self.console.print(result.stdout)
                else:
                    print(result.stdout)
            else:
                if self.console:
                    self.console.print(f"[red]获取配置失败: {result.stderr}[/red]")
                else:
                    print(f"获取配置失败: {result.stderr}")

        except Exception as e:
            if self.console:
                self.console.print(f"[red]运行配置命令出错: {e}[/red]")
            else:
                print(f"运行配置命令出错: {e}")

    def show_help(self):
        """显示帮助"""
        help_text = """
QSSS 量化选股系统 - 使用帮助

系统功能:
• AI驱动选股 - 使用LightGBM机器学习模型预测5日上涨概率
• 多因子评分 - 综合上涨概率、动量、爆发潜力、风险控制
• 技术指标分析 - RSI、MACD、布林带、成交量综合分析
• 智能筛选 - 自动识别优质标的和爆发潜力股

使用流程:
1. 选择"运行选股分析"开始分析
2. 设置分析参数(股票数量、是否保存等)
3. 查看分析结果和推荐标的
4. 可选择保存结果到CSV文件

结果说明:
• 上涨概率: AI模型预测的未来5日上涨概率(0-1)
• 动量得分: 综合1M/3M/6M动量指标(-1到1)
• 爆发潜力: 超短线爆发可能性评分(0-3)
• 15日均线: 技术面支撑位参考

注意事项:
• 首次运行需要连接数据源，可能需要一些时间
• 分析结果基于历史数据，不构成投资建议
• 投资有风险，决策需谨慎
        """

        if self.console:
            self.console.print(Panel(help_text, title="帮助信息", border_style="blue"))
        else:
            print(help_text)

    def run(self):
        """运行交互式CLI"""
        try:
            self.print_banner()

            # 检查是否为交互模式
            if not sys.stdin.isatty():
                # 非交互模式，直接运行分析
                self.console.print("[yellow]检测到非交互模式，直接运行分析...[/yellow]")
                self.run_analysis()
                return

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
                    if self.console and sys.stdin.isatty():
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


def main():
    """主入口点"""
    cli = StandaloneInteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
