#!/usr/bin/env python3
"""QSSS改进版交互式命令行界面"""

import subprocess
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.text import Text

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


class ImprovedInteractiveCLI:
    """改进版交互式CLI，更好的错误处理"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.last_results_file = None

    def print_banner(self):
        """显示欢迎横幅"""
        if console:
            banner = Text()
            banner.append(
                "+--============================================================+\n",
                style="bright_blue",
            )
            banner.append("|                    ", style="bright_blue")
            banner.append("QSSS 量化选股系统", style="bold yellow")
            banner.append("                      |\n", style="bright_blue")
            banner.append("|              ", style="bright_blue")
            banner.append(
                "Quantitative Stock Selection System v2.0", style="bright_green"
            )
            banner.append("             |\n", style="bright_blue")
            banner.append(
                "+--============================================================+\n",
                style="bright_blue",
            )
            console.print(banner)
        else:
            print("\n" + "=" * 70)
            print("                  QSSS 量化选股系统")
            print("            Quantitative Stock Selection System v2.0")
            print("=" * 70)

    def print_menu(self):
        """显示主菜单"""
        if console:
            console.print("\n[bold cyan]=== 主菜单 ===[/bold cyan]")
            console.print("[1]  运行选股分析")
            console.print("[2]  查看分析结果")
            console.print("[3]  系统配置")
            console.print("[4]  帮助信息")
            console.print("[0]  退出系统")
        else:
            print("\n=== QSSS 主菜单 ===")
            print("1. 运行选股分析")
            print("2. 查看分析结果")
            print("3. 系统配置")
            print("4. 帮助信息")
            print("0. 退出系统")

    def get_choice(self):
        """获取用户选择"""
        if console:
            return Prompt.ask(
                "\n请选择操作", choices=["0", "1", "2", "3", "4"], default="1"
            )
        else:
            return input("\n请选择操作 (0-4): ").strip()

    def run_analysis(self):
        """运行分析，带进度显示"""
        if console:
            console.print("\n[bold cyan]=== 选股分析 ===[/bold cyan]")
            console.print("[yellow]正在初始化数据源，请稍候...[/yellow]")
        else:
            print("\n=== 选股分析 ===")
            print("正在初始化数据源，请稍候...")

        # 使用进度条显示初始化过程
        try:
            if console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:

                    task = progress.add_task("正在连接数据源...", total=None)
                    time.sleep(0.5)

                    # 运行qsss analyze命令
                    limit = 20  # 默认分析20只股票
                    cmd = [".venv/bin/qsss", "analyze", "--limit", str(limit)]

                    progress.update(task, description="正在获取股票列表...")

            else:
                limit = 20
                cmd = [".venv/bin/qsss", "analyze", "--limit", str(limit)]

            # 运行分析命令
            if console:
                console.print("[green] 数据源连接成功！[/green]")
                console.print(f"[cyan]开始分析 {limit} 只股票...[/cyan]")

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="/Volumes/Work/code/QSSS"
            )

            if result.returncode == 0:
                if console:
                    console.print("\n[bold green] 分析完成！[/bold green]")
                    console.print("\n[bold cyan]=== 分析结果 ===[/bold cyan]")

                    # 解析输出结果
                    lines = result.stdout.strip().split("\n")
                    result_lines = []
                    in_results = False

                    for line in lines:
                        if "===" in line and "选出的标的" in line:
                            in_results = True
                        elif in_results and line.strip() and not line.startswith("==="):
                            result_lines.append(line)
                        elif in_results and line.startswith("==="):
                            break

                    if result_lines:
                        console.print("[green]找到符合条件的股票:[/green]")
                        for line in result_lines[:10]:  # 显示前10行
                            console.print(f"  {line}")
                        if len(result_lines) > 10:
                            console.print(
                                f"[yellow]... 还有 {len(result_lines) - 10} 只股票[/yellow]"
                            )
                    else:
                        console.print(result.stdout)

                else:
                    print(" 分析完成！")
                    print("\n=== 分析结果 ===")
                    print(result.stdout)

            else:
                if console:
                    console.print("[red] 分析失败[/red]")
                    console.print(f"[yellow]错误信息: {result.stderr}[/yellow]")
                    console.print(
                        "[cyan]提示: 部分股票数据可能不可用，这是正常现象[/cyan]"
                    )
                    console.print("[cyan]系统已自动过滤无效数据[/cyan]")
                else:
                    print(" 分析失败")
                    print(f"错误信息: {result.stderr}")

        except Exception as e:
            if console:
                console.print(f"[red] 运行分析出错: {e}[/red]")
                console.print(
                    "[yellow]提示: 请确保网络连接正常，数据源服务可用[/yellow]"
                )
                console.print("[cyan]您可以尝试:[/cyan]")
                console.print("• 检查网络连接")
                console.print("• 稍后再试")
                console.print("• 使用传统命令: .venv/bin/qsss analyze")
            else:
                print(f" 运行分析出错: {e}")
                print("提示: 请确保网络连接正常，数据源服务可用")

    def view_last_results(self):
        """查看上次分析结果"""
        if console:
            console.print("\n[bold cyan]=== 查看分析结果 ===[/bold cyan]")
            console.print("[yellow]此功能需要您之前运行过分析并保存了结果[/yellow]")
            console.print("[cyan]建议: 运行分析时选择保存结果到CSV文件[/cyan]")
        else:
            print("\n=== 查看分析结果 ===")
            print("此功能需要您之前运行过分析并保存了结果")
            print("建议: 运行分析时选择保存结果到CSV文件")

    def show_config(self):
        """显示配置"""
        if console:
            console.print("\n[bold cyan]=== 系统配置 ===[/bold cyan]")
        else:
            print("\n=== 系统配置 ===")

        try:
            result = subprocess.run(
                [".venv/bin/qsss", "config"],
                capture_output=True,
                text=True,
                cwd="/Volumes/Work/code/QSSS",
            )

            if result.returncode == 0:
                if console:
                    console.print(result.stdout)
                else:
                    print(result.stdout)
            else:
                if console:
                    console.print(f"[red]获取配置失败: {result.stderr}[/red]")
                else:
                    print(f"获取配置失败: {result.stderr}")

        except Exception as e:
            if console:
                console.print(f"[red]运行配置命令出错: {e}[/red]")
            else:
                print(f"运行配置命令出错: {e}")

    def show_help(self):
        """显示帮助"""
        help_text = """
[bold cyan]QSSS 量化选股系统 - 使用帮助[/bold cyan]

[bold yellow]系统功能:[/bold yellow]
• AI驱动选股 - 使用LightGBM机器学习模型预测5日上涨概率
• 多因子评分 - 综合上涨概率、动量、爆发潜力、风险控制
• 技术指标分析 - RSI、MACD、布林带、成交量综合分析
• 智能筛选 - 自动识别优质标的和爆发潜力股

[bold yellow]使用流程:[/bold yellow]
1. 选择"运行选股分析"开始分析
2. 系统会自动获取市场数据（可能需要一些时间）
3. 查看分析结果和推荐标的
4. 可选择保存结果到CSV文件

[bold yellow]结果说明:[/bold yellow]
• [green]上涨概率[/green] - AI模型预测的未来5日上涨概率(0-1)
• [yellow]动量得分[/yellow] - 综合1M/3M/6M动量指标(-1到1)
• [red]爆发潜力[/red] - 超短线爆发可能性评分(0-3)
• [blue]15日均线[/blue] - 技术面支撑位参考

[bold yellow]注意事项:[/bold yellow]
 首次运行需要连接数据源，可能需要一些时间
 分析结果基于历史数据，不构成投资建议
 投资有风险，决策需谨慎

[bold cyan]故障排除:[/bold cyan]
• 如果连接失败，请检查网络连接
• 部分股票数据可能不可用，这是正常现象
• 系统会自动过滤无效数据并继续分析
        """

        if console:
            console.print(Panel(help_text, title="帮助信息", border_style="blue"))
        else:
            print(help_text)

    def run(self):
        """运行交互式CLI"""
        try:
            self.print_banner()

            while True:
                self.print_menu()
                choice = self.get_choice()

                if choice == "0":
                    if console:
                        console.print(
                            "\n[bold green] 感谢使用QSSS量化选股系统！[/bold green]"
                        )
                        console.print("[cyan]祝您投资顺利！[/cyan]")
                    else:
                        print("\n 感谢使用QSSS量化选股系统！")
                        print("祝您投资顺利！")
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
                    if console:
                        Prompt.ask("\n按回车键继续...")
                    else:
                        input("\n按回车键继续...")

        except KeyboardInterrupt:
            if console:
                console.print("\n\n[yellow] 程序被用户中断，感谢使用！[/yellow]")
            else:
                print("\n\n 程序被用户中断，感谢使用！")
        except Exception as e:
            if console:
                console.print(f"\n[red] 程序运行出错: {e}[/red]")
                console.print("[cyan]请检查系统配置或联系技术支持[/cyan]")
            else:
                print(f"\n 程序运行出错: {e}")
                print("请检查系统配置或联系技术支持")


def main():
    """主入口点"""
    cli = ImprovedInteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
