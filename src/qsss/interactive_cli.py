"""交互式命令行界面"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from .config.settings import settings

# 延迟导入避免初始化问题
from .core.strategy import QuantStrategy
from .data.manager import data_manager

console = Console()


class InteractiveCLI:
    """交互式命令行界面类"""

    def __init__(self) -> None:
        self.strategy: Optional[QuantStrategy] = None
        self.last_results = None
        self.current_config: dict[str, object] = {}

    def display_banner(self) -> None:
        """显示欢迎横幅"""
        banner = Text()
        banner.append(
            "+--============================================================+\n",
            style="cyan",
        )
        banner.append("|                    ", style="cyan")
        banner.append("QSSS 量化选股系统", style="bold yellow")
        banner.append("                      |\n", style="cyan")
        banner.append("|              ", style="cyan")
        banner.append("Quantitative Stock Selection System v2.0", style="green")
        banner.append("             |\n", style="cyan")
        banner.append(
            "+--============================================================+\n",
            style="cyan",
        )

        console.print(banner)

    def display_main_menu(self) -> str:
        """显示主菜单并返回用户选择"""
        console.print("\n[bold cyan]=== 主菜单 ===[/bold cyan]")

        menu_items = [
            "[1]  运行选股分析",
            "[2]  查看分析结果",
            "[3]  系统配置",
            "[4]  策略参数设置",
            "[5]  数据管理",
            "[6]  Web界面",
            "[7]  帮助信息",
            "[0]  退出系统",
        ]

        for item in menu_items:
            console.print(f"  {item}")

        choice = Prompt.ask(
            "\n请选择操作",
            choices=["0", "1", "2", "3", "4", "5", "6", "7"],
            default="1",
        )
        return choice

    def init_strategy(self) -> bool:
        """初始化策略，带进度显示"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                task = progress.add_task("正在初始化数据源...", total=None)

                # 延迟初始化策略
                self.strategy = QuantStrategy()

                progress.update(task, description="数据源连接成功！")
                time.sleep(0.5)

                return True

        except Exception as e:
            console.print(f"[red]初始化失败: {e}[/red]")
            return False

    def run_analysis_menu(self) -> None:
        """运行分析菜单"""
        console.print("\n[bold cyan]=== 选股分析 ===[/bold cyan]")

        if not self.strategy and not self.init_strategy():
            return

        # 分析参数设置
        console.print("\n[yellow]分析参数设置:[/yellow]")

        limit = IntPrompt.ask("最大输出股票数量", default=50)
        save_results = Confirm.ask("是否保存结果到文件", default=True)

        if save_results:
            default_filename = (
                f"qsss_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            filename = Prompt.ask("文件名", default=default_filename)
        else:
            filename = None

        # 开始分析
        console.print("\n[green]开始运行选股分析...[/green]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:

                task = progress.add_task("正在获取市场数据...", total=None)

                # 运行分析
                assert self.strategy is not None
                results = self.strategy.run_analysis()

                progress.update(task, description="分析完成！")
                time.sleep(0.5)

        except KeyboardInterrupt:
            console.print("\n[yellow]分析被用户中断[/yellow]")
            return
        except Exception as e:
            console.print(f"\n[red]分析过程中出现错误: {e}[/red]")
            logger.exception("分析失败")
            return

        if results.empty:
            console.print("[yellow]未找到符合条件的股票[/yellow]")
            return

        self.last_results = results

        # 显示结果
        self.display_analysis_results(results, limit)

        # 保存结果
        if filename:
            self.save_results(results, filename)

    def display_analysis_results(self, results: pd.DataFrame, limit: int) -> None:
        """显示分析结果"""
        console.print(
            f"\n[bold green]分析完成！找到 {len(results)} 只符合条件的股票[/bold green]"
        )

        # 综合评分前N名
        console.print(f"\n[cyan]=== 综合评分前{min(limit, 20)}名 ===[/cyan]")
        self.display_stock_table(results.head(min(limit, 20)))

        # 15元以内股票
        low_price = results[results.get("ma15", 0) <= 15]
        if not low_price.empty:
            console.print(
                f"\n[cyan]=== 15元以内优质股 (共{len(low_price)}只) ===[/cyan]"
            )
            self.display_stock_table(low_price.head(min(limit, 10)))

        # 爆发潜力股
        if "explosion_score" in results.columns:
            explosion = results[results["explosion_score"] > 1.5].sort_values(
                "explosion_score", ascending=False
            )
            if not explosion.empty:
                console.print(
                    f"\n[cyan]=== 爆发潜力股 (共{len(explosion)}只) ===[/cyan]"
                )
                self.display_stock_table(explosion.head(min(limit, 10)))

    def display_stock_table(self, data: pd.DataFrame) -> None:
        """显示股票表格"""
        if data.empty:
            console.print("[yellow]暂无数据[/yellow]")
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

        console.print(table)

    def save_results(self, results: pd.DataFrame, filename: str) -> None:
        """保存结果到文件"""
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results.to_csv(output_path, index=False, encoding="utf-8-sig")
            console.print(f"[green]结果已保存到: {output_path.absolute()}[/green]")

        except Exception as e:
            console.print(f"[red]保存文件失败: {e}[/red]")

    def show_config_menu(self) -> None:
        """显示配置菜单"""
        console.print("\n[bold cyan]=== 系统配置 ===[/bold cyan]")

        config_items = [
            "[1] 数据源设置",
            "[2] 性能参数",
            "[3] 选股条件",
            "[4] 缓存设置",
            "[5] 数据源健康状态",
            "[0] 返回主菜单",
        ]

        for item in config_items:
            console.print(f"  {item}")

        choice = Prompt.ask(
            "\n请选择", choices=["0", "1", "2", "3", "4", "5"], default="0"
        )

        if choice == "1":
            self.config_data_source()
        elif choice == "2":
            self.config_performance()
        elif choice == "3":
            self.config_selection_criteria()
        elif choice == "4":
            self.config_cache()
        elif choice == "5":
            self.show_data_source_health()

    def config_data_source(self) -> None:
        """配置数据源"""
        console.print("\n[yellow]当前数据源配置:[/yellow]")
        console.print(f"  主数据源: {settings.primary_data_source}")
        console.print(f"  备用数据源: {settings.backup_data_source}")

        if Confirm.ask("是否修改数据源配置"):
            sources = ["pytdx", "akshare"]
            primary = Prompt.ask(
                "选择主数据源", choices=sources, default=settings.primary_data_source
            )
            backup = Prompt.ask(
                "选择备用数据源", choices=sources, default=settings.backup_data_source
            )

            # 这里可以添加保存配置的逻辑
            console.print(
                f"[green]数据源配置已更新: 主={primary}, 备用={backup}[/green]"
            )

    def config_performance(self) -> None:
        """配置性能参数"""
        console.print("\n[yellow]当前性能配置:[/yellow]")
        console.print(f"  最小线程数: {settings.min_workers}")
        console.print(f"  最大线程数: {settings.max_workers}")
        console.print(f"  CPU阈值: {settings.cpu_threshold}%")
        console.print(f"  内存阈值: {settings.memory_threshold}%")

        if Confirm.ask("是否修改性能参数"):
            min_workers = IntPrompt.ask("最小线程数", default=settings.min_workers)
            max_workers = IntPrompt.ask("最大线程数", default=settings.max_workers)

            settings.min_workers = min_workers
            settings.max_workers = max_workers

            console.print("[green]性能参数已更新[/green]")

    def config_selection_criteria(self) -> None:
        """配置选股条件"""
        console.print("\n[yellow]当前选股条件:[/yellow]")
        console.print(f"  最小上涨概率: {settings.min_prediction_threshold}")
        console.print(f"  最小动量得分: {settings.min_momentum_score}")
        console.print(f"  RSI范围: {settings.rsi_range}")
        console.print(f"  最大波动率: {settings.max_volatility}")

        if Confirm.ask("是否修改选股条件"):
            threshold = Prompt.ask(
                "最小上涨概率", default=str(settings.min_prediction_threshold)
            )
            volatility = Prompt.ask("最大波动率", default=str(settings.max_volatility))

            try:
                settings.min_prediction_threshold = float(threshold)
                settings.max_volatility = float(volatility)
            except ValueError:
                console.print("[red]输入的选股参数无效，已忽略本次修改[/red]")
            else:
                console.print("[green]选股条件已更新[/green]")

    def config_cache(self) -> None:
        """配置缓存设置"""
        console.print("\n[yellow]当前缓存配置:[/yellow]")
        console.print(f"  缓存启用: {settings.cache_enabled}")
        console.print(f"  缓存目录: {settings.cache_dir}")
        console.print(f"  缓存时间: {settings.cache_ttl}秒")

        if Confirm.ask("是否修改缓存设置"):
            enabled = Confirm.ask("启用缓存", default=settings.cache_enabled)
            ttl = IntPrompt.ask("缓存时间(秒)", default=settings.cache_ttl)

            settings.cache_enabled = enabled
            settings.cache_ttl = ttl

            console.print("[green]缓存设置已更新[/green]")

    def show_data_source_health(self) -> None:
        """展示当前数据源及其健康状态。"""
        console.print("\n[bold cyan]=== 数据源健康状态 ===[/bold cyan]")

        sources = data_manager.get_available_sources()
        if not sources:
            console.print("[yellow]当前没有已注册的数据源[/yellow]")
            return

        health = data_manager.get_source_health()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("数据源")
        table.add_column("状态")
        table.add_column("连续失败次数")
        table.add_column("降级截止时间戳")

        now = time.time()
        for name in sources:
            info = health.get(name) or {}
            failures = info.get("consecutive_failures", 0)
            degraded_until = float(info.get("degraded_until") or 0.0)
            is_degraded = degraded_until > now
            status = "[red]降级中[/red]" if is_degraded else "[green]正常[/green]"
            table.add_row(
                name,
                status,
                str(failures),
                f"{degraded_until:.0f}" if degraded_until else "N/A",
            )

        console.print(table)

    def show_help(self) -> None:
        """显示帮助信息"""
        help_text = """
[bold cyan]QSSS 量化选股系统 - 使用帮助[/bold cyan]

[yellow]系统功能:[/yellow]
- AI驱动选股 - 使用LightGBM机器学习模型预测5日上涨概率
- 多因子评分 - 综合上涨概率、动量、爆发潜力、风险控制
- 技术指标分析 - RSI、MACD、布林带、成交量综合分析
- 智能筛选 - 自动识别优质标的和爆发潜力股

[yellow]使用流程:[/yellow]
1. 选择"运行选股分析"开始分析
2. 设置分析参数(股票数量、是否保存等)
3. 查看分析结果和推荐标的
4. 可选择保存结果到CSV文件

[yellow]结果说明:[/yellow]
- 上涨概率: AI模型预测的未来5日上涨概率(0-1)
- 动量得分: 综合1M/3M/6M动量指标(-1到1)
- 爆发潜力: 超短线爆发可能性评分(0-3)
- 15日均线: 技术面支撑位参考

[yellow]注意事项:[/yellow]
- 首次运行需要连接数据源，可能需要一些时间
- 分析结果基于历史数据，不构成投资建议
- 投资有风险，决策需谨慎
"""
        console.print(Panel(help_text, title="帮助信息", border_style="blue"))

        Prompt.ask("\n按回车键返回主菜单")

    def run(self) -> None:
        """运行交互式CLI"""
        try:
            self.display_banner()

            while True:
                choice = self.display_main_menu()

                if choice == "0":
                    console.print("\n[green]感谢使用QSSS量化选股系统！[/green]")
                    break
                elif choice == "1":
                    self.run_analysis_menu()
                elif choice == "2":
                    self.view_last_results()
                elif choice == "3":
                    self.show_config_menu()
                elif choice == "4":
                    self.show_strategy_settings()
                elif choice == "5":
                    self.show_data_management()
                elif choice == "6":
                    self.start_web_interface()
                elif choice == "7":
                    self.show_help()

                if choice != "0":
                    Prompt.ask("\n按回车键继续...")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]程序被用户中断[/yellow]")
        except Exception as e:
            console.print(f"\n[red]程序运行出错: {e}[/red]")

    def view_last_results(self) -> None:
        """查看上次分析结果"""
        if self.last_results is None:
            console.print("[yellow]暂无分析结果，请先运行分析[/yellow]")
            return

        console.print("\n[bold cyan]=== 上次分析结果 ===[/bold cyan]")
        self.display_analysis_results(self.last_results, 50)

    def show_strategy_settings(self) -> None:
        """显示策略设置"""
        console.print("\n[bold cyan]=== 策略参数设置 ===[/bold cyan]")
        console.print("[yellow]此功能正在开发中...[/yellow]")

    def show_data_management(self) -> None:
        """显示数据管理"""
        console.print("\n[bold cyan]=== 数据管理 ===[/bold cyan]")
        console.print("[yellow]此功能正在开发中...[/yellow]")

    def start_web_interface(self) -> None:
        """启动Web界面"""
        console.print("\n[bold cyan]=== Web界面 ===[/bold cyan]")
        console.print("[yellow]Web界面功能正在开发中...[/yellow]")
        console.print("请使用命令: python web/run.py --env dev")


def main() -> None:
    """交互式CLI入口点"""
    try:
        cli = InteractiveCLI()
        cli.run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]程序被用户中断，正在退出...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]程序运行出错: {e}[/red]")
        logger.exception("程序运行出错")
        sys.exit(1)


if __name__ == "__main__":
    main()
