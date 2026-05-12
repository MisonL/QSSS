#!/usr/bin/env python3
"""
QSSS性能测试脚本
对比原始版本和优化版本的性能差异
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入性能测试所需的模块
try:
    import psutil
except ImportError:
    print("警告: 未安装psutil，部分内存监控功能将不可用")
    psutil = None

# 可选导入matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("警告: 未安装matplotlib，图表生成功能将不可用")
    plt = None

from src.qsss.core.optimized_strategy import OptimizedQuantStrategy  # noqa: E402
from src.qsss.core.strategy import QuantStrategy as OriginalStrategy  # noqa: E402
from src.qsss.data.manager import data_manager  # noqa: E402


class PerformanceTester:
    """性能测试器"""

    def __init__(self):
        self.test_results = {"original": {}, "optimized": {}, "comparison": {}}

    def test_original_strategy(
        self, stock_list: pd.DataFrame, sample_size: int = 50
    ) -> Dict[str, Any]:
        """测试原始策略"""
        logger.info(f"测试原始策略，样本数量: {sample_size}")

        # 取样
        sample_stocks = stock_list.head(sample_size)

        start_time = time.time()
        memory_start = psutil.virtual_memory().percent if psutil else 0

        try:
            strategy = OriginalStrategy()
            # 限制分析的股票数量，避免测试时间过长
            original_get_stock_list = strategy.get_stock_list

            def limited_get_stock_list():
                stocks = original_get_stock_list()
                return stocks.head(min(20, len(stocks)))  # 限制最多分析20只股票

            strategy.get_stock_list = limited_get_stock_list

            results = strategy.run_analysis()

            end_time = time.time()
            memory_end = psutil.virtual_memory().percent if psutil else 0

            result = {
                "execution_time": end_time - start_time,
                "memory_usage": memory_end - memory_start,
                "processed_stocks": min(20, len(sample_stocks)),
                "selected_stocks": len(results) if not results.empty else 0,
                "success": True,
                "error": None,
            }

            logger.info(
                f"原始策略测试完成 - 耗时: {result['execution_time']:.2f}秒, "
                f"选中股票: {result['selected_stocks']}"
            )

            return result

        except Exception as e:
            logger.error(f"原始策略测试失败: {e}")
            return {
                "execution_time": time.time() - start_time,
                "memory_usage": 0,
                "processed_stocks": len(sample_stocks),
                "selected_stocks": 0,
                "success": False,
                "error": str(e),
            }

    def test_optimized_strategy(
        self, stock_list: pd.DataFrame, sample_size: int = 50
    ) -> Dict[str, Any]:
        """测试优化策略"""
        logger.info(f"测试优化策略，样本数量: {sample_size}")

        # 取样
        sample_stocks = stock_list.head(sample_size)

        start_time = time.time()
        memory_start = psutil.virtual_memory().percent if psutil else 0

        try:
            with OptimizedQuantStrategy(
                use_distributed=False, use_cache=True
            ) as strategy:
                # 限制分析的股票数量，避免测试时间过长
                original_get_stock_list = strategy.get_stock_list

                def limited_get_stock_list():
                    stocks = original_get_stock_list()
                    return stocks.head(min(50, len(stocks)))  # 限制最多分析50只股票

                strategy.get_stock_list = limited_get_stock_list

                results = strategy.run_distributed_analysis(sample_stocks)

                end_time = time.time()
                memory_end = psutil.virtual_memory().percent if psutil else 0

                # 获取分析统计
                summary = strategy.get_analysis_summary()

                result = {
                    "execution_time": end_time - start_time,
                    "memory_usage": memory_end - memory_start,
                    "processed_stocks": min(50, len(sample_stocks)),
                    "selected_stocks": len(results) if not results.empty else 0,
                    "cache_hits": summary.get("analysis_stats", {}).get(
                        "cache_hits", 0
                    ),
                    "cache_misses": summary.get("analysis_stats", {}).get(
                        "cache_misses", 0
                    ),
                    "hit_rate": (
                        summary.get("cache_stats", {}).get("hit_rate", 0)
                        if summary.get("cache_stats")
                        else 0
                    ),
                    "success": True,
                    "error": None,
                    "summary": summary,
                }

                logger.info(
                    f"优化策略测试完成 - 耗时: {result['execution_time']:.2f}秒, "
                    f"选中股票: {result['selected_stocks']}, "
                    f"缓存命中率: {result['hit_rate']:.1%}"
                )

                return result

        except Exception as e:
            logger.error(f"优化策略测试失败: {e}")
            return {
                "execution_time": time.time() - start_time,
                "memory_usage": 0,
                "processed_stocks": len(sample_stocks),
                "selected_stocks": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0,
                "success": False,
                "error": str(e),
            }

    def test_data_loading_performance(self) -> Dict[str, Any]:
        """测试数据加载性能"""
        logger.info("测试数据加载性能")

        test_symbols = ["000001", "000002", "600000", "600036", "000858"]

        # 测试原始数据加载
        start_time = time.time()
        for symbol in test_symbols:
            data_manager.get_daily_data(symbol, "20240101")
        original_time = time.time() - start_time

        # 测试缓存数据加载
        start_time = time.time()
        for symbol in test_symbols:
            data_manager.get_daily_data(symbol, "20240101")
        cached_time = time.time() - start_time

        return {
            "original_load_time": original_time,
            "cached_load_time": cached_time,
            "improvement": (original_time - cached_time) / original_time * 100,
            "symbols_tested": len(test_symbols),
        }

    def test_concurrent_analysis(
        self, stock_list: pd.DataFrame, thread_counts: List[int] = [2, 4]
    ) -> Dict[str, Any]:
        """测试并发分析性能"""
        logger.info("测试并发分析性能")

        sample_size = min(5, len(stock_list))
        sample_stocks = stock_list.head(sample_size)

        results = {}

        for thread_count in thread_counts:
            logger.info(f"测试线程数: {thread_count}")

            start_time = time.time()

            try:
                with OptimizedQuantStrategy(
                    use_distributed=False, use_cache=False
                ) as strategy:
                    # 限制分析的股票数量，避免测试时间过长
                    original_get_stock_list = strategy.get_stock_list

                    def limited_get_stock_list():
                        stocks = original_get_stock_list()
                        return stocks.head(min(10, len(stocks)))  # 限制最多分析10只股票

                    strategy.get_stock_list = limited_get_stock_list

                    # 手动设置线程数
                    strategy.max_workers = thread_count
                    results_df = strategy.run_distributed_analysis(sample_stocks)

                    execution_time = time.time() - start_time

                    results[f"{thread_count}_threads"] = {
                        "execution_time": execution_time,
                        "selected_stocks": (
                            len(results_df) if not results_df.empty else 0
                        ),
                        "throughput": sample_size / execution_time,  # 股票/秒
                    }

            except Exception as e:
                logger.error(f"并发测试失败 (线程数: {thread_count}): {e}")
                results[f"{thread_count}_threads"] = {
                    "execution_time": 0,
                    "selected_stocks": 0,
                    "throughput": 0,
                    "error": str(e),
                }

        return results

    def generate_performance_report(self) -> str:
        """生成性能测试报告"""
        report = []
        report.append("=" * 60)
        report.append("QSSS 性能测试报告")
        report.append("=" * 60)
        report.append("")

        # 对比分析
        if self.test_results["original"] and self.test_results["optimized"]:
            orig = self.test_results["original"]
            opt = self.test_results["optimized"]

            report.append("性能对比:")
            time_improvement = (
                (orig["execution_time"] - opt["execution_time"])
                / orig["execution_time"]
                * 100
            )
            report.append(f"  执行时间改进: {time_improvement:.1f}%")

            memory_improvement = (
                (orig["memory_usage"] - opt["memory_usage"])
                / abs(orig["memory_usage"] or 1)
                * 100
            )
            report.append(f"  内存使用改进: {memory_improvement:.1f}%")
            report.append(
                f"  选中股票数量: 原始={orig['selected_stocks']}, 优化={opt['selected_stocks']}"
            )

            if opt.get("hit_rate"):
                report.append(f"  缓存命中率: {opt['hit_rate']:.1%}")

            report.append("")

        # 数据加载性能
        if "data_loading" in self.test_results:
            dl = self.test_results["data_loading"]
            report.append("数据加载性能:")
            report.append(f"  原始加载时间: {dl['original_load_time']:.3f}秒")
            report.append(f"  缓存加载时间: {dl['cached_load_time']:.3f}秒")
            report.append(f"  改进幅度: {dl['improvement']:.1f}%")
            report.append("")

        # 并发性能
        if "concurrent" in self.test_results:
            report.append("并发分析性能:")
            for thread_config, result in self.test_results["concurrent"].items():
                if not result.get("error"):
                    report.append(
                        f"  {thread_config}: {result['execution_time']:.2f}秒, "
                        f"吞吐量: {result['throughput']:.1f}股票/秒"
                    )
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def save_results(self, filename: str = "performance_test_results.json"):
        """保存测试结果"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"测试结果已保存到: {filename}")

    def run_full_test(self, sample_sizes: List[int] = [3]):
        """运行完整性能测试"""
        logger.info("开始完整性能测试")

        try:
            # 获取股票列表
            logger.info("获取股票列表...")
            stock_list = data_manager.get_stock_list()
            if stock_list.empty:
                logger.error("无法获取股票列表")
                return

            logger.info(f"获取到 {len(stock_list)} 只股票")

            # 测试数据加载性能
            logger.info("=" * 40)
            self.test_results["data_loading"] = self.test_data_loading_performance()

            # 测试不同样本大小
            for sample_size in sample_sizes:
                if sample_size > len(stock_list):
                    continue

                logger.info(f"\n{'='*40}")
                logger.info(f"测试样本大小: {sample_size}")

                # 测试原始策略
                logger.info("测试原始策略...")
                self.test_results["original"] = self.test_original_strategy(
                    stock_list, sample_size
                )

                # 测试优化策略
                logger.info("测试优化策略...")
                self.test_results["optimized"] = self.test_optimized_strategy(
                    stock_list, sample_size
                )

                # 如果样本足够大，测试并发性能
                if sample_size >= 3:
                    logger.info("测试并发性能...")
                    self.test_results["concurrent"] = self.test_concurrent_analysis(
                        stock_list
                    )

            # 生成报告
            report = self.generate_performance_report()
            print("\n" + report)

            # 保存结果
            self.save_results()

            logger.info("性能测试完成")

        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            import traceback

            traceback.print_exc()


def main():
    """主函数"""
    logger.add("performance_test.log", rotation="10 MB")

    tester = PerformanceTester()
    tester.run_full_test()


if __name__ == "__main__":
    main()
