"""数据管理模块"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from ..config.settings import settings
from .adapters import DataAdapter, PytdxAdapter
from .sources import akshare_adapter as _akshare_module
from .sources.ths_local_adapter import ThsLocalBoardAdapter

# 可选的 Tushare 适配器支持：只有在安装了 tushare 且配置了 Token 时才会生效
try:  # pragma: no cover - 缺少 tushare 时的分支在测试中通常不会覆盖
    from .sources.tushare_adapter import TushareAdapter as _TushareAdapter

    _HAS_TUSHARE = True
except ImportError:
    _TushareAdapter = None  # type: ignore[assignment, misc]
    _HAS_TUSHARE = False

# 可选的 Baostock 适配器支持：仅在安装并登录成功时生效
try:  # pragma: no cover
    from .sources.baostock_adapter import BaostockAdapter as _BaostockAdapter

    _HAS_BAOSTOCK = True
except ImportError:
    _BaostockAdapter = None  # type: ignore[assignment, misc]
    _HAS_BAOSTOCK = False


class DataManager:
    """统一数据管理器

    默认注册 pytdx 适配器。Tushare / Baostock 仅在被配置为主源或备用源时初始化；
    也可以通过 register_adapter 在此基础上扩展其他数据源（如新的行情接口、
    回测数据源等），核心策略层无需改动。
    """

    def __init__(self) -> None:
        # 主数据源/备用数据源来自全局配置，便于通过环境变量切换。
        self.primary_source = settings.primary_data_source
        self.backup_source = settings.backup_data_source or None
        configured_sources = {
            source for source in [self.primary_source, self.backup_source] if source
        }

        # 已注册的数据源适配器；键为数据源名称
        self.adapters: Dict[str, Any] = {
            "pytdx": PytdxAdapter(),
        }

        if _akshare_module.AKSHARE_AVAILABLE:
            try:
                self.adapters["akshare"] = _akshare_module.AkshareAdapter()
                logger.info("已注册 AkshareAdapter 作为板块和资金流数据源 'akshare'")
            except Exception as e:
                logger.warning(f"初始化 AkshareAdapter 失败，将忽略该数据源: {e}")

        if settings.ths_conception_path or settings.ths_industry_path:
            self.adapters["ths_local_cache"] = ThsLocalBoardAdapter(
                conception_path=settings.ths_conception_path or None,
                industry_path=settings.ths_industry_path or None,
            )

        # Tushare / Baostock 会触发凭据或登录流程，只在配置引用时初始化。
        if (
            "tushare" in configured_sources
            and _TushareAdapter is not None
            and _HAS_TUSHARE
        ):
            try:
                self.adapters["tushare"] = _TushareAdapter()
                logger.info("已注册 TushareAdapter 作为可选数据源 'tushare'")
            except Exception as e:
                logger.warning(f"初始化 TushareAdapter 失败，将忽略该数据源: {e}")

        if (
            "baostock" in configured_sources
            and _BaostockAdapter is not None
            and _HAS_BAOSTOCK
        ):
            try:
                self.adapters["baostock"] = _BaostockAdapter()
                logger.info("已注册 BaostockAdapter 作为可选数据源 'baostock'")
            except Exception as e:
                logger.warning(f"初始化 BaostockAdapter 失败，将忽略该数据源: {e}")

        # 数据源健康监控结构
        self._source_health: Dict[str, Any] = {}
        self.health_failure_threshold: int = settings.datasource_failure_threshold
        self.health_cooldown: float = settings.datasource_cooldown_seconds

    def register_adapter(self, name: str, adapter: DataAdapter) -> None:
        """注册新的数据源适配器。

        典型用法：在将来新增 TushareAdapter、XXAdapter 等时，在程序初始化
        阶段调用一次即可：

            data_manager.register_adapter("tushare", TushareAdapter(...))
        """
        self.adapters[name] = adapter

    def get_available_sources(self) -> List[str]:
        """返回当前已注册且可用的数据源名称列表。"""
        return sorted(self.adapters.keys())

    # ------------------------------------------------------------------
    # 数据源健康状态相关工具方法
    # ------------------------------------------------------------------

    def _get_or_create_health(self, source_name: str) -> Dict[str, Any]:
        info = self._source_health.get(source_name)
        if info is None:
            info = {
                "consecutive_failures": 0,
                "last_success_ts": 0.0,
                "last_error_ts": 0.0,
                "degraded_until": 0.0,
            }
            self._source_health[source_name] = info
        return info

    def _mark_success(self, source_name: str) -> None:
        info = self._get_or_create_health(source_name)
        degraded_before = float(info.get("degraded_until") or 0.0)
        info["consecutive_failures"] = 0
        info["last_success_ts"] = time.time()
        info["degraded_until"] = 0.0

        # 如该数据源之前处于降级状态，现在视为恢复正常
        if degraded_before > 0.0 and degraded_before > time.time():
            logger.info(
                f"数据源 {source_name} 已恢复正常 (之前降级截止时间戳={degraded_before:.0f})"
            )

    def _mark_failure(self, source_name: str) -> None:
        info = self._get_or_create_health(source_name)
        now = time.time()
        info["consecutive_failures"] += 1
        info["last_error_ts"] = now

        # 当连续失败次数达到阈值时，进入降级窗口
        if (
            self.health_failure_threshold > 0
            and info["consecutive_failures"] >= self.health_failure_threshold
            and self.health_cooldown > 0
        ):
            if not self._is_degraded(source_name):
                degraded_until = now + self.health_cooldown
                info["degraded_until"] = degraded_until
                logger.warning(
                    f"数据源 {source_name} 连续失败 {info['consecutive_failures']} 次，"
                    f"进入降级状态，冷却 {self.health_cooldown}s "
                    f"(degraded_until={degraded_until:.0f})"
                )

    def _is_degraded(self, source_name: str) -> bool:
        info = self._source_health.get(source_name)
        if not info:
            return False
        degraded_until = float(info.get("degraded_until") or 0.0)
        return degraded_until > 0.0 and degraded_until > time.time()

    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """返回当前各数据源的健康状态快照，便于调试和监控。"""
        return {name: dict(info) for name, info in self._source_health.items()}

    def get_adapter(self, source: Optional[str] = None) -> Optional[DataAdapter]:
        """获取数据源适配器

        优先返回显式指定的数据源；否则使用主数据源。如果主数据源未注册，
        且配置了备用数据源，则回退到备用数据源。
        """
        source_name = source or self.primary_source
        adapter = self.adapters.get(source_name)
        if adapter is None and self.backup_source:
            adapter = self.adapters.get(self.backup_source)
        return adapter

    def _get_stock_list_from_adapter(self, source_name: str) -> pd.DataFrame:
        """从指定数据源获取股票列表，并统一异常与空结果处理。

        注意：本方法会更新内部健康状态（成功/失败统计和降级时间）。
        """
        adapter = self.adapters.get(source_name)
        if not adapter:
            logger.error(f"不支持的数据源: {source_name}")
            self._mark_failure(source_name)
            return pd.DataFrame()

        try:
            stocks = adapter.get_stock_list()
            if stocks is None:
                logger.warning(f"{source_name} 获取股票列表返回None")
                self._mark_failure(source_name)
                return pd.DataFrame()
            if getattr(stocks, "empty", False):
                logger.warning(f"{source_name} 获取股票列表为空")
                self._mark_failure(source_name)
                return pd.DataFrame()

            self._mark_success(source_name)
            return stocks
        except Exception as e:
            logger.error(f"{source_name} 获取股票列表失败: {e}")
            self._mark_failure(source_name)
            return pd.DataFrame()

    def _get_daily_from_adapter(
        self,
        source_name: str,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """从指定数据源获取日线数据，带默认日期与统一日志。

        注意：本方法会更新内部健康状态（成功/失败统计和降级时间）。
        """
        adapter = self.adapters.get(source_name)
        if not adapter:
            logger.error(f"不支持的数据源: {source_name}")
            self._mark_failure(source_name)
            return pd.DataFrame()

        # 设置默认日期范围
        if not start_date or not end_date:
            from datetime import datetime, timedelta

            end_date_val = datetime.now().strftime("%Y%m%d")
            start_date_val = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        else:
            start_date_val = start_date
            end_date_val = end_date

        try:
            data = adapter.get_daily_data(symbol, start_date_val, end_date_val)
            # 添加更详细的日志信息
            logger.debug(
                f"适配器 {source_name} 返回数据类型: {type(data)}, 是否为None: {data is None}, "
                f"是否为空: {data.empty if hasattr(data, 'empty') else 'N/A'}"
            )
            if data is None:
                logger.warning(f"{source_name} 获取{symbol}数据返回None")
                self._mark_failure(source_name)
                return pd.DataFrame()
            if getattr(data, "empty", False):
                logger.warning(f"{source_name} 获取{symbol}数据为空")
                self._mark_failure(source_name)
                return pd.DataFrame()

            self._mark_success(source_name)
            return data
        except Exception as e:
            logger.error(f"{source_name} 获取{symbol}日线数据失败: {e}")
            self._mark_failure(source_name)
            return pd.DataFrame()

    def get_stock_list(self, source: Optional[str] = None) -> pd.DataFrame:
        """获取股票列表。

        - 如显式指定 source，则只使用该数据源；
        - 否则：优先使用主数据源，如为空或失败且配置了不同的备用数据源，则回退到备用数据源。

        在未显式指定 source 的情况下，会优先跳过当前处于降级窗口内的主数据源，
        直接尝试备用数据源，以减少整体延迟。
        """
        # 显式指定数据源时，不做自动回退，保持调用方语义清晰
        if source:
            return self._get_stock_list_from_adapter(source)

        primary = self.primary_source
        backup = (
            self.backup_source
            if self.backup_source and self.backup_source != primary
            else None
        )

        first_source = primary
        second_source = backup

        # 如果主数据源处于降级状态，优先尝试备用数据源
        if primary and backup and self._is_degraded(primary):
            logger.info(
                f"主数据源 {primary} 当前处于降级窗口，优先尝试备用数据源 {backup} 获取股票列表"
            )
            first_source, second_source = backup, primary

        stocks = self._get_stock_list_from_adapter(first_source)
        if not stocks.empty or not second_source:
            return stocks

        logger.info(
            f"数据源 {first_source} 股票列表为空或失败，尝试备用数据源 {second_source}"
        )
        fallback_stocks = self._get_stock_list_from_adapter(second_source)
        return fallback_stocks

    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取日线数据。

        - 显式指定 source 时，仅使用对应数据源；
        - 未指定时，优先主数据源，如返回空/失败且配置了不同的备用数据源，则自动回退。

        在未显式指定 source 的情况下，会优先跳过当前处于降级窗口内的主数据源，
        直接尝试备用数据源，以减少整体延迟。
        """
        # 显式指定数据源：不做自动回退
        if source:
            return self._get_daily_from_adapter(source, symbol, start_date, end_date)

        primary = self.primary_source
        backup = (
            self.backup_source
            if self.backup_source and self.backup_source != primary
            else None
        )

        first_source = primary
        second_source = backup

        if primary and backup and self._is_degraded(primary):
            logger.info(
                f"主数据源 {primary} 当前处于降级窗口，优先尝试备用数据源 {backup} 获取 {symbol} 日线数据"
            )
            first_source, second_source = backup, primary

        # 默认走主/备数据源链路
        data = self._get_daily_from_adapter(first_source, symbol, start_date, end_date)
        if not data.empty or not second_source:
            return data

        logger.info(
            f"数据源 {first_source} 获取 {symbol} 日线数据为空或失败，"
            f"尝试备用数据源 {second_source}"
        )
        backup_data = self._get_daily_from_adapter(
            second_source, symbol, start_date, end_date
        )
        return backup_data

    def get_realtime_data(
        self, symbols: List[str], source: Optional[str] = None
    ) -> pd.DataFrame:
        """获取实时数据。

        实时数据通常对时效性要求高，这里仍然只使用单一数据源；
        如需更复杂的聚合策略，可在后续扩展。
        """
        adapter = self.get_adapter(source)
        if not adapter:
            logger.error(f"不支持的数据源: {source}")
            return pd.DataFrame()

        try:
            return adapter.get_realtime_data(symbols)
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return pd.DataFrame()

    def get_boards(
        self, board_type: str = "concept", source: str = "akshare"
    ) -> pd.DataFrame:
        """通过指定板块数据源获取板块列表。"""
        adapter = self.adapters.get(source)
        if adapter is None or not hasattr(adapter, "get_boards"):
            raise ValueError(f"数据源 {source} 不支持板块列表")
        return adapter.get_boards(board_type=board_type)

    def get_board_flows(
        self, board_type: str = "concept", source: str = "akshare"
    ) -> pd.DataFrame:
        """通过指定板块数据源获取板块资金流。"""
        adapter = self.adapters.get(source)
        if adapter is None or not hasattr(adapter, "get_board_flows"):
            raise ValueError(f"数据源 {source} 不支持板块资金流")
        return adapter.get_board_flows(board_type=board_type)

    def get_board_members(
        self,
        board: str,
        board_type: str = "concept",
        source: str = "ths_local_cache",
    ) -> pd.DataFrame:
        """通过指定板块成分数据源获取板块成分。"""
        adapter = self.adapters.get(source)
        if adapter is None or not hasattr(adapter, "get_board_members"):
            raise ValueError(f"数据源 {source} 不支持板块成分")
        return adapter.get_board_members(board, board_type=board_type)


# 全局数据管理器实例
data_manager = DataManager()
