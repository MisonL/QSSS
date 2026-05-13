"""数据源适配器模块"""

import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ..config.settings import settings
from ..utils.decorators import retry_on_exception

try:
    from pytdx.hq import TdxHq_API
    from pytdx.util.best_ip import select_best_ip

    PYTDX_AVAILABLE = True
except ImportError:
    PYTDX_AVAILABLE = False
    logger.warning("pytdx未安装，请运行: pip install pytdx")

# akshare已移除，仅使用pytdx
AKSHARE_AVAILABLE = False


class DataAdapter(ABC):
    """数据源适配器基类"""

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        raise NotImplementedError

    @abstractmethod
    def get_daily_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取日线数据"""
        raise NotImplementedError

    @abstractmethod
    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时数据"""
        raise NotImplementedError


class PytdxAdapter(DataAdapter):
    """高性能pytdx数据源适配器"""

    def __init__(self) -> None:
        self.api: Any = None
        self.best_ip: Any = None
        self._connected: bool = False
        # 连接池管理
        self._connection_pool: List[Any] = []
        # 可通过 Settings 调整 pytdx 连接池与线程池参数
        self._pool_size: int = settings.pytdx_pool_size
        self._pool_lock: threading.Lock = threading.Lock()
        # 缓存机制
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._cache_lock: threading.Lock = threading.Lock()
        # 默认为 300 秒，可通过 Settings.pytdx_cache_ttl 调整
        self._cache_ttl: float = settings.pytdx_cache_ttl
        # 批量处理配置
        self._batch_size: int = 50
        self._max_workers: int = settings.pytdx_max_workers

    def _connect(self) -> None:
        """连接到最佳服务器"""
        try:
            if not PYTDX_AVAILABLE:
                raise ImportError("pytdx库未安装")

            # 临时重定向stdout，屏蔽select_best_ip的输出
            import io
            from contextlib import redirect_stdout

            # 捕获并忽略select_best_ip的输出
            f = io.StringIO()
            with redirect_stdout(f):
                self.best_ip = select_best_ip("stock")

            self.api = TdxHq_API()

            if not self.api.connect(self.best_ip["ip"], self.best_ip["port"]):
                raise ConnectionError("无法连接到pytdx服务器")

            logger.info(
                f"已连接到pytdx服务器: {self.best_ip['ip']}:{self.best_ip['port']}"
            )
            self._connected = True

        except Exception as e:
            logger.error(f"pytdx连接失败: {e}")
            self._connect_backup()

    def _connect_backup(self) -> None:
        """连接备用服务器"""
        backup_servers = [
            {"ip": "119.147.212.81", "port": 7709},
            {"ip": "119.147.212.83", "port": 7709},
            {"ip": "218.108.98.244", "port": 7709},
            {"ip": "218.108.47.69", "port": 7709},
            {"ip": "115.238.90.165", "port": 7709},
            {"ip": "123.125.108.23", "port": 7709},
            {"ip": "123.125.108.24", "port": 7709},
            {"ip": "60.191.117.167", "port": 7709},
            {"ip": "180.153.39.51", "port": 7709},
            {"ip": "218.80.248.229", "port": 7709},
        ]

        for server in backup_servers:
            try:
                self.api = TdxHq_API()
                if self.api.connect(server["ip"], server["port"]):
                    self.best_ip = server
                    logger.info(
                        f"已连接到备用pytdx服务器: {server['ip']}:{server['port']}"
                    )
                    self._connected = True
                    return
            except Exception as e:
                logger.warning(
                    f"备用服务器连接失败 {server['ip']}:{server['port']}: {e}"
                )
                continue

        raise ConnectionError("所有pytdx服务器连接失败")

    def ensure_connected(self) -> None:
        """确保连接已建立"""
        if not self._connected:
            try:
                self._connect()
                self._init_connection_pool()
                self._connected = True
            except Exception as e:
                logger.error(f"pytdx连接失败: {e}")
                self._connected = False
                raise

    def _init_connection_pool(self) -> None:
        """初始化连接池"""
        with self._pool_lock:
            for i in range(self._pool_size):
                try:
                    api = TdxHq_API()
                    if api.connect(self.best_ip["ip"], self.best_ip["port"]):
                        self._connection_pool.append(api)
                        logger.info(f"连接池初始化成功: {i+1}/{self._pool_size}")
                    else:
                        logger.warning(f"连接池初始化失败: {i+1}")
                except Exception as e:
                    logger.error(f"连接池初始化异常: {e}")

    def _get_connection(self) -> Any:
        """从连接池获取连接"""
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop(0)
        # 如果连接池为空，创建新连接
        api: Any = TdxHq_API()
        if api.connect(self.best_ip["ip"], self.best_ip["port"]):
            return api
        return None

    def _return_connection(self, api: Any) -> None:
        """归还连接到连接池"""
        if api:
            with self._pool_lock:
                if len(self._connection_pool) < self._pool_size:
                    self._connection_pool.append(api)
                else:
                    # 连接池已满，关闭连接
                    try:
                        api.disconnect()
                    except Exception as e:
                        logger.warning(f"关闭溢出 pytdx 连接失败: {e}")

    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        return f"{symbol}_{start_date}_{end_date}"

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        with self._cache_lock:
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    logger.debug(f"缓存命中: {cache_key}")
                    return data.copy()
                else:
                    # 缓存过期，删除
                    del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """设置缓存"""
        with self._cache_lock:
            self._cache[cache_key] = (data.copy(), time.time())
            logger.debug(f"缓存设置: {cache_key}")

    def _clear_expired_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        with self._cache_lock:
            expired_keys = [
                key
                for key, (data, timestamp) in self._cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.debug(f"清理过期缓存: {len(expired_keys)}个")

    def _get_market_code(self, symbol: str) -> int:
        """根据股票代码获取市场代码 - 增强版本"""
        symbol = str(symbol).strip()
        if not symbol:
            logger.error("股票代码为空")
            return 0

        # 更精确的市场代码判断
        if symbol.startswith("60") or symbol.startswith("68"):
            return 1  # 上交所
        elif symbol.startswith("00") or symbol.startswith("30"):
            return 0  # 深交所
        elif (
            symbol.startswith("83")
            or symbol.startswith("87")
            or symbol.startswith("43")
        ):
            return 2  # 北交所
        else:
            # 默认规则：根据代码长度和特征判断
            if len(symbol) == 6:
                if symbol[0] in ["5", "6"]:
                    return 1  # 上交所
                else:
                    return 0  # 默认深交所
            logger.warning(f"无法确定股票 {symbol} 的市场代码，默认使用深交所")
            return 0

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        self.ensure_connected()
        try:
            stocks = []

            # 获取上交所股票
            try:
                sh_count = self.api.get_security_count(1)
                if sh_count is not None:
                    for i in range(0, sh_count, 1000):
                        data = self.api.get_security_list(1, i)
                        if data is None:  # 检查None返回值
                            logger.warning(f"获取上交所股票列表返回None，跳过位置{i}")
                            continue
                        for item in data:
                            if item["code"].startswith(("60", "68")):
                                market_type = (
                                    "主板"
                                    if item["code"].startswith("60")
                                    else "科创板"
                                )
                                stocks.append(
                                    {
                                        "symbol": item["code"],
                                        "name": item["name"].strip(),
                                        "market": f"上交所-{market_type}",
                                    }
                                )
                else:
                    logger.warning("获取上交所股票总数返回None")
            except Exception as e:
                logger.error(f"获取上交所股票列表失败: {e}")

            # 获取深交所股票
            try:
                sz_count = self.api.get_security_count(0)
                if sz_count is not None:
                    for i in range(0, sz_count, 1000):
                        data = self.api.get_security_list(0, i)
                        if data is None:  # 检查None返回值
                            logger.warning(f"获取深交所股票列表返回None，跳过位置{i}")
                            continue
                        for item in data:
                            if item["code"].startswith(("00", "30")):
                                market_type = (
                                    "主板"
                                    if item["code"].startswith("00")
                                    else "创业板"
                                )
                                stocks.append(
                                    {
                                        "symbol": item["code"],
                                        "name": item["name"].strip(),
                                        "market": f"深交所-{market_type}",
                                    }
                                )
                else:
                    logger.warning("获取深交所股票总数返回None")
            except Exception as e:
                logger.error(f"获取深交所股票列表失败: {e}")

            # 过滤ST和退市股票
            filtered_stocks = [
                stock
                for stock in stocks
                if not any(keyword in stock["name"] for keyword in ["ST", "退", "退市"])
            ]

            df = pd.DataFrame(filtered_stocks)
            logger.info(f"成功获取股票列表，共 {len(df)} 只股票")
            return df

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    @retry_on_exception(retries=3, delay=1.0, backoff=1.5)
    def get_daily_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取股票日线数据"""
        # 确保连接
        try:
            self.ensure_connected()
        except Exception as e:
            logger.error(f"连接失败，无法获取{symbol}日线数据: {e}")
            return pd.DataFrame()

        try:
            market = self._get_market_code(symbol)

            # 验证日期参数
            if not start_date or not end_date:
                logger.error(
                    f"日期参数不能为空: start_date={start_date}, end_date={end_date}"
                )
                return pd.DataFrame()

            # 计算需要获取的K线数量
            try:
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                days = (end_dt - start_dt).days
                if days <= 0:
                    logger.error(
                        f"日期范围无效: start_date={start_date}, end_date={end_date}"
                    )
                    return pd.DataFrame()
            except ValueError as e:
                logger.error(
                    f"日期格式错误: {e}, start_date={start_date}, end_date={end_date}"
                )
                return pd.DataFrame()

            # 获取K线数据 - 修复pytdx接口调用问题
            data = []

            # 尝试不同的数据获取策略 - 线程安全版本
            strategies = [
                # 策略1: 获取最近365天的数据 (最高优先级)
                {
                    "category": 9,
                    "market": market,
                    "symbol": symbol,
                    "start": 0,
                    "count": 365,
                },
                # 策略2: 获取最近180天的数据 (减少数据量)
                {
                    "category": 9,
                    "market": market,
                    "symbol": symbol,
                    "start": 0,
                    "count": 180,
                },
                # 策略3: 获取最近90天的数据 (最小数据量)
                {
                    "category": 9,
                    "market": market,
                    "symbol": symbol,
                    "start": 0,
                    "count": 90,
                },
                # 策略4: 使用不同的category参数 (备选方案)
                {
                    "category": 0,
                    "market": market,
                    "symbol": symbol,
                    "start": 0,
                    "count": 365,
                },
            ]

            # 线程安全的超时控制 - 使用concurrent.futures替代signal
            import concurrent.futures

            bars = None
            for i, strategy in enumerate(strategies):
                try:
                    logger.debug(f"尝试策略{i+1}: {strategy}")

                    # 使用线程安全的超时机制
                    def get_bars_with_timeout() -> Any:
                        return self.api.get_security_bars(
                            strategy["category"],
                            strategy["market"],
                            strategy["symbol"],
                            strategy["start"],
                            strategy["count"],
                        )

                    # 使用线程池执行，设置超时
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=1
                    ) as executor:
                        future = executor.submit(get_bars_with_timeout)
                        try:
                            bars = future.result(timeout=settings.pytdx_kline_timeout)
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"策略{i+1}超时: 8秒内未完成")
                            future.cancel()
                            continue
                        except Exception as e:
                            logger.warning(f"策略{i+1}执行异常: {e}")
                            future.cancel()
                            continue

                    if bars is not None and len(bars) > 0:
                        logger.info(
                            f"策略{i+1}成功: 获取{symbol} K线数据，共{len(bars)}条"
                        )
                        break
                    else:
                        logger.warning(f"策略{i+1}失败: 返回None或空数据")

                except Exception as e:
                    logger.warning(f"策略{i+1}设置异常: {e}")
                    continue

            # 添加详细的日志信息
            logger.debug(
                f"最终获取{symbol} K线数据，市场代码: {market}, 返回类型: {type(bars)}, "
                f"是否为None: {bars is None}"
            )

            if bars is not None and len(bars) > 0:
                data.extend(bars)
                logger.info(f"成功获取 {symbol} 的K线数据，共 {len(bars)} 条记录")
            else:
                logger.warning(f"所有策略均未能获取{symbol}的K线数据")
                # 尝试备用服务器 - 线程安全版本
                try:
                    logger.info(f"尝试备用服务器获取{symbol}数据")

                    # 使用线程安全的超时机制
                    def get_backup_bars() -> Any:
                        self._connect_backup()
                        return self.api.get_security_bars(9, market, symbol, 0, 180)

                    # 使用线程池执行，设置超时
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=1
                    ) as executor:
                        future = executor.submit(get_backup_bars)
                        try:
                            bars = future.result(timeout=settings.pytdx_backup_timeout)

                            if bars is not None and len(bars) > 0:
                                data.extend(bars)
                                logger.info(f"备用服务器成功获取 {symbol} 的K线数据")
                            else:
                                logger.warning(f"备用服务器获取{symbol}数据也为空")

                        except concurrent.futures.TimeoutError:
                            logger.error(f"备用服务器获取{symbol}数据超时")
                            future.cancel()
                        except Exception as backup_e:
                            logger.error(f"备用服务器也失败: {backup_e}")
                            future.cancel()

                except Exception as e:
                    logger.error(f"备用服务器设置失败: {e}")

            if not data:
                logger.warning(f"未能获取到{symbol}的K线数据")
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame(data)

            # 检查必要字段是否存在 - 处理pytdx返回的数据结构
            required_fields = [
                "datetime",
                "open",
                "close",
                "high",
                "low",
                "vol",
                "amount",
            ]
            missing_fields = [
                field for field in required_fields if field not in df.columns
            ]
            if missing_fields:
                logger.warning(f"股票 {symbol} 数据字段不完整，缺少: {missing_fields}")
                logger.debug(f"实际字段: {list(df.columns)}")
                # 尝试处理可能的字段名差异
                column_mapping = {}
                for field in required_fields:
                    if field in df.columns:
                        column_mapping[field] = field
                    elif field == "vol" and "volume" in df.columns:
                        column_mapping["volume"] = "vol"
                    elif field == "datetime" and "time" in df.columns:
                        column_mapping["time"] = "datetime"

                if column_mapping:
                    df = df.rename(columns=column_mapping)
                    logger.debug(f"字段映射: {column_mapping}")

                # 再次检查
                missing_fields = [
                    field for field in required_fields if field not in df.columns
                ]
                if missing_fields:
                    logger.error(f"股票 {symbol} 数据字段仍然不完整，无法处理")
                    return pd.DataFrame()

            # 重命名列以兼容现有代码
            df = df.rename(
                columns={
                    "datetime": "date",
                    "open": "open",
                    "close": "close",
                    "high": "high",
                    "low": "low",
                    "vol": "volume",
                    "amount": "amount",
                }
            )

            # 数据清洗和验证
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])  # 移除无效日期

            if df.empty:
                return pd.DataFrame()

            # 按日期排序
            df = df.sort_values("date").reset_index(drop=True)

            # 转换日期格式
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            # 计算其他必要指标
            df["amplitude"] = (
                (df["high"] - df["low"]) / df["close"].shift(1) * 100
            ).round(2)
            df["pct_chg"] = (
                (df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100
            ).round(2)
            df["change"] = (df["close"] - df["close"].shift(1)).round(2)
            df["turn"] = 0.0  # 暂时设为0，后续可优化

            # 按日期排序并清理数据
            df = df.sort_values("date").reset_index(drop=True)
            df = df.dropna()

            logger.info(f"成功获取 {symbol} 的日线数据，共 {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"获取{symbol}日线数据失败: {e}")
            import traceback

            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情数据"""
        if not symbols:
            return pd.DataFrame()
        result = []
        for attempt in range(2):
            self.ensure_connected()
            result = []
            fetched_at = datetime.now().isoformat(timespec="seconds")
            for symbol in symbols:
                market = self._get_market_code(symbol)
                data = self.api.get_security_quotes([(market, symbol)])

                if data:
                    item = data[0]
                    quote_time = item.get("servertime") or item.get("time") or ""
                    result.append(
                        {
                            "symbol": symbol,
                            "name": item.get("name", ""),
                            "price": item["price"],
                            "last_close": item["last_close"],
                            "open": item["open"],
                            "high": item["high"],
                            "low": item["low"],
                            "volume": item.get("vol", item.get("volume", 0)),
                            "amount": item["amount"],
                            "quote_time": quote_time,
                            "time": quote_time,
                            "source": "pytdx",
                            "fetched_at": fetched_at,
                        }
                    )

            if result or attempt == 1:
                return pd.DataFrame(result)

            logger.warning("实时行情返回为空，重新连接 pytdx 后重试一次")
            self._connected = False
            if self.api:
                try:
                    self.api.disconnect()
                except Exception as e:
                    logger.warning(f"断开旧 pytdx 连接失败: {e}")

        return pd.DataFrame(result)

    def __del__(self) -> None:
        """析构函数，关闭连接"""
        if self.api:
            self.api.disconnect()


# AkshareAdapter类已完全移除，仅使用pytdx数据源
