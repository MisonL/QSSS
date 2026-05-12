"""pytdx数据源适配器"""

from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
from loguru import logger

try:
    from pytdx.hq import TdxHq_API
    from pytdx.util.best_ip import select_best_ip

    PYTDX_AVAILABLE = True
except ImportError:
    PYTDX_AVAILABLE = False
    logger.warning("pytdx未安装，请运行: pip install pytdx")


class PytdxAdapter:
    """pytdx数据源适配器"""

    def __init__(self) -> None:
        self.api: Any = None
        self.best_ip: Any = None
        self._connect()

    def _connect(self) -> None:
        """连接到最佳服务器"""
        if not PYTDX_AVAILABLE:
            raise ImportError("pytdx库未安装")

        try:
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

        except Exception as e:
            logger.error(f"pytdx连接失败: {e}")
            self._connect_backup()

    def _connect_backup(self) -> None:
        """连接备用服务器"""
        backup_servers = [
            {"ip": "119.147.212.81", "port": 7709},
            {"ip": "119.147.212.83", "port": 7709},
            {"ip": "218.80.248.229", "port": 7709},
            {"ip": "60.191.117.167", "port": 7709},
        ]

        for server in backup_servers:
            try:
                self.api = TdxHq_API()
                if self.api.connect(server["ip"], server["port"]):
                    self.best_ip = server
                    logger.info(f"已连接到备用服务器: {server['ip']}:{server['port']}")
                    return
            except Exception as e:
                logger.warning(
                    f"备用服务器连接失败 {server['ip']}:{server['port']}: {e}"
                )
                continue

        raise ConnectionError("所有pytdx服务器连接失败")

    def _get_market_code(self, symbol: str) -> int:
        """根据股票代码获取市场代码"""
        prefix = symbol[:2]
        if prefix.startswith(("60", "68")):
            return 1  # 上交所
        elif prefix.startswith(("00", "30")):
            return 0  # 深交所
        elif prefix.startswith(("83", "87", "43")):
            return 2  # 北交所
        else:
            return 0  # 默认深交所

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        try:
            stocks = []

            # 获取上交所股票
            sh_count = self.api.get_security_count(1)
            for i in range(0, sh_count, 1000):
                data = self.api.get_security_list(1, i)
                for item in data:
                    if item["code"].startswith(("60", "68")):
                        stocks.append(
                            {
                                "symbol": item["code"],
                                "name": item["name"].strip(),
                                "market": "上交所-"
                                + (
                                    "主板"
                                    if item["code"].startswith("60")
                                    else "科创板"
                                ),
                            }
                        )

            # 获取深交所股票
            sz_count = self.api.get_security_count(0)
            for i in range(0, sz_count, 1000):
                data = self.api.get_security_list(0, i)
                for item in data:
                    if item["code"].startswith(("00", "30")):
                        market_type = (
                            "主板" if item["code"].startswith("00") else "创业板"
                        )
                        stocks.append(
                            {
                                "symbol": item["code"],
                                "name": item["name"].strip(),
                                "market": f"深交所-{market_type}",
                            }
                        )

            # 过滤ST和退市股票
            filtered_stocks = [
                stock
                for stock in stocks
                if not any(keyword in stock["name"] for keyword in ["ST", "退", "退市"])
            ]

            return pd.DataFrame(filtered_stocks)

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取股票日线数据"""
        try:
            market = self._get_market_code(symbol)

            # 验证日期参数
            if start_date and end_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y%m%d")
                    end_dt = datetime.strptime(end_date, "%Y%m%d")
                    days = (end_dt - start_dt).days
                    if days <= 0:
                        logger.error(
                            f"日期范围无效: start_date={start_date}, end_date={end_date}"
                        )
                        return pd.DataFrame()
                    count = min(days, 800)
                except ValueError as e:
                    logger.error(
                        f"日期格式错误: {e}, start_date={start_date}, end_date={end_date}"
                    )
                    return pd.DataFrame()
            else:
                count = 800

            # 获取K线数据
            data = []
            for i in range(0, count, 800):
                bars = self.api.get_security_bars(
                    9, market, symbol, i, min(800, count - i)
                )
                data.extend(bars)

            if not data:
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame(data)

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

            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            # 计算其他必要指标
            df["amplitude"] = (
                (df["high"] - df["low"]) / df["close"].shift(1) * 100
            ).round(2)
            df["pct_chg"] = (
                (df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100
            ).round(2)
            df["change"] = (df["close"] - df["close"].shift(1)).round(2)
            df["turn"] = 0.0  # pytdx不直接提供换手率

            # 按日期排序
            df = df.sort_values("date").reset_index(drop=True)
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"获取{symbol}日线数据失败: {e}")
            return pd.DataFrame()

    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情数据"""
        try:
            result = []
            for symbol in symbols:
                market = self._get_market_code(symbol)
                data = self.api.get_security_quotes([(market, symbol)])

                if data:
                    item = data[0]
                    result.append(
                        {
                            "symbol": symbol,
                            "name": item["name"],
                            "price": item["price"],
                            "last_close": item["last_close"],
                            "open": item["open"],
                            "high": item["high"],
                            "low": item["low"],
                            "volume": item["volume"],
                            "amount": item["amount"],
                            "time": item["time"],
                        }
                    )

            return pd.DataFrame(result)

        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return pd.DataFrame()

    def __del__(self) -> None:
        """析构函数，关闭连接"""
        if self.api:
            self.api.disconnect()
