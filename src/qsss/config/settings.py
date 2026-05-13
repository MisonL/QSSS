"""配置管理模块"""

from typing import Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置类"""

    model_config = ConfigDict(env_file=".env", env_prefix="QSSS_")

    # 数据源配置 - 支持 pytdx / tushare / baostock
    # 通过环境变量 QSSS_PRIMARY_DATA_SOURCE / QSSS_BACKUP_DATA_SOURCE 覆盖
    primary_data_source: str = Field(default="pytdx", description="主要数据源")
    backup_data_source: str = Field(default="", description="备用数据源")

    # 系统资源限制
    min_workers: int = Field(default=4, description="最小线程数")
    max_workers: int = Field(default=10, description="最大线程数")
    cpu_threshold: float = Field(default=75.0, description="CPU使用率阈值")
    memory_threshold: float = Field(default=85.0, description="内存使用率阈值")

    # 数据缓存配置
    cache_enabled: bool = Field(default=True, description="是否启用缓存")
    cache_dir: str = Field(default="data/cache", description="缓存目录")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")

    # 机器学习参数
    ml_test_size: float = Field(default=0.2, description="ML测试集占比")
    ml_random_state: int = Field(default=42, description="ML随机种子")

    # 选股参数
    min_prediction_threshold: float = Field(default=0.6, description="最小预测概率阈值")
    min_momentum_score: float = Field(default=-0.1, description="最小动量得分")
    rsi_range: tuple = Field(default=(30, 75), description="RSI范围")
    max_volatility: float = Field(default=0.6, description="最大波动率")
    min_volume: int = Field(default=50000, description="最小成交量")
    min_price: float = Field(default=3.0, description="最小股价")

    # 数据要求
    min_data_days: int = Field(default=120, description="最小数据天数")

    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default="logs/qsss.log", description="日志文件路径")

    # 重试配置
    retry_count: int = Field(default=3, description="重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟(秒)")

    # 数据源健康监控配置
    datasource_failure_threshold: int = Field(
        default=3,
        description="单个数据源连续失败多少次后进入降级状态",
    )
    datasource_cooldown_seconds: float = Field(
        default=300.0,
        description="数据源被标记为降级后保持冷却的时间(秒)",
    )

    # pytdx 适配器性能参数
    pytdx_pool_size: int = Field(default=3, description="pytdx 连接池大小")
    pytdx_max_workers: int = Field(default=8, description="pytdx 内部线程池最大线程数")
    pytdx_cache_ttl: int = Field(
        default=300, description="pytdx 日线数据内存缓存 TTL(秒)"
    )
    pytdx_kline_timeout: float = Field(
        default=8.0, description="pytdx 主服务器 K 线请求超时时间(秒)"
    )
    pytdx_backup_timeout: float = Field(
        default=15.0, description="pytdx 备用服务器 K 线请求超时时间(秒)"
    )

    # Tushare / Baostock 节流参数
    tushare_min_interval: float = Field(
        default=0.2, description="Tushare 相邻请求最小时间间隔(秒)"
    )
    baostock_min_interval: float = Field(
        default=0.1, description="Baostock 相邻请求最小时间间隔(秒)"
    )

    # 同花顺本地板块缓存文件路径
    ths_conception_path: str = Field(
        default="", description="同花顺 block_conception.ini 本地路径"
    )
    ths_industry_path: str = Field(
        default="", description="同花顺 block_industry.ini 本地路径"
    )


# 全局配置实例
settings = Settings()
