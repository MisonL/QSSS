"""
Quantitative Stock Selection System (QSSS)
A professional A-share quantitative trading strategy system
"""

__version__ = "2.0.0"
__author__ = "Mison"
__email__ = "1360962086@qq.com"

__all__ = ["QuantStrategy", "DataManager", "Settings"]


def __getattr__(name):
    """Lazy-load public classes so package import stays lightweight."""
    if name == "QuantStrategy":
        from .core.strategy import QuantStrategy

        return QuantStrategy
    if name == "DataManager":
        from .data.manager import DataManager

        return DataManager
    if name == "Settings":
        from .config.settings import Settings

        return Settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
