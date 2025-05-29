import time
import os
import sys
from datetime import datetime
from threading import Lock

import pandas as pd
import numpy as np # Although not directly used by the moved functions, it's often used with pandas

try:
    import akshare as ak
except ImportError as e:
    print(f"导入akshare时出错: {e}")
    print("尝试重新安装akshare...")
    os.system("pip install --upgrade akshare py-mini-racer")
    import akshare as ak

import baostock as bs

from src.utils.helpers import retry_on_exception

class DataFetcher:
    def __init__(self):
        self.retry_count = 3
        self.retry_delay = 1
        self._init_baostock()
        self.stock_data_cache = {}  # 股票日线数据缓存
        self.cache_lock = Lock()  # 添加缓存锁
        # performance_stats might be needed if get_stock_daily_data uses it for total_retries
        # For now, assuming it's not essential for DataFetcher's core responsibility
        self.performance_stats = {'total_retries': 0} # Minimal placeholder

    def _init_baostock(self):
        """初始化 baostock 连接"""
        for i in range(self.retry_count):
            try:
                bs.login()
                return
            except Exception as e:
                if i == self.retry_count - 1:
                    print(f"baostock登录失败: {e}")
                    sys.exit(1)
                print(f"baostock登录重试 ({i + 1}/{self.retry_count})...")
                time.sleep(self.retry_delay)

    def cleanup(self):
        """程序退出时的清理函数"""
        try:
            bs.logout()
            print("Baostock logout successful.")
        except Exception as e:
            print(f"Error during Baostock logout: {e}")

    @retry_on_exception(retries=3, delay=1)
    def get_stock_list(self):
        """获取股票列表"""
        try:
            # 使用 akshare 获取A股列表
            stock_info = ak.stock_zh_a_spot_em()

            # 选择需要的列并重命名
            stock_info = stock_info[['代码', '名称']].copy()
            stock_info.columns = ['symbol', 'name']

            # 过滤掉退市和ST股票
            stock_info = stock_info[
                ~stock_info['name'].str.contains('退市|退') &
                ~stock_info['name'].str.contains('ST')
                ]

            # 添加交易所和板块信息
            def get_market_info(symbol):
                if symbol.startswith('60'):
                    return '上交所-主板'
                elif symbol.startswith('688'):
                    return '上交所-科创板'
                elif symbol.startswith('000'):
                    return '深交所-主板'
                elif symbol.startswith('002'):
                    return '深交所-中小板'
                elif symbol.startswith('300'):
                    return '深交所-创业板'
                elif symbol.startswith('301'):
                    return '深交所-创业板'
                elif symbol.startswith('430'):
                    return '北交所-主板'
                elif symbol.startswith('83'):
                    return '北交所-主板'
                elif symbol.startswith('87'):
                    return '北交所-创新层'
                elif symbol.startswith('889'):
                    return '北交所-基础层'
                else:
                    return '其他'

            stock_info['market'] = stock_info['symbol'].apply(get_market_info)

            # 添加完整代码（带上交所信息）
            stock_info['ts_code'] = stock_info.apply(
                lambda x: x['symbol'] + '.SH' if x['market'].startswith('上交所') else (
                    x['symbol'] + '.BJ' if x['market'].startswith('北交所') else x['symbol'] + '.SZ'
                ), axis=1
            )

            print(f"成功获取股票列表，共 {len(stock_info)} 只股票")
            print("\n各市场股票数量统计：")
            market_stats = stock_info['market'].value_counts()
            for market, count in market_stats.items():
                print(f"{market}: {count}只")

            return stock_info
        except Exception as e:
            print(f"获取股票列表时出错: {e}")
            return pd.DataFrame()

    def get_etf_list(self):
        """获取ETF基金列表"""
        try:
            # 获取场内ETF列表
            etf_list = ak.fund_etf_category_sina()
            return etf_list
        except Exception as e:
            print(f"获取ETF列表时出错: {e}")
            return pd.DataFrame()

    @retry_on_exception(retries=3, delay=1)
    def get_stock_daily_data(self, symbol, start_date='20220101'):
        """获取股票日线数据（优化版），包含基本面指标。"""
        # 检查缓存
        cache_key = f"{symbol}_{start_date}_fundamental" # Updated cache key for new data type
        with self.cache_lock:
            if cache_key in self.stock_data_cache:
                return self.stock_data_cache[cache_key]

        try:
            # 1. 获取历史价格数据
            df_price = pd.DataFrame() # Initialize to ensure it's defined
            for _ in range(self.retry_count):
                try:
                    df_price = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=datetime.now().strftime('%Y%m%d'),
                        adjust="qfq"
                    )
                    break
                except Exception as e:
                    print(f"获取股票 {symbol} 价格数据失败，重试...")
                    time.sleep(self.retry_delay)
                    self.performance_stats['total_retries'] += 1
            else:
                print(f"获取股票 {symbol} 价格数据失败，跳过该股票")
                return pd.DataFrame()

            if df_price.empty:
                return df_price

            # 重命名和选择价格数据列
            df_price = df_price[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']]
            df_price.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'change', 'turn']
            
            # 确保价格数据日期格式为 'YYYY-MM-DD' (ak.stock_zh_a_hist usually returns this)
            # And convert to datetime objects for robust merging, then back to string if needed.
            df_price['date'] = pd.to_datetime(df_price['date']).dt.strftime('%Y-%m-%d')


            # 2. 获取基本面数据
            df_fundamental = pd.DataFrame() # Initialize
            try:
                # print(f"Fetching fundamental data for {symbol}...") # Debug print
                df_fundamental = ak.stock_a_indicator_lg(symbol=symbol)
                # print(f"Fundamental data for {symbol} head:\n{df_fundamental.head()}") # Debug print
            except Exception as e:
                print(f"获取股票 {symbol} 基本面数据时出错: {e}. 将仅使用价格数据。")
                # Proceed without fundamental data if it fails
                df_fundamental = pd.DataFrame() # Ensure it's an empty DataFrame

            if not df_fundamental.empty:
                # 3. 重命名 trade_date to date
                df_fundamental = df_fundamental.rename(columns={'trade_date': 'date'})
                
                # 4. 确保基本面数据日期格式为 'YYYY-MM-DD'
                # ak.stock_a_indicator_lg returns date as YYYYMMDD string or datetime object.
                # Convert to datetime then format to string to be sure.
                df_fundamental['date'] = pd.to_datetime(df_fundamental['date']).dt.strftime('%Y-%m-%d')

                # 选择需要的列 (pe, pb, ps, dv_ratio, total_mv) and date for merging
                fundamental_cols_to_keep = ['date', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv']
                # Filter out columns not present in df_fundamental to avoid KeyError
                actual_fundamental_cols = [col for col in fundamental_cols_to_keep if col in df_fundamental.columns]
                df_fundamental = df_fundamental[actual_fundamental_cols]
                
                # 5. 左合并价格数据和基本面数据
                # print(f"Merging price data (shape: {df_price.shape}) with fundamental data (shape: {df_fundamental.shape}) for {symbol}") # Debug
                merged_df = pd.merge(df_price, df_fundamental, on='date', how='left')
                # print(f"Merged data for {symbol} head:\n{merged_df.head()}") # Debug
            else:
                # print(f"No fundamental data found or fetched for {symbol}. Using only price data.") # Debug
                merged_df = df_price # If no fundamental data, use only price data

            # 确保数据类型正确 (for original price columns, fundamental columns are usually float)
            numeric_price_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'turn', 'pct_chg']
            for col in numeric_price_columns:
                if col in merged_df.columns: # Check if column exists after merge
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            
            # Fundamental columns from ak.stock_a_indicator_lg are usually float.
            # Explicitly convert just in case, and handle if they weren't merged.
            fundamental_numeric_cols = ['pe', 'pb', 'ps', 'dv_ratio', 'total_mv']
            for col in fundamental_numeric_cols:
                if col in merged_df.columns:
                     merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')


            # 删除任何在关键价格列中包含NaN的行 (open, close, high, low, volume)
            # Fundamental data NaNs are often expected and should be handled by downstream analysis (e.g. imputation or ignoring)
            critical_price_cols_for_dropna = ['open', 'close', 'high', 'low', 'volume']
            merged_df = merged_df.dropna(subset=critical_price_cols_for_dropna)
            
            # Fill NaNs in fundamental columns with 0 or a specific marker if preferred,
            # or leave them for downstream processing. For now, let's fill with 0 for simplicity,
            # as factors calculation might expect numeric values.
            # This is a simple imputation strategy. More sophisticated ones could be used.
            for col in fundamental_numeric_cols:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(0)


            # 存入缓存
            with self.cache_lock:
                self.stock_data_cache[cache_key] = merged_df

            return merged_df

        except Exception as e:
            print(f"获取股票 {symbol} 日线及基本面数据时出错: {e}")
            return pd.DataFrame()

    @retry_on_exception(retries=3, delay=1)
    def get_financial_analysis_indicators(self, symbol: str) -> pd.DataFrame:
        """
        Fetches key annual financial analysis indicators for a given stock symbol.
        Returns a DataFrame with columns ['date', 'eps', 'bvps', 'roe', 
        'debt_to_asset_ratio', 'gross_profit_margin', 'net_profit_margin'],
        where 'date' contains the report dates.
        """
        # Define expected columns for the output, even in case of error
        expected_cols = ['date', 'eps', 'bvps', 'roe', 'debt_to_asset_ratio', 
                           'gross_profit_margin', 'net_profit_margin']
        try:
            # print(f"Fetching financial analysis indicators for {symbol} (annual)...") # Debug
            raw_financial_df = ak.stock_financial_analysis_indicator(symbol=symbol, indicator="年度")
            
            if raw_financial_df.empty:
                # print(f"No annual financial analysis data found for {symbol}.")
                return pd.DataFrame(columns=expected_cols)

            # Transpose the DataFrame
            transposed_df = raw_financial_df.set_index('指标').T
            
            # Convert report date index to datetime objects
            transposed_df.index = pd.to_datetime(transposed_df.index, format='%Y%m%d')
            
            indicators_map = {
                '每股收益': 'eps',
                '每股净资产': 'bvps',
                '净资产收益率': 'roe', # This is ROE - 摊薄
                '资产负债率': 'debt_to_asset_ratio',
                '销售毛利率': 'gross_profit_margin',
                '销售净利率': 'net_profit_margin'
            }
            
            # Filter for required indicators (now columns after transpose)
            required_cn_indicators = list(indicators_map.keys())
            
            # Select only the indicators present in the transposed DataFrame
            # Some indicators might be missing for some stocks/years
            available_cn_indicators = [cn_name for cn_name in required_cn_indicators if cn_name in transposed_df.columns]
            
            if not available_cn_indicators:
                # print(f"None of the required financial indicators found for {symbol} after transpose.")
                return pd.DataFrame(columns=expected_cols)
                
            processed_df = transposed_df[available_cn_indicators].copy() # Use .copy() to avoid SettingWithCopyWarning
            
            # Rename columns to English names
            processed_df.rename(columns={cn: en for cn, en in indicators_map.items() if cn in available_cn_indicators}, inplace=True)
            
            # Convert all indicator columns to numeric, coercing errors
            for col_en_name in indicators_map.values():
                if col_en_name in processed_df.columns:
                    processed_df[col_en_name] = pd.to_numeric(processed_df[col_en_name], errors='coerce')
            
            # Reset index to make 'date' a column
            processed_df.reset_index(inplace=True)
            processed_df.rename(columns={'index': 'date'}, inplace=True)
            
            # Ensure all expected columns are present, adding missing ones with NaN
            for col in expected_cols:
                if col not in processed_df.columns:
                    processed_df[col] = np.nan
            
            # Select and reorder to final expected columns
            final_df = processed_df[expected_cols]
            
            # print(f"Processed financial indicators for {symbol}:\n{final_df.head()}") # Debug
            return final_df

        except Exception as e:
            print(f"获取股票 {symbol} 年度财务分析指标时出错: {e}")
            return pd.DataFrame(columns=expected_cols)


if __name__ == '__main__':
    # Example Usage (optional, for testing)
    fetcher = DataFetcher()
    
    # Test get_stock_list
    print("\nTesting get_stock_list...")
    stock_list = fetcher.get_stock_list()
    if not stock_list.empty:
        print(f"First 5 stocks:\n{stock_list.head()}")
    else:
        print("Failed to get stock list or list is empty.")

    # Test get_etf_list
    print("\nTesting get_etf_list...")
    etf_list = fetcher.get_etf_list()
    if not etf_list.empty:
        print(f"First 5 ETFs:\n{etf_list.head()}")
    else:
        print("Failed to get ETF list or list is empty.")

    # Test get_stock_daily_data (example with a common stock symbol)
    if not stock_list.empty:
        example_symbol = stock_list['symbol'].iloc[0] # Use the first stock from the list
        print(f"\nTesting get_stock_daily_data for symbol: {example_symbol}...")
        stock_data = fetcher.get_stock_daily_data(example_symbol)
        if not stock_data.empty:
            print(f"Data for {example_symbol} (first 5 rows):\n{stock_data.head()}")
            # Also test the new financial indicators method
            print(f"\nTesting get_financial_analysis_indicators for symbol: {example_symbol}...")
            financial_indicators = fetcher.get_financial_analysis_indicators(example_symbol)
            if not financial_indicators.empty:
                print(f"Financial indicators for {example_symbol}:\n{financial_indicators}")
            else:
                print(f"Failed to get financial indicators for {example_symbol} or data is empty.")
        else:
            print(f"Failed to get daily data for {example_symbol} or data is empty.")
        
        # Test with a different symbol that might have data
        example_symbol_2 = "600519" # Kweichow Moutai - often has good data
        print(f"\nTesting get_financial_analysis_indicators for symbol: {example_symbol_2}...")
        financial_indicators_2 = fetcher.get_financial_analysis_indicators(example_symbol_2)
        if not financial_indicators_2.empty:
            print(f"Financial indicators for {example_symbol_2}:\n{financial_indicators_2}")
        else:
            print(f"Failed to get financial indicators for {example_symbol_2} or data is empty.")

    else:
        # Fallback symbol if stock list is empty
        example_symbol_fallback = "000001" # Ping An Bank
        print(f"\nTesting get_stock_daily_data for fallback symbol: {example_symbol_fallback}...")
        stock_data_fallback = fetcher.get_stock_daily_data(example_symbol_fallback)
        if not stock_data_fallback.empty:
            print(f"Data for {example_symbol_fallback} (first 5 rows):\n{stock_data_fallback.head()}")
            print(f"\nTesting get_financial_analysis_indicators for fallback symbol: {example_symbol_fallback}...")
            financial_indicators_fallback = fetcher.get_financial_analysis_indicators(example_symbol_fallback)
            if not financial_indicators_fallback.empty:
                print(f"Financial indicators for {example_symbol_fallback}:\n{financial_indicators_fallback}")
            else:
                print(f"Failed to get financial indicators for {example_symbol_fallback} or data is empty.")
        else:
            print(f"Failed to get daily data for {example_symbol_fallback} or data is empty.")
            
    # Ensure cleanup is called at the end if you run this file directly
    import atexit
    atexit.register(fetcher.cleanup)
    print("\nDataFetcher tests completed. Waiting for atexit cleanup...")
