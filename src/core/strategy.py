import concurrent.futures
import time
import warnings
from threading import Lock
import atexit # For registering cleanup

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from src.utils.helpers import retry_on_exception # If still needed by any QuantStrategy method directly
from src.core.data_fetcher import DataFetcher
from src.core import indicators # This will give access to calculate_momentum, calculate_factors, calculate_maN
from src.core.model import train_ml_model

warnings.filterwarnings('ignore')

class QuantStrategy:
    def __init__(self, silent_mode: bool = False): # Add silent_mode parameter
        self.silent_mode = silent_mode # Store it
        self.retry_count = 3 
        self.retry_delay = 1 

        self.data_fetcher = DataFetcher()
        atexit.register(self.data_fetcher.cleanup)

        self.stock_data = {} 
        self.etf_data = {}   
        self.factor_data = {}
        self.factor_cache = {}

        self.performance_stats = {
            'start_time': None, 'end_time': None, 'processed_stocks': 0,
            'success_count': 0, 'failed_count': 0, 'total_retries': 0, # total_retries might be DataFetcher's concern
            'avg_process_time': 0, 'peak_memory': 0, 'peak_cpu': 0
        }

        self.print_lock = Lock()
        self.progress_lock = Lock()
        self.data_lock = Lock()
        
        self.min_workers = 4
        self.max_workers = 10
        self.cpu_threshold = 75
        self.memory_threshold = 85

        self.analysis_stats = {
            'insufficient_data': [], 'invalid_data': [], 'failed_stocks': []
        }

    def __del__(self):
        pass # Logout is handled by DataFetcher's cleanup via atexit

    def analyze_short_term_explosion(self, df: pd.DataFrame) -> float:
        """分析超短线爆发潜力"""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 20:
                # print("Warning: DataFrame for analyze_short_term_explosion is invalid or too short.")
                return 0.0

            required_cols = ['close', 'volume', 'turn', 'macd', 'signal']
            for col in required_cols:
                if col not in df.columns:
                    # print(f"Warning: Missing required column '{col}' in analyze_short_term_explosion.")
                    return 0.0
            
            # Ensure no NaNs in critical columns for calculation (last few rows)
            if df[['close', 'volume', 'turn', 'macd', 'signal']].iloc[-20:].isnull().any().any():
                # print("Warning: NaN values found in critical columns for analyze_short_term_explosion.")
                return 0.0


            # 1. 成交量分析
            recent_volume_mean = df['volume'].iloc[-20:].mean()
            volume_ratio = df['volume'].iloc[-1] / recent_volume_mean if recent_volume_mean > 0 else 0

            # 2. 换手率分析
            recent_turn_mean = df['turn'].iloc[-20:].mean()
            turnover_ratio = df['turn'].iloc[-1] / recent_turn_mean if recent_turn_mean > 0 else 0

            # 3. MACD金叉预判
            last_macd_diff = df['macd'].iloc[-1] - df['signal'].iloc[-1]
            prev_macd_diff = df['macd'].iloc[-2] - df['signal'].iloc[-2]
            macd_cross_score = 1 if (last_macd_diff > 0 and prev_macd_diff < 0) else 0

            # 4. 股价趋势分析
            price_range = df['close'].iloc[-20:].max() - df['close'].iloc[-20:].min()
            if price_range > 0:
                price_position = (df['close'].iloc[-1] - df['close'].iloc[-20:].min()) / price_range
            else:
                price_position = 0.5

            # 5. 短期动量加速
            if len(df['close']) >= 3 and df['close'].iloc[-3] > 0:
                 recent_momentum = df['close'].iloc[-1] / df['close'].iloc[-3] - 1
            else:
                recent_momentum = 0


            volume_ratio = min(max(volume_ratio, 0), 5)
            turnover_ratio = min(max(turnover_ratio, 0), 5)
            recent_momentum = min(max(recent_momentum * 100, -20), 20)

            explosion_score = (
                    volume_ratio * 0.3 +
                    turnover_ratio * 0.2 +
                    macd_cross_score * 0.2 +
                    (1 - price_position) * 0.15 +
                    (recent_momentum + 20) / 40 * 0.15
            )
            return float(explosion_score)

        except Exception as e:
            # print(f"计算超短线爆发潜力时出错: {e}")
            return 0.0

    def analyze_stock(self, stock_info: pd.Series) -> dict | None:
        """分析单个股票"""
        try:
            df = self.data_fetcher.get_stock_daily_data(stock_info['symbol'])
            if df is None or df.empty:
                with self.data_lock:
                    self.analysis_stats['invalid_data'].append(f"{stock_info['name']}({stock_info['symbol']})")
                return None

            if len(df) < 120:
                with self.data_lock:
                    self.analysis_stats['insufficient_data'].append(
                        f"{stock_info['name']}({stock_info['symbol']}): {len(df)}天")
                return None

            if df['close'].isnull().any() or df['volume'].isnull().any(): # Basic check
                with self.data_lock:
                    self.analysis_stats['invalid_data'].append(f"{stock_info['name']}({stock_info['symbol']})")
                return None

            df = indicators.calculate_momentum(df)
            df = indicators.calculate_factors(df) # This already returns df with indicators

            # Convert daily df 'date' to datetime for merge_asof
            # It's already in YYYY-MM-DD string format from get_stock_daily_data
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

            # Fetch and merge annual financial indicators
            annual_indicators_df = self.data_fetcher.get_financial_analysis_indicators(stock_info['symbol'])
            
            annual_cols_to_add = ['eps', 'bvps', 'roe', 'debt_to_asset_ratio', 
                                  'gross_profit_margin', 'net_profit_margin']

            if not annual_indicators_df.empty and 'date' in annual_indicators_df.columns:
                # Ensure annual_indicators_df['date'] is datetime and sorted
                annual_indicators_df['date'] = pd.to_datetime(annual_indicators_df['date'])
                annual_indicators_df.sort_values('date', inplace=True)
                
                # print(f"Daily df for merge_asof (head):\n{df[['date']].head()}") # Debug
                # print(f"Annual indicators for merge_asof (head):\n{annual_indicators_df[['date'] + annual_cols_to_add].head()}") # Debug

                df = pd.merge_asof(df, annual_indicators_df, on='date', direction='backward')
                # print(f"Merged df for {stock_info['symbol']} (head with annual):\n{df[['date'] + annual_cols_to_add].head()}") # Debug
                # print(f"Merged df for {stock_info['symbol']} (tail with annual):\n{df[['date'] + annual_cols_to_add].tail()}") # Debug
            else:
                # If no annual data, ensure the columns exist in df with NaNs
                for col in annual_cols_to_add:
                    df[col] = np.nan


            required_indicator_columns = ['momentum_1m', 'momentum_3m', 'momentum_6m',
                                'volatility', 'vol_ratio', 'rsi', 'macd', 'signal']
            # Check for NaNs only in these technical indicator columns for model input validity
            if any(col not in df.columns or df[col].iloc[-1:].isnull().any() for col in required_indicator_columns):
                # print(f"警告: {stock_info['name']} ({stock_info['symbol']}) 最新技术指标计算结果存在无效值或缺失列")
                return None

            model, scaler = train_ml_model(df)
            if model is None:
                # print(f"警告: {stock_info['name']} ({stock_info['symbol']}) 模型训练失败")
                return None

            latest_data = df.iloc[-1:][required_indicator_columns]
            if latest_data.isnull().any().any():
                # print(f"警告: {stock_info['name']} ({stock_info['symbol']}) 最新数据行包含 NaN，无法预测")
                return None

            latest_scaled = scaler.transform(latest_data)
            prediction = model.predict_proba(latest_scaled)[0][1]

            explosion_score = self.analyze_short_term_explosion(df)

            macd_status = ''
            if df['macd'].iloc[-1] > 0:
                if df['macd'].iloc[-1] > df['signal'].iloc[-1] and df['macd'].iloc[-2] <= df['signal'].iloc[-2]:
                    macd_status = '金叉'
                elif abs(df['macd'].iloc[-1] - df['signal'].iloc[-1]) < abs(
                        df['macd'].iloc[-2] - df['signal'].iloc[-2]):
                    macd_status = '即将金叉'
            
            result_dict = {
                'symbol': stock_info['symbol'], 'name': stock_info['name'], 'market': stock_info['market'],
                'prediction': float(prediction), 
                'momentum_score': float(df['momentum_1m'].iloc[-1]) if 'momentum_1m' in df.columns and pd.notnull(df['momentum_1m'].iloc[-1]) else np.nan,
                'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns and pd.notnull(df['rsi'].iloc[-1]) else np.nan,
                'volatility': float(df['volatility'].iloc[-1]) if 'volatility' in df.columns and pd.notnull(df['volatility'].iloc[-1]) else np.nan,
                'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns and pd.notnull(df['macd'].iloc[-1]) else np.nan,
                'macd_status': macd_status,
                'close': float(df['close'].iloc[-1]) if 'close' in df.columns and pd.notnull(df['close'].iloc[-1]) else np.nan,
                'volume': float(df['volume'].iloc[-1]) if 'volume' in df.columns and pd.notnull(df['volume'].iloc[-1]) else np.nan,
                'turn': float(df['turn'].iloc[-1]) if 'turn' in df.columns and pd.notnull(df['turn'].iloc[-1]) else np.nan,
                'explosion_score': float(explosion_score),
                # Populate annual financial indicators from the last row of merged df
                'eps': float(df['eps'].iloc[-1]) if 'eps' in df.columns and pd.notnull(df['eps'].iloc[-1]) else np.nan,
                'bvps': float(df['bvps'].iloc[-1]) if 'bvps' in df.columns and pd.notnull(df['bvps'].iloc[-1]) else np.nan,
                'roe': float(df['roe'].iloc[-1]) if 'roe' in df.columns and pd.notnull(df['roe'].iloc[-1]) else np.nan,
                'debt_to_asset_ratio': float(df['debt_to_asset_ratio'].iloc[-1]) if 'debt_to_asset_ratio' in df.columns and pd.notnull(df['debt_to_asset_ratio'].iloc[-1]) else np.nan,
                'gross_profit_margin': float(df['gross_profit_margin'].iloc[-1]) if 'gross_profit_margin' in df.columns and pd.notnull(df['gross_profit_margin'].iloc[-1]) else np.nan,
                'net_profit_margin': float(df['net_profit_margin'].iloc[-1]) if 'net_profit_margin' in df.columns and pd.notnull(df['net_profit_margin'].iloc[-1]) else np.nan,
            }
            
            # Final check for NaN in any of the *critical float values for model/scoring* (excluding new financial which can be NaN and are for info)
            # Prediction, momentum_score, rsi, volatility, macd, close, volume, turn, explosion_score
            critical_keys_for_nan_check = [
                'prediction', 'momentum_score', 'rsi', 'volatility', 'macd', 
                'close', 'volume', 'turn', 'explosion_score'
            ]
            for key in critical_keys_for_nan_check:
                if pd.isnull(result_dict[key]):
                    # print(f"警告: {stock_info['name']} ({stock_info['symbol']}) 关键结果 '{key}' 包含NaN值: {result_dict[key]}")
                    return None # Skip stock if critical data is NaN
            return result_dict

        except Exception as e:
            # print(f"分析股票 {stock_info['name']} ({stock_info['symbol']}) 时发生内部错误: {e}") # Consider logging this
            with self.data_lock: # Ensure thread-safe access to shared analysis_stats
                self.analysis_stats['failed_stocks'].append(f"{stock_info['name']}({stock_info['symbol']}): {str(e)}")
            return None

    def get_optimal_thread_count(self) -> int:
        """根据系统资源使用情况动态计算最优线程数（优化版）"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            swap = psutil.swap_memory()
            swap_percent = swap.percent

            self.performance_stats['peak_cpu'] = max(self.performance_stats.get('peak_cpu', 0), cpu_percent)
            self.performance_stats['peak_memory'] = max(self.performance_stats.get('peak_memory', 0), memory_percent)

            system_load = max(cpu_percent / 100, memory_percent / 100, swap_percent / 100)

            if system_load > 0.8:
                optimal_threads = self.min_workers
            elif system_load > 0.6:
                optimal_threads = int(
                    self.min_workers + (self.max_workers - self.min_workers) * (0.8 - system_load) / 0.2)
            else:
                optimal_threads = int(self.max_workers * (0.9 - system_load * 0.5))
            
            optimal_threads = min(max(optimal_threads, self.min_workers), self.max_workers)

            if not self.silent_mode:
                with self.print_lock:
                    print(f"\n系统状态 - CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%, 交换空间: {swap_percent:.1f}%")
                    print(f"可用内存: {memory.available / 1024 / 1024:.0f}MB, 系统负载: {system_load:.2f}")
                    print(f"线程数调整: {optimal_threads}（最小{self.min_workers}, 最大{self.max_workers}）")
                # pass # Silencing prints for cleaner output during refactor -> Now conditional

            return optimal_threads
        except Exception as e:
            if not self.silent_mode:
                print(f"获取系统资源信息时出错: {e}")
            return self.min_workers


    def run_analysis(self) -> pd.DataFrame:
        """运行完整的分析流程（优化版）"""
        if not self.silent_mode:
            print("开始分析...")
        self.performance_stats['start_time'] = time.time()
        self.performance_stats['processed_stocks'] = 0
        self.performance_stats['success_count'] = 0
        self.performance_stats['failed_count'] = 0
        self.analysis_stats = {'insufficient_data': [], 'invalid_data': [], 'failed_stocks': []}

        stocks = self.data_fetcher.get_stock_list()
        if stocks is None or stocks.empty:
            if not self.silent_mode:
                print("未能获取股票列表")
            return pd.DataFrame()

        if not self.silent_mode:
            print(f"共获取到 {len(stocks)} 只股票")
        results = []
        total_stocks = len(stocks)

        def analyze_stock_wrapper(stock_series: pd.Series):
            start_time_stock = time.time()
            try:
                result = self.analyze_stock(stock_series) # Pass the Series directly
                if result:
                    with self.data_lock: # Ensure thread-safe append
                        results.append(result)
                        self.performance_stats['success_count'] += 1
                else:
                    with self.data_lock:
                        self.performance_stats['failed_count'] += 1
                
                process_time = time.time() - start_time_stock
                with self.data_lock:
                    current_processed = self.performance_stats['processed_stocks']
                    current_avg_time = self.performance_stats['avg_process_time']
                    self.performance_stats['avg_process_time'] = \
                        (current_avg_time * current_processed + process_time) / (current_processed + 1) if current_processed > 0 else process_time
                    self.performance_stats['processed_stocks'] += 1
                return result is not None
            except Exception as e:
                # print(f"\n分析股票 {stock_series['symbol']} 时出错 (wrapper): {e}")
                with self.data_lock:
                    self.performance_stats['failed_count'] += 1
                return False

        thread_count = self.get_optimal_thread_count()
        batch_size = min(max(20, len(stocks) // (thread_count * 2 if thread_count > 0 else 1)), 100)


        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            with tqdm(total=total_stocks, desc="分析进度", ncols=100, mininterval=0.5) as pbar:
                futures = []
                for i in range(0, len(stocks), batch_size):
                    current_thread_count = self.get_optimal_thread_count()
                    if executor._max_workers != current_thread_count:
                        if not self.silent_mode:
                            print(f"Adjusting executor max_workers to {current_thread_count}")
                        executor._max_workers = current_thread_count
                    
                    batch_stocks_df = stocks.iloc[i:i + batch_size]
                    for _, stock_series in batch_stocks_df.iterrows(): # Iterate over Series
                        futures.append(executor.submit(analyze_stock_wrapper, stock_series))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result() # We handle results inside wrapper, just raise exceptions here
                    except Exception as e:
                        if not self.silent_mode:
                            print(f"\n处理股票时发生错误 (future): {e}")
                        # pass # Already handled in wrapper or by failed_count
                    pbar.update(1)
                    
                    # Optional: Intermediate performance stats display (can be noisy)
                    if not self.silent_mode and pbar.n % (batch_size * 2) == 0: # Every few batches
                         with self.print_lock:
                             elapsed_time = time.time() - self.performance_stats['start_time']
                             success_rate = (self.performance_stats['success_count'] /
                                             max(1, self.performance_stats['processed_stocks']) * 100) if self.performance_stats['processed_stocks'] > 0 else 0
                             print(f"\n性能统计 (Batch {i // batch_size +1}):")
                             print(f"处理股票数: {self.performance_stats['processed_stocks']}")
                             print(f"成功率: {success_rate:.1f}%")
                             print(f"平均处理时间: {self.performance_stats['avg_process_time']:.2f}秒")
                             print(f"总耗时: {elapsed_time:.0f}秒")
                             print(f"峰值CPU使用率: {self.performance_stats.get('peak_cpu',0):.1f}%")
                             print(f"峰值内存使用率: {self.performance_stats.get('peak_memory',0):.1f}%")
                
                # Final update after loop if some tasks didn't trigger pbar update (e.g. due to mininterval)
                if self.performance_stats['processed_stocks'] > 0 : # Avoid division by zero if no stocks processed
                     pbar.n = self.performance_stats['processed_stocks']
                     pbar.refresh()


        self.performance_stats['end_time'] = time.time()
        total_time_secs = self.performance_stats['end_time'] - self.performance_stats['start_time']
        if not self.silent_mode:
            print(f"\n分析完成: 总耗时: {total_time_secs:.0f}秒, 成功: {self.performance_stats['success_count']}, 失败: {self.performance_stats['failed_count']}")

        if not results:
            if not self.silent_mode:
                print("没有符合条件的股票")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        if results_df.empty: # Should be caught by `if not results` already, but defensive check.
            return results_df

        results_df['total_score'] = (
            results_df['prediction'] * 0.3 +
            results_df['momentum_score'] * 0.2 +
            results_df['explosion_score'] * 0.35 +
            (1 - results_df['volatility'].clip(upper=1.0)) * 0.15 # Ensure volatility doesn't make score negative
        )

        prediction_filter = results_df['prediction'] > 0.6
        momentum_filter = results_df['momentum_score'] > -0.1
        rsi_upper_filter = results_df['rsi'] < 75
        rsi_lower_filter = results_df['rsi'] > 30
        volatility_filter = results_df['volatility'] < 0.6
        volume_filter = results_df['volume'] > 50000  # Ensure 'volume' exists and is not NaN
        price_filter = results_df['close'] > 3       # Ensure 'close' exists and is not NaN

        # New financial indicator filters
        # Assuming NaNs are handled such that direct comparison is safe, or they evaluate to False.
        # For PE and PB, we only want positive values.
        pe_filter = (results_df['pe'].fillna(float('inf')) > 0) & (results_df['pe'].fillna(float('inf')) < 50)
        pb_filter = (results_df['pb'].fillna(float('inf')) > 0) & (results_df['pb'].fillna(float('inf')) < 5)
        roe_filter = results_df['roe'].fillna(-float('inf')) > 0.05 # ROE > 5%
        debt_to_asset_filter = results_df['debt_to_asset_ratio'].fillna(float('inf')) < 0.8 # Debt to Asset Ratio < 80%
        
        if not self.silent_mode:
            print(f"\n筛选条件统计：")
            print(f"预测概率 > 60% 的股票数: {prediction_filter.sum()}")
            print(f"动量得分 > -10% 的股票数: {momentum_filter.sum()}")
            print(f"RSI在30-75之间的股票数: {(rsi_upper_filter & rsi_lower_filter).sum()}")
            print(f"波动率 < 60% 的股票数: {volatility_filter.sum()}")
            print(f"日均成交量 > 5万的股票数: {volume_filter.sum()}")
            print(f"股价 > 3元的股票数: {price_filter.sum()}")
            print(f"P/E (0-50) 的股票数: {pe_filter.sum()}")
            print(f"P/B (0-5) 的股票数: {pb_filter.sum()}")
            print(f"ROE > 5% 的股票数: {roe_filter.sum()}")
            print(f"负债资产率 < 80% 的股票数: {debt_to_asset_filter.sum()}")

        selected_df = results_df[
            prediction_filter & momentum_filter & rsi_upper_filter &
            rsi_lower_filter & volatility_filter & volume_filter & price_filter &
            pe_filter & pb_filter & roe_filter & debt_to_asset_filter
        ].sort_values('explosion_score', ascending=False)
        
        if not self.silent_mode:
            print(f"\n最终符合所有条件的股票数: {len(selected_df)}")

        if len(selected_df) > 50:
            if not self.silent_mode:
                print(f"保留得分最高的前50只股票")
            selected_df = selected_df.head(50)
        
        if not self.silent_mode:
            print("\n=== 数据质量说明 ===")
            print(f"   - 要求至少120天的历史数据（不满足的股票数：{len(self.analysis_stats['insufficient_data'])}）")
            print(f"   - 要求数据完整无缺失（数据无效的股票数：{len(self.analysis_stats['invalid_data'])}）")
            print(f"   - 技术指标计算正常（分析失败的股票数：{len(self.analysis_stats['failed_stocks'])}）")
        
        return selected_df

    def calculate_ma_for_stocks(self, stock_df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
        """
        Calculates the moving average for a list of stocks in a DataFrame.
        Adds a new column 'maN' (e.g., 'ma15') to the DataFrame.
        This method is intended to replace the calculate_ma15_for_main function.
        """
        if stock_df.empty or 'symbol' not in stock_df.columns:
            # print("Stock DataFrame is empty or 'symbol' column is missing for MA calculation.")
            stock_df[f'ma{window}'] = np.nan # Add empty column if it doesn't exist
            return stock_df

        ma_values = []
        # print(f"Calculating MA{window} for {len(stock_df)} stocks...")
        # Use tqdm for progress if it's a long list
        for symbol in tqdm(stock_df['symbol'], desc=f"Calculating MA{window}", ncols=80, leave=False, disable=self.silent_mode):
            try:
                # Fetch daily data for the symbol
                daily_df = self.data_fetcher.get_stock_daily_data(symbol)
                if daily_df is not None and not daily_df.empty:
                    # Calculate MA using the function from indicators module
                    ma_val = indicators.calculate_maN(daily_df, window=window)
                    ma_values.append(ma_val)
                else:
                    ma_values.append(np.nan)
            except Exception as e:
                # print(f"Error calculating MA{window} for {symbol}: {e}")
                ma_values.append(np.nan)
        
        stock_df[f'ma{window}'] = ma_values
        return stock_df

# Example of how QuantStrategy might be used (similar to the old __main__ block)
if __name__ == '__main__':
    # Example of how QuantStrategy might be used (similar to the old __main__ block)
if __name__ == '__main__':
    # This example block in strategy.py is for testing the class directly.
    # app.py will handle command-line arguments for silent_mode.
    # Here, we can test both modes if desired, or just default.
    
    # Test with normal output
    print("Running QuantStrategy example (normal mode)...")
    strategy_instance_normal = QuantStrategy(silent_mode=False)
    selected_stocks_df_normal = strategy_instance_normal.run_analysis()
    
    if not selected_stocks_df_normal.empty:
        print(f"\nNormal mode: Analysis returned {len(selected_stocks_df_normal)} selected stocks.")
        selected_stocks_with_ma15_normal = strategy_instance_normal.calculate_ma_for_stocks(selected_stocks_df_normal.copy(), window=15)
        print("\nNormal mode: Selected stocks with MA15 (first 5):")
        print(selected_stocks_with_ma15_normal[['name', 'symbol', 'close', 'ma15', 'explosion_score']].head())
    else:
        print("\nNormal mode: No stocks selected by the analysis.")
    
    print("\n" + "="*50 + "\n")

    # Test with silent output (fewer prints, no tqdm bar)
    print("Running QuantStrategy example (silent mode)...")
    strategy_instance_silent = QuantStrategy(silent_mode=True)
    selected_stocks_df_silent = strategy_instance_silent.run_analysis()
    
    if not selected_stocks_df_silent.empty:
        print(f"\nSilent mode: Analysis returned {len(selected_stocks_df_silent)} selected stocks.")
        # MA calculation might still print tqdm if not disabled there too, but strategy.py's tqdm is now silent.
        selected_stocks_with_ma15_silent = strategy_instance_silent.calculate_ma_for_stocks(selected_stocks_df_silent.copy(), window=15)
        print("\nSilent mode: Selected stocks with MA15 (first 5 - note: MA calc tqdm might still show if not internally silenced by its own tqdm call):")
        print(selected_stocks_with_ma15_silent[['name', 'symbol', 'close', 'ma15', 'explosion_score']].head())
        
        # In silent mode, we wouldn't typically do extensive printing like this,
        # but for testing the class, we might want to see the result.
        # The JSON output would be handled by app.py.
        columns_map = {
            'name': '股票名称', 'symbol': '股票代码', 'market': '交易所-板块',
            'prediction': '上涨概率', 'momentum_score': '动量得分', 'rsi': 'RSI指标',
            'close': '收盘价', 'ma15': '15日均线价格', 'explosion_score': '爆发潜力值',
            'macd_status': 'MACD状态',
            # Adding new financial indicators to the display map
            'eps': '每股收益(年)', 'bvps': '每股净资产(年)', 'roe': '净资产收益率(年)',
            'debt_to_asset_ratio': '资产负债率(年)', 
            'gross_profit_margin': '销售毛利率(年)', 'net_profit_margin': '销售净利率(年)'
        }
        # Ensure only existing columns are selected for renaming
        existing_cols_in_map = {k: v for k, v in columns_map.items() if k in selected_stocks_with_ma15.columns}
        display_df = selected_stocks_with_ma15[list(existing_cols_in_map.keys())].copy()
        display_df.rename(columns=existing_cols_in_map, inplace=True)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        # pd.set_option('display.unicode.ambiguous_as_wide', True) 
        # pd.set_option('display.unicode.east_asian_width', True)

        # Example of what app.py might do with the JSON output
        # results_json = selected_stocks_with_ma15_silent.to_json(orient='records', force_ascii=False, indent=2)
        # print("\nSilent mode: Example JSON output (first ~500 chars):")
        # print(results_json[:500] + "..." if len(results_json) > 500 else results_json)

    else:
        print("\nSilent mode: No stocks selected by the analysis.")

    print("\nQuantStrategy example run completed for both modes.")
