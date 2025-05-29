# import concurrent.futures -> No longer directly used in main.py __main__
import sys # For sys.exit if needed
import time # For retry delay
import argparse # New import
import json     # New import
# import warnings -> warnings.filterwarnings('ignore') can be removed if not needed or moved
# from threading import Lock -> No longer needed in main.py
# from src.utils.helpers import retry_on_exception -> Not used here
# from src.core.data_fetcher import DataFetcher -> Not used directly here
# from src.core import indicators -> Not used directly here
# from src.core.model import train_ml_model -> Not used directly here
# import numpy as np -> Not used directly here
import pandas as pd # Still needed for DataFrame display options
# import psutil -> Not used here
# import atexit -> Not used here
# from tqdm import tqdm -> Not used here

from src.core.strategy import QuantStrategy # Import the refactored class

# warnings.filterwarnings('ignore') # Can be removed or kept based on preference


# class QuantStrategy: # Definition moved to src/core/strategy.py
#    ... (all class content removed) ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative Stock Analysis Strategy")
    parser.add_argument('--json-output', action='store_true', help="Output results as JSON to stdout")
    args = parser.parse_args()

    # 设置最大重试次数
    max_retries = 3
    retry_delay = 2
    strategy_instance = None # Initialize to None

    # 修改浮点数显示格式 (This can stay as it affects display)
    # Only set these options if not in JSON output mode, to avoid interfering with JSON format
    if not args.json_output:
        pd.set_option('display.float_format', lambda x: '{:.2%}'.format(x) if isinstance(x, float) and x < 10 and (
                    'prediction' in str(x) or 'rsi' in str(x) or 'explosion_score' in str(x)) else (
            '{:.2f}'.format(x / 100000000) if isinstance(x, (int, float)) and 'volume' in str(x) else '{:.2f}'.format(x) # Added int check for volume
        ))
    
    # Attempt to instantiate QuantStrategy with retries
    for attempt in range(max_retries):
        try:
            if not args.json_output:
                print(f"Attempting to initialize QuantStrategy ({attempt + 1}/{max_retries})...")
            # Pass silent_mode based on args.json_output
            strategy_instance = QuantStrategy(silent_mode=args.json_output) 
            if not args.json_output:
                print("QuantStrategy initialized successfully.")
            break # Exit loop if successful
        except Exception as e:
            if not args.json_output:
                print(f"Error during QuantStrategy initialization: {e}")
            if attempt == max_retries - 1:
                if not args.json_output:
                    print("Max retries reached. Failed to initialize QuantStrategy.")
                # If JSON output is expected, we might want to output a JSON error
                if args.json_output:
                    print(json.dumps({"error": "Failed to initialize QuantStrategy", "details": str(e)}))
                sys.exit(1) # Exit if all retries fail
            if not args.json_output:
                print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    if strategy_instance is None: # Should be caught by sys.exit above, but as a safeguard
        if not args.json_output:
            print("Exiting: QuantStrategy could not be initialized.")
        else:
            print(json.dumps({"error": "QuantStrategy could not be initialized."}))
        sys.exit(1)

    # Run the analysis
    if not args.json_output:
        print("Running stock analysis...")
    selected_stocks_df = strategy_instance.run_analysis()

    if not selected_stocks_df.empty:
        if not args.json_output:
            print(f"\nAnalysis returned {len(selected_stocks_df)} selected stocks.")
        
        # Calculate MA15 for the selected stocks using the method from QuantStrategy
        if not args.json_output:
            print("Calculating MA15 for selected stocks...")
        # Pass a copy to avoid SettingWithCopyWarning if strategy modifies it internally
        selected_stocks_with_ma15_df = strategy_instance.calculate_ma_for_stocks(selected_stocks_df.copy(), window=15) 
        if not args.json_output:
            print("MA15 calculation complete.")

        if args.json_output:
            # Convert all relevant columns to a JSON string
            # We use selected_stocks_with_ma15_df as it contains the MA values
            results_json = selected_stocks_with_ma15_df.to_json(orient='records', force_ascii=False, date_format='iso')
            print(results_json)
            sys.exit(0)

        # Standard console output logic (if not args.json_output)
        # 更新列名映射 (Column names in selected_stocks_with_ma15_df should match this map)
        columns_map = {
            'name': '股票名称', 'symbol': '股票代码', 'market': '交易所-板块',
            'prediction': '上涨概率', 'momentum_score': '动量得分', 'rsi': 'RSI指标',
            'close': '收盘价', 'ma15': '15日均线价格', 
            'explosion_score': '爆发潜力值', 'macd_status': 'MACD状态',
            # New financial indicators for display
            'pe': '市盈率(PE)', 'pb': '市净率(PB)', 'roe': '净资产收益率(ROE)',
            'debt_to_asset_ratio': '资产负债率',
            'eps': '每股收益(年)', 'bvps': '每股净资产(年)', # Added from strategy.py's map
            'gross_profit_margin': '销售毛利率(年)', 'net_profit_margin': '销售净利率(年)' # Added from strategy.py's map
        }
        
        # Ensure only existing columns are selected for renaming
        # This also handles if 'ma15' or other columns are somehow missing
        existing_cols_in_map = {k: v for k, v in columns_map.items() if k in selected_stocks_with_ma15_df.columns}
        display_df = selected_stocks_with_ma15_df[list(existing_cols_in_map.keys())].copy()
        display_df.rename(columns=existing_cols_in_map, inplace=True)

        # 设置pandas显示选项 (These can stay)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("\n=== 选出的标的（前20名） ===")
        # print("注：MACD状态标记 - 金叉：MACD在0轴上方形成金叉，即将金叉：MACD在0轴上方且即将形成金叉") # Note was in strategy.py, can add back if needed
        print(display_df.head(20).to_string(index=False, justify='center'))

        # 筛选15日均线在15元以内的股票
        if '15日均线价格' in display_df.columns: # Check if column exists after mapping
            low_price_stocks = display_df[display_df['15日均线价格'] <= 15].copy()
            if not low_price_stocks.empty:
                print("\n=== 15日均线在15元以内的标的 ===")
                print(low_price_stocks.to_string(index=False, justify='center'))
        else:
            print("\n'15日均线价格' column not found in display_df for low price stock filtering.")

        # 显示超短线爆发潜力股票
        # Ensure 'explosion_score' column exists before filtering
        if 'explosion_score' in selected_stocks_with_ma15_df.columns:
            explosion_stocks_df = selected_stocks_with_ma15_df[
                selected_stocks_with_ma15_df['explosion_score'] > 1.5
            ].sort_values('explosion_score', ascending=False)

            if not explosion_stocks_df.empty:
                # 更新爆发潜力股票的列名映射
                explosion_columns = {
                    'name': '股票名称', 'symbol': '股票代码', 'market': '交易所-板块',
                    'explosion_score': '爆发潜力值', 'volume': '成交量（亿）', 
                    'turn': '换手率', 'close': '收盘价', 'ma15': '15日均线价格',
                    'rsi': 'RSI指标', 'macd_status': 'MACD状态',
                    # Adding new financial indicators to explosion_df display map
                    'pe': '市盈率(PE)', 'pb': '市净率(PB)', 'roe': '净资产收益率(ROE)',
                    'debt_to_asset_ratio': '资产负债率',
                    'eps': '每股收益(年)', 'bvps': '每股净资产(年)' 
                    # Not adding margins here to keep explosion table more focused, can be added if needed
                }
                # Ensure only existing columns for explosion_df
                existing_explosion_cols = {k:v for k,v in explosion_columns.items() if k in explosion_stocks_df.columns}
                final_explosion_df = explosion_stocks_df[list(existing_explosion_cols.keys())].copy()
                final_explosion_df.rename(columns=existing_explosion_cols, inplace=True)


                print("\n=== 超短线爆发潜力股票（前20名） ===")
                print(final_explosion_df.head(20).to_string(index=False, justify='center'))

                # 筛选15日均线在15元以内的爆发潜力股票
                # Ensure '15日均线价格' column exists before filtering
                if '15日均线价格' in final_explosion_df.columns:
                    low_price_explosion_df = final_explosion_df[final_explosion_df['15日均线价格'] <= 15].copy()
                    if not low_price_explosion_df.empty:
                        print("\n=== 15日均线在15元以内的爆发潜力股票 ===")
                        print(low_price_explosion_df.to_string(index=False, justify='center'))
                else:
                    print("\n'15日均线价格' column not found in explosion_df for filtering.") # Should be final_explosion_df
            else:
                print("\nNo stocks meet the explosion potential criteria (>1.5).")
        else:
            print("\n'explosion_score' column not found in selected stocks for explosion potential analysis.") # Should be selected_stocks_with_ma15_df

            print("\n超短线选股说明：") # This and following sections are part of normal console output
            print("1. 爆发潜力值计算：")
            print("   - 成交量突增：权重30%")
            print("   - 换手率变化：权重20%")
            print("   - MACD金叉：权重20%")
            print("   - 价格位置：权重15%")
            print("   - 短期动量：权重15%")

            print("\n2. 建议关注：")
            print("   - 成交量和换手率较前期显著放大的股票")
            print("   - MACD即将金叉或刚形成金叉的股票")
            print("   - 股价处于近期低位但有上涨趋势的股票")
            print("   - 15日均线呈现向上趋势的股票")

            print("\n3. 风险提示：")
            print("   - 超短线交易属于高风险操作，建议仅供技术分析参考")
            print("   - 入场前请务必结合分时走势、盘口数据和市场情绪")
            print("   - 注意设置止损，控制好仓位，严格执行交易纪律")

        # 添加分析说明
        print("\n=== 选股条件说明 ===")
        print("1. 基础筛选条件：")
        print("   - 上涨概率 > 60%（基于机器学习模型预测）")
        print("   - 动量得分 > -10%（20日涨跌幅）")
        print("   - RSI指标：30-75之间（避免超买超卖）")
        print("   - 波动率 < 60%（年化计算，控制风险）")
        print("   - 日均成交量 > 5万（确保流动性）")
        print("   - 股价 > 3元（规避低价股风险）")

        print("\n2. 综合得分权重：")
        print("   - 上涨概率：30%（机器学习模型预测结果）")
        print("   - 动量得分：20%（价格走势趋势）")
        print("   - 爆发潜力：35%（短线上涨概率）")
        print("   - 波动率：15%（风险控制指标）")

        print("\n3. 特别说明：")
        print("   - 所有技术指标均经过标准化处理，消除量纲影响")
        print("   - 波动率采用对数收益率计算，更准确反映风险")
        print("   - 异常值已通过中位数法进行处理")
        print("   - 模型每日自动更新，适应市场变化")

        print("\n4. 风险提示：")
        print("   - 本程序基于历史数据和技术分析建模，不构成投资建议")
        print("   - 任何投资决策请结合市场环境、政策因素和个人风险承受能力")
        print("   - 模型预测结果仅供参考，交易需自行承担风险")
        print("   - 股市有风险，投资需谨慎")
    else: # This 'else' corresponds to 'if not selected_stocks_df.empty:'
        if not args.json_output:
            print("\n未找到符合条件的标的")
        else:
            # Output empty JSON array if no stocks are found and JSON output is requested
            print(json.dumps([])) 
            sys.exit(0)
