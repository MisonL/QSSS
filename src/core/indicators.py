import pandas as pd
import numpy as np

# Moved from QuantStrategy in main.py
def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """计算动量因子"""
    df = df.copy()
    df['momentum_1m'] = df['close'].pct_change(20)  # 1个月动量
    df['momentum_3m'] = df['close'].pct_change(60)  # 3个月动量
    df['momentum_6m'] = df['close'].pct_change(120)  # 6个月动量
    return df

# Moved from QuantStrategy in main.py
def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算多因子模型指标"""
    try:
        # 数据预处理：处理异常值
        df = df.copy()

        # 使用中位数填充极端值
        for col in ['close', 'volume', 'turn']:
            # Ensure column exists and is numeric before processing
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                std = df[col].std()
                # Ensure std is not zero to avoid issues with clip if all values are the same
                if std == 0:
                    # If std is 0, clipping is not meaningful, or all values are median.
                    # Consider logging or specific handling if necessary.
                    pass 
                else:
                    df[col] = df[col].clip(median - 3 * std, median + 3 * std)
            else:
                # Log or handle cases where expected columns are missing or not numeric
                print(f"Warning: Column '{col}' not found or not numeric in calculate_factors.")


        # 确保没有零值和负值 (Ensure these columns exist before trying to replace)
        if 'close' in df.columns:
            df['close'] = df['close'].replace(0, np.nan)
        if 'volume' in df.columns:
            df['volume'] = df['volume'].replace(0, np.nan)
        if 'turn' in df.columns:
            df['turn'] = df['turn'].replace(0, np.nan)
            
        # 计算技术指标 (Ensure 'close' and 'volume' columns exist)
        if 'close' in df.columns:
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=5).mean()

            # 计算波动率，使用对数收益率，添加数据验证
            returns = df['close'].pct_change()
            df['volatility'] = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)
            df['volatility'] = df['volatility'].fillna(df['volatility'].mean()).clip(0, 2)  # 限制在0-200%之间

            # 计算RSI，添加异常值处理
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)  # 避免除以零
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50).clip(0, 100)  # 使用50填充NaN，确保RSI在0-100之间

            # 计算MACD，使用更稳健的计算方法
            df['ema12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()

            # 填充MACD相关的NaN值
            df['macd'] = df['macd'].fillna(0)
            df['signal'] = df['signal'].fillna(0)

            # 计算布林带
            df['boll_mid'] = df['close'].rolling(window=20, min_periods=5).mean()
            df['boll_std'] = df['close'].rolling(window=20, min_periods=5).std()
            df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_down'] = df['boll_mid'] - 2 * df['boll_std']
        else:
            print("Warning: 'close' column not found for factor calculation.")


        if 'volume' in df.columns:
             # 计算成交量比率，添加数据验证
            volume_ma = df['volume'].rolling(window=20, min_periods=5).mean()
            df['vol_ratio'] = df['volume'] / volume_ma.replace(0, np.nan)
            df['vol_ratio'] = df['vol_ratio'].fillna(1.0).clip(0, 10)  # 使用1.0填充NaN，并限制范围
        else:
            print("Warning: 'volume' column not found for factor calculation.")

        # 使用前向填充处理剩余的NaN值
        df = df.fillna(method='ffill')
        # 使用后向填充处理开始部分的NaN值
        df = df.fillna(method='bfill')

        return df

    except Exception as e:
        print(f"计算技术指标时出错: {e}")
        # Return original df if error occurs to allow further processing if possible
        return df

# Moved from QuantStrategy in main.py (originally calculate_ma15 method)
# This function now expects the DataFrame to be fetched by the caller (e.g. DataFetcher)
def calculate_maN(df: pd.DataFrame, window: int = 15) -> float | None:
    """计算N日均线"""
    if df is not None and not df.empty and 'close' in df.columns:
        if len(df) >= window:
            return df['close'].rolling(window=window).mean().iloc[-1]
        else:
            # Not enough data for the window, could return mean of available data or None
            # print(f"Warning: Not enough data for MA{window}, returning None.")
            return None 
    return None

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    # Create a sample DataFrame
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                                '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20']),
        'open': np.random.rand(20) * 100,
        'high': np.random.rand(20) * 100 + 5,
        'low': np.random.rand(20) * 100 - 5,
        'close': np.random.rand(20) * 100,
        'volume': np.random.randint(10000, 100000, size=20),
        'turn': np.random.rand(20) * 5
    }
    sample_df = pd.DataFrame(data)
    sample_df = sample_df.sort_values(by='date').reset_index(drop=True)

    print("Original DataFrame head:")
    print(sample_df.head())

    # Test calculate_momentum
    df_with_momentum = calculate_momentum(sample_df.copy())
    print("\nDataFrame with Momentum head:")
    print(df_with_momentum[['date', 'close', 'momentum_1m', 'momentum_3m', 'momentum_6m']].head())

    # Test calculate_factors
    df_with_factors = calculate_factors(sample_df.copy())
    print("\nDataFrame with Factors head (selected columns):")
    print(df_with_factors[['date', 'close', 'ma5', 'ma20', 'vol_ratio', 'volatility', 'rsi', 'macd', 'signal']].head())
    
    # Test calculate_maN
    ma15 = calculate_maN(sample_df.copy(), window=15)
    print(f"\nMA15: {ma15}")
    
    ma5 = calculate_maN(sample_df.copy(), window=5)
    print(f"MA5: {ma5}")

    ma30 = calculate_maN(sample_df.copy(), window=30) # Should be None or trigger warning
    print(f"MA30: {ma30}")

    empty_df = pd.DataFrame()
    ma_empty = calculate_maN(empty_df)
    print(f"MA on empty df: {ma_empty}")

    minimal_df = pd.DataFrame({'close': [10,11,12,13,14]})
    ma_minimal = calculate_maN(minimal_df, window=5)
    print(f"MA5 on minimal df: {ma_minimal}")

    ma_minimal_not_enough = calculate_maN(minimal_df, window=10)
    print(f"MA10 on minimal df (not enough data): {ma_minimal_not_enough}")
