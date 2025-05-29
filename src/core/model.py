import pandas as pd
import numpy as np # Often used with pandas and sklearn
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_ml_model(df: pd.DataFrame):
    """
    Trains a machine learning model to predict stock price movements.

    Args:
        df: DataFrame containing stock data with features and a 'close' column.
            Required features: 'momentum_1m', 'momentum_3m', 'momentum_6m',
                               'volatility', 'vol_ratio', 'rsi', 'macd', 'signal'.
            The 'close' column is used to generate the target variable.

    Returns:
        A tuple (model, scaler) containing the trained LightGBM model
        and the StandardScaler instance used for feature scaling.
        Returns (None, None) if training is not possible (e.g., insufficient data).
    """
    # Prepare features
    features = [
        # Existing technical indicators
        'momentum_1m', 'momentum_3m', 'momentum_6m',
        'volatility', 'vol_ratio', 'rsi', 'macd', 'signal',
        # New fundamental indicators from daily data (stock_a_indicator_lg)
        'pe', 'pb', 'ps', 'dv_ratio', 'total_mv',
        # New annual financial indicators (stock_financial_analysis_indicator, merged via merge_asof)
        'eps', 'bvps', 'roe', 'debt_to_asset_ratio', 
        'gross_profit_margin', 'net_profit_margin'
    ]

    # Ensure all required features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features for model training: {missing_features}. Skipping training.")
        return None, None
        
    # Create target variable (5日收益率)
    if 'close' not in df.columns:
        print("Warning: 'close' column missing, cannot create target variable. Skipping training.")
        return None, None
        
    df = df.copy() # Avoid SettingWithCopyWarning
    df['target'] = df['close'].shift(-5) / df['close'] - 1

    # Remove rows with NaN in target or features (especially important after shift)
    df = df.dropna(subset=features + ['target'])

    if len(df) < 100:  # Data太少，不足以训练
        print(f"Warning: Insufficient data for training ({len(df)} rows). Need at least 100. Skipping training.")
        return None, None

    # 准备训练数据
    X = df[features]
    y = (df['target'] > df['target'].mean()).astype(int)  # 二分类问题

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42 # Using shuffle=True by default
    )

    # 训练LightGBM模型，添加参数来避免警告
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,  # 增加树的数量
        num_leaves=31,  # 控制树的复杂度
        min_child_samples=5,  # 降低最小叶子节点样本数
        min_split_gain=0.0,  # 降低分裂增益阈值
        max_depth=5,  # 限制树的深度
        learning_rate=0.1,  # 添加学习率
        verbose=-1  # 使用 verbose 替代 silent
    )
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None

    return model, scaler

if __name__ == '__main__':
    # Example Usage (optional, for testing)
    # Create a sample DataFrame similar to what train_ml_model expects
    num_rows = 200
    data = {
        'date': pd.to_datetime(['2023-01-01'] * num_rows) + pd.to_timedelta(np.arange(num_rows), 'D'),
        'close': np.random.rand(num_rows) * 100 + 50, 
        # Technical Indicators
        'momentum_1m': np.random.rand(num_rows) * 0.1 - 0.05,
        'momentum_3m': np.random.rand(num_rows) * 0.2 - 0.1,
        'momentum_6m': np.random.rand(num_rows) * 0.3 - 0.15,
        'volatility': np.random.rand(num_rows) * 0.5 + 0.1, 
        'vol_ratio': np.random.rand(num_rows) * 2 + 0.5, 
        'rsi': np.random.rand(num_rows) * 70 + 15, 
        'macd': np.random.rand(num_rows) * 2 - 1,
        'signal': np.random.rand(num_rows) * 2 - 1,
        # Fundamental Indicators (daily)
        'pe': np.random.rand(num_rows) * 30 + 5, # Example PE ratios
        'pb': np.random.rand(num_rows) * 3 + 0.5,  # Example PB ratios
        'ps': np.random.rand(num_rows) * 2 + 0.2,  # Example PS ratios
        'dv_ratio': np.random.rand(num_rows) * 5, # Example dividend yield
        'total_mv': np.random.rand(num_rows) * 100000 + 5000, # Example total market value
        # Annual Financial Indicators (these would be forward-filled in reality)
        'eps': np.random.rand(num_rows) * 2 + 0.5, # Example EPS
        'bvps': np.random.rand(num_rows) * 10 + 1, # Example BVPS
        'roe': np.random.rand(num_rows) * 20 + 5,  # Example ROE
        'debt_to_asset_ratio': np.random.rand(num_rows) * 60 + 20, # Example debt-to-asset
        'gross_profit_margin': np.random.rand(num_rows) * 50 + 10, # Example gross profit margin
        'net_profit_margin': np.random.rand(num_rows) * 15 + 1,   # Example net profit margin
    }
    sample_df = pd.DataFrame(data)
    sample_df = sample_df.sort_values(by='date').reset_index(drop=True)

    print("Sample DataFrame head:")
    print(sample_df.head())

    # Test train_ml_model
    print("\nTraining model with sample data...")
    model, scaler = train_ml_model(sample_df.copy())

    if model and scaler:
        print("\nModel training successful.")
        print(f"Model: {model}")
        print(f"Scaler mean: {scaler.mean_}")

        # Example of how to use the model for prediction (on the last row of the sample data)
        latest_data_sample = sample_df[model.feature_name_].iloc[-1:].copy() # Use feature_name_ from LGBM
        if not latest_data_sample.isnull().any().any():
            latest_scaled_sample = scaler.transform(latest_data_sample)
            prediction_proba = model.predict_proba(latest_scaled_sample)
            print(f"\nExample prediction probability for the last sample: {prediction_proba[0][1]:.4f}")
        else:
            print("\nCould not make a sample prediction due to NaNs in the latest data.")

    else:
        print("\nModel training failed or was skipped.")

    print("\nTesting with insufficient data (less than 100 rows):")
    insufficient_df = sample_df.head(50)
    model_insufficient, scaler_insufficient = train_ml_model(insufficient_df.copy())
    if not model_insufficient and not scaler_insufficient:
        print("Correctly skipped training for insufficient data.")
    else:
        print("Error: Training was not skipped for insufficient data.")

    print("\nTesting with missing feature:")
    missing_feature_df = sample_df.drop(columns=['momentum_1m'])
    model_missing, scaler_missing = train_ml_model(missing_feature_df.copy())
    if not model_missing and not scaler_missing:
        print("Correctly skipped training due to missing feature.")
    else:
        print("Error: Training was not skipped for missing feature df.")
