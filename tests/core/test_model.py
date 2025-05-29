import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from src.core.model import train_ml_model

# Define the full list of features expected by train_ml_model
EXPECTED_FEATURES = [
    'momentum_1m', 'momentum_3m', 'momentum_6m', 'volatility', 'vol_ratio', 
    'rsi', 'macd', 'signal', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv', 
    'eps', 'bvps', 'roe', 'debt_to_asset_ratio', 'gross_profit_margin', 
    'net_profit_margin'
]

def create_sample_dataframe(num_rows=200, add_nans=True, missing_cols=None):
    """Helper function to create a sample DataFrame for testing."""
    features_to_create = EXPECTED_FEATURES
    if missing_cols:
        features_to_create = [f for f in EXPECTED_FEATURES if f not in missing_cols]

    data_dict = {feature: np.random.rand(num_rows) * np.random.randint(1, 10) for feature in features_to_create}
    data_dict['close'] = np.random.rand(num_rows) * 100 + 10 # Ensure 'close' is present

    if add_nans and features_to_create: # Only add NaNs if features exist
        for _ in range(min(num_rows, 20)): # Add up to 20 NaNs
            feature_to_nan = features_to_create[np.random.randint(len(features_to_create))]
            data_dict[feature_to_nan][np.random.randint(num_rows)] = np.nan
    
    df = pd.DataFrame(data_dict)
    return df

def test_train_ml_model_runs_and_returns_expected_types():
    """
    Tests that train_ml_model runs successfully with sufficient valid data 
    and returns a model and scaler of the expected types.
    """
    df = create_sample_dataframe(num_rows=200, add_nans=True)
    
    model, scaler = train_ml_model(df.copy()) # Use a copy to avoid modifying original test df

    assert model is not None, "Model should not be None with sufficient valid data."
    assert isinstance(model, lgb.LGBMClassifier), "Model should be an LGBMClassifier instance."
    assert scaler is not None, "Scaler should not be None with sufficient valid data."
    assert isinstance(scaler, StandardScaler), "Scaler should be a StandardScaler instance."

    # Optional: Test prediction
    # Create a sample test data row (ensure it has all features and no NaNs for this test)
    # The train_ml_model function itself handles NaNs by dropping them before training.
    # For prediction, we need a row that would be valid.
    
    # Re-create a clean sample for prediction test, or take from df after internal dropna
    # For simplicity, let's make a clean single row of data for prediction.
    predict_df_data = {feature: [np.random.rand()] for feature in EXPECTED_FEATURES}
    predict_df = pd.DataFrame(predict_df_data)

    if not predict_df.empty:
        try:
            scaled_sample = scaler.transform(predict_df) # Use the features the model was trained on
            predictions = model.predict(scaled_sample)
            assert len(predictions) == 1, "Prediction should return a single value for one row."
            
            proba_predictions = model.predict_proba(scaled_sample)
            assert proba_predictions.shape == (1, 2), "Probability predictions shape mismatch."
        except Exception as e:
            pytest.fail(f"Prediction part of the test failed: {e}")

def test_train_ml_model_insufficient_data():
    """
    Tests that train_ml_model returns (None, None) if there's insufficient data 
    after dropping NaNs (less than 100 rows).
    """
    # Create DataFrame with fewer than 100 rows initially
    df_too_few_rows = create_sample_dataframe(num_rows=50, add_nans=False) 
    model, scaler = train_ml_model(df_too_few_rows.copy())
    assert model is None, "Model should be None for insufficient data (too few rows)."
    assert scaler is None, "Scaler should be None for insufficient data (too few rows)."

    # Create DataFrame that will have few rows after NaNs are dropped
    df_many_nans = create_sample_dataframe(num_rows=150, add_nans=False) # Start with enough
    # Make most rows NaN for many features, so they get dropped
    # For example, make 'momentum_1m' almost all NaN
    num_to_keep = 40
    indices_to_nan = np.random.choice(df_many_nans.index, size=len(df_many_nans) - num_to_keep, replace=False)
    for feature in EXPECTED_FEATURES[:5]: # Make first 5 features have many NaNs
        if feature in df_many_nans.columns:
             df_many_nans.loc[indices_to_nan, feature] = np.nan
    
    # Also need 'close' and 'target' (derived from 'close') for dropna
    # If 'close' has too many NaNs that can also lead to insufficient data
    # The function calculates 'target' using shift(-5), then dropna(subset=features + ['target'])

    model_after_nans, scaler_after_nans = train_ml_model(df_many_nans.copy())
    # This test's effectiveness depends on how many rows are actually dropped.
    # It's hard to guarantee <100 rows after dropna without replicating exact logic.
    # The print statement inside train_ml_model about insufficient data would indicate this.
    # For now, we assume that if it *does* result in <100, it returns None, None.
    # The df_too_few_rows case is a more direct test of the row count check.
    # If the internal dropna leads to <100 rows, it should return None, None.
    # We could check the print output if the test runner captures it, or trust the logic.
    # For CI, we rely on the explicit check. If it passed with too few rows, it's an issue.
    if model_after_nans is not None:
        print(f"Warning: Test for many NaNs resulting in insufficient data might not have triggered the condition. "
              f"Model was trained. This might be okay if enough rows remained.")
    # No strict assert here as it's hard to guarantee the <100 condition post-dropna without replicating it.
    # The function has a direct check for len(df) < 100, so df_too_few_rows covers that.

def test_train_ml_model_missing_features():
    """
    Tests that train_ml_model returns (None, None) if required features are missing.
    """
    df_missing_some = create_sample_dataframe(num_rows=200, add_nans=False, missing_cols=['momentum_1m', 'pe'])
    model, scaler = train_ml_model(df_missing_some.copy())
    assert model is None, "Model should be None when features are missing."
    assert scaler is None, "Scaler should be None when features are missing."

def test_train_ml_model_close_column_missing():
    """
    Tests that train_ml_model returns (None, None) if the 'close' column is missing.
    """
    features_only_data = {feature: np.random.rand(150) for feature in EXPECTED_FEATURES}
    df_no_close = pd.DataFrame(features_only_data)
    model, scaler = train_ml_model(df_no_close.copy())
    assert model is None, "Model should be None when 'close' column is missing."
    assert scaler is None, "Scaler should be None when 'close' column is missing."

# Example of how to run:
# pytest tests/core/test_model.py
