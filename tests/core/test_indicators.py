import pytest
import pandas as pd
import numpy as np
from src.core.indicators import calculate_momentum, calculate_factors, calculate_maN

# Helper function to create sample DataFrames
def create_sample_df(num_rows=150, start_price=100.0, price_increment=0.5, vol_base=10000, turn_base=1.0):
    dates = pd.to_datetime(['2023-01-01'] * num_rows) + pd.to_timedelta(np.arange(num_rows), 'D')
    data = {
        'date': dates,
        'close': start_price + np.arange(num_rows) * price_increment + np.random.randn(num_rows) * 0.1, # Add some noise
        'volume': vol_base + np.arange(num_rows) * 100 + np.random.randint(-500, 500, size=num_rows),
        'turn': turn_base + np.arange(num_rows) * 0.01 + np.random.rand(num_rows) * 0.1,
        'open': start_price + np.arange(num_rows) * price_increment - 0.2 + np.random.randn(num_rows) * 0.1,
        'high': start_price + np.arange(num_rows) * price_increment + 0.3 + np.random.randn(num_rows) * 0.1,
        'low': start_price + np.arange(num_rows) * price_increment - 0.3 + np.random.randn(num_rows) * 0.1,
    }
    df = pd.DataFrame(data)
    # Ensure no negative prices or volumes if noise is too high, though unlikely with these bases
    df['close'] = df['close'].clip(lower=0.01)
    df['open'] = df['open'].clip(lower=0.01)
    df['high'] = df['high'].clip(lower=0.01)
    df['low'] = df['low'].clip(lower=0.01)
    df['volume'] = df['volume'].clip(lower=1)
    return df

# --- Tests for calculate_maN ---
def test_calculate_maN_basic():
    df = pd.DataFrame({'close': [10, 11, 12, 13, 14, 15]})
    ma5 = calculate_maN(df.copy(), window=5)
    assert ma5 == pytest.approx(13.0) # (11+12+13+14+15)/5 = 13 for the last point if window includes it
    
    # Test with a different window
    ma3 = calculate_maN(df.copy(), window=3)
    assert ma3 == pytest.approx(14.0) # (13+14+15)/3

def test_calculate_maN_not_enough_data():
    df = pd.DataFrame({'close': [10, 11, 12]})
    ma5 = calculate_maN(df.copy(), window=5)
    assert ma5 is None # Not enough data for MA5

def test_calculate_maN_empty_df():
    df = pd.DataFrame({'close': []})
    ma5 = calculate_maN(df.copy(), window=5)
    assert ma5 is None

def test_calculate_maN_df_without_close_col():
    df = pd.DataFrame({'other_col': [10, 11, 12]})
    ma5 = calculate_maN(df.copy(), window=5)
    assert ma5 is None

def test_calculate_maN_with_nans_in_close():
    df = pd.DataFrame({'close': [10, 11, np.nan, 13, 14, 15]})
    # Rolling mean should handle NaNs by skipping them if possible, or result in NaN
    # The current implementation of calculate_maN doesn't explicitly handle NaNs before rolling.
    # Pandas rolling().mean() itself skips NaNs by default.
    # For a window of 3 on [13, 14, 15], result is 14.
    # For a window of 3 on [np.nan, 13, 14], result is (13+14)/2 = 13.5, but we take iloc[-1]
    # If the last values are [13, 14, 15], MA3 is 14.
    ma3 = calculate_maN(df.copy(), window=3)
    assert ma3 == pytest.approx(14.0)

    df_nan_at_end = pd.DataFrame({'close': [10, 11, 12, 13, np.nan, np.nan]})
    ma3_nan_end = calculate_maN(df_nan_at_end.copy(), window=3)
    # Last 3 are [13, nan, nan]. If MA uses only last 3, it should be NaN.
    # If it uses [12,13,nan] -> (12+13)/2 = 12.5. Current takes iloc[-1] of rolling
    # The rolling window ending at the last NaN will be NaN.
    assert pd.isna(ma3_nan_end)


# --- Tests for calculate_momentum ---
def test_calculate_momentum_adds_columns():
    df = create_sample_df(num_rows=150) # Needs enough rows for all momentums
    df_out = calculate_momentum(df.copy())
    assert 'momentum_1m' in df_out.columns
    assert 'momentum_3m' in df_out.columns
    assert 'momentum_6m' in df_out.columns
    assert len(df_out) == len(df)

def test_calculate_momentum_known_values():
    # 20 days for 1m, 60 for 3m, 120 for 6m
    close_prices = list(range(101, 121)) # 20 values, ends at 120
    df = pd.DataFrame({'close': close_prices})
    df_out = calculate_momentum(df.copy())
    # momentum_1m = (P_t / P_{t-20}) - 1. P_{t-20} is the first element (index 0)
    # P_t is the last element (index 19)
    expected_m1m = (120.0 / 101.0) - 1
    assert df_out['momentum_1m'].iloc[-1] == pytest.approx(expected_m1m)

    # Test with insufficient data for some momentums
    df_short = pd.DataFrame({'close': close_prices[:10]}) # 10 data points
    df_short_out = calculate_momentum(df_short.copy())
    assert pd.isna(df_short_out['momentum_1m'].iloc[-1]) # Not enough data for 20-day lag

    df_medium = pd.DataFrame({'close': close_prices * 2}) # 40 data points
    df_medium_out = calculate_momentum(df_medium.copy())
    assert pd.notna(df_medium_out['momentum_1m'].iloc[-1])
    assert pd.isna(df_medium_out['momentum_3m'].iloc[-1]) # Not enough for 60-day lag

def test_calculate_momentum_with_nans_in_close():
    close_prices = list(range(101, 121))
    close_prices[10] = np.nan # A NaN in the middle
    df = pd.DataFrame({'close': close_prices})
    df_out = calculate_momentum(df.copy())
    # pct_change will propagate NaNs. If a value used in calculation is NaN, result is NaN.
    # The last value is not NaN, but a value 20 periods ago might be.
    # Here, P_t-20 (close_prices[0]) is not NaN. So m1m should be calculable.
    if pd.isna(close_prices[0]): # Should not be the case here
         assert pd.isna(df_out['momentum_1m'].iloc[-1])
    else:
         assert pd.notna(df_out['momentum_1m'].iloc[-1])

    close_prices_nan_start = list(range(101, 121))
    close_prices_nan_start[0] = np.nan
    df_nan_start = pd.DataFrame({'close': close_prices_nan_start})
    df_nan_start_out = calculate_momentum(df_nan_start.copy())
    assert pd.isna(df_nan_start_out['momentum_1m'].iloc[-1]) # P_{t-20} is NaN


# --- Tests for calculate_factors ---
def test_calculate_factors_adds_columns():
    df = create_sample_df(num_rows=150)
    df_out = calculate_factors(df.copy())
    expected_cols = ['ma5', 'ma20', 'vol_ratio', 'volatility', 'rsi', 'macd', 'signal', 
                     'boll_mid', 'boll_up', 'boll_down', 'ema12', 'ema26']
    for col in expected_cols:
        assert col in df_out.columns
    assert len(df_out) == len(df)

def test_calculate_factors_rsi_fill_and_clip():
    # Create data that would lead to RSI 0 or 100 or NaN if not handled
    # RSI becomes NaN if loss is 0 for a period (all gains or no change)
    df_rsi_extreme = pd.DataFrame({
        'close': [10] * 30, # All same prices, gain/loss = 0, RSI -> NaN, should be filled to 50
        'volume': [1000] * 30,
        'turn': [1.0] * 30
    })
    df_out = calculate_factors(df_rsi_extreme.copy())
    assert df_out['rsi'].iloc[-1] == pytest.approx(50.0)

    # Test clipping (RSI already clipped 0-100 by formula, but check fill)
    # This test mainly checks fillna(50)
    all_up = np.arange(10, 40) # 30 data points, all up
    df_all_up = pd.DataFrame({'close': all_up, 'volume': [1000]*30, 'turn': [1.0]*30})
    df_out_up = calculate_factors(df_all_up.copy())
    # If all up, loss is 0, RS is inf, RSI is 100.
    assert df_out_up['rsi'].iloc[-1] == pytest.approx(100.0) 


def test_calculate_factors_volatility_clip():
    # Create data with extremely high volatility
    prices = [10, 1000, 10, 1000, 10, 1000] * 5 # 30 points
    df_high_vol = pd.DataFrame({'close': prices, 'volume': [1000]*30, 'turn': [1.0]*30})
    df_out = calculate_factors(df_high_vol.copy())
    # print(f"Volatility before clip: {df_out['volatility']}") # For debug
    assert df_out['volatility'].max() <= 2.0 # Clipped at 2 (200%)

def test_calculate_factors_vol_ratio_clip_and_fill():
    # Test fillna(1.0) for vol_ratio (e.g. if volume_ma is 0 or NaN initially)
    df_low_vol_ma = pd.DataFrame({
        'close': [10]*30,
        'volume': [0]*5 + [1000]*25, # Initial volumes are 0, so volume_ma might be 0
        'turn': [1.0]*30
    })
    df_out = calculate_factors(df_low_vol_ma.copy())
    # If volume_ma was 0, vol_ratio would be inf/NaN, then filled to 1.0
    # Check a point where it might have been filled
    # The first few points of rolling mean will be NaN.
    # rolling(window=20, min_periods=5).mean()
    # For row 4 (0-indexed), volume_ma is NaN. vol_ratio is NaN. Filled to 1.0
    assert df_out['vol_ratio'].iloc[4] == pytest.approx(1.0) 

    # Test clipping vol_ratio at 10
    df_high_vol_ratio = create_sample_df(num_rows=30)
    df_high_vol_ratio['volume'] = df_high_vol_ratio['volume'] * 100 # Make volume very high
    df_out_high = calculate_factors(df_high_vol_ratio.copy())
    # print(f"Vol ratio before clip: {df_out_high['vol_ratio']}") # For debug
    assert df_out_high['vol_ratio'].max() <= 10.0


def test_calculate_factors_handle_missing_cols():
    # Test with 'close' column missing
    df_no_close = pd.DataFrame({'volume': [1000]*30, 'turn': [1.0]*30})
    df_out_no_close = calculate_factors(df_no_close.copy())
    # Should return the df as is, possibly with warnings printed (not checked by test)
    # and no new indicator columns that depend on 'close'
    assert 'ma5' not in df_out_no_close.columns
    assert 'rsi' not in df_out_no_close.columns
    assert 'vol_ratio' not in df_out_no_close.columns # vol_ratio depends on volume but also factors df is copied

    # Test with 'volume' column missing
    df_no_volume = pd.DataFrame({'close': [10]*30, 'turn': [1.0]*30})
    df_out_no_vol = calculate_factors(df_no_volume.copy())
    assert 'ma5' in df_out_no_vol.columns # MA should be calculated
    assert 'vol_ratio' not in df_out_no_vol.columns

def test_calculate_factors_with_all_nans_input():
    df = create_sample_df(num_rows=30)
    for col in ['close', 'volume', 'turn']:
        df[col] = np.nan
    df_out = calculate_factors(df.copy())
    # Most indicators should be NaN or filled with their default (e.g. RSI 50, vol_ratio 1.0)
    # Check a few key ones
    assert pd.isna(df_out['ma5'].iloc[-1])
    assert df_out['rsi'].iloc[-1] == 50.0
    assert df_out['vol_ratio'].iloc[-1] == 1.0 # Filled with 1.0
    assert pd.isna(df_out['volatility'].iloc[-1]) # Filled by mean, but mean of NaNs is NaN. Then clipped.
                                                 # If all NaNs, mean is NaN. Clip(NaN) is still NaN.

def test_calculate_factors_extreme_value_clipping_close():
    prices = [1.0] * 20 + [1000.0] + [1.0] * 9 # 30 data points, one extreme spike
    df = pd.DataFrame({'close': prices, 'volume': [1000]*30, 'turn': [1.0]*30})
    
    # Calculate expected median and std for clipping 'close'
    # The function calculate_factors makes a copy, so original df is not modified
    temp_df_for_stats = df.copy()
    median_close = temp_df_for_stats['close'].median()
    std_close = temp_df_for_stats['close'].std()
    
    df_out = calculate_factors(df.copy()) # Pass a copy
    
    # The 'close' column itself in df_out is NOT modified by the clipping.
    # The clipping happens on a copy *within* calculate_factors for calculating indicators.
    # So, df_out['close'] should be the original prices.
    assert df_out['close'].iloc[20] == 1000.0 
    
    # However, indicators calculated from the clipped 'close' should be affected.
    # E.g. 'ma5' around the spike should be based on clipped value.
    # This is harder to test precisely without replicating the exact internal state.
    # A simpler check: if clipping worked, ma5 shouldn't be excessively large.
    # If 1000 was clipped to median + 3*std = 1 + 3*std_close, ma5 would be much smaller.
    # If std_close is large due to 1000, clipping might not be that aggressive.
    # std_close for [1... (20 times), 1000, 1... (9 times)] is approx 182.
    # median is 1. clip_max = 1 + 3 * 182 = 547.
    # So 1000 would be clipped to 547 for internal calculations.
    # MA5 including the clipped value: (1+1+1+1+547)/5 = 110.2 (if spike is last)
    # MA5 including original value: (1+1+1+1+1000)/5 = 200.8
    
    # Let's check the ma5 value at an index affected by the spike
    # Spike is at index 20. ma5 at index 20 uses close[16] to close[20]
    # Original: close[16-19] are 1.0, close[20] is 1000.0. ma5 = (1+1+1+1+1000)/5 = 200.8
    # Clipped: close[20] becomes approx 547. ma5_clipped = (1+1+1+1+547)/5 = 110.2
    # This assumes the df passed to rolling is the one with clipped values.
    
    # The current implementation of calculate_factors applies clipping to df_copy
    # then all calculations are on this df_copy.
    # So df_out['ma5'] should reflect the clipped calculation.
    assert df_out['ma5'].iloc[20] < 200.0 # Should be significantly less than if 1000 was used directly
    assert df_out['ma5'].iloc[20] > 50.0  # And greater than if it was just 1.0s
    # This is an indirect way to test clipping's effect.
    # A more direct test would involve checking the value of 'close' inside the function after clipping.

# More specific tests for MACD and Bollinger Bands can be added if needed,
# but they rely on MA/EMA which are indirectly tested.
# Example:
def test_calculate_factors_macd_signal_basic():
    df = create_sample_df(num_rows=50) # Need enough for MACD
    df_out = calculate_factors(df.copy())
    assert 'macd' in df_out.columns
    assert 'signal' in df_out.columns
    assert not df_out['macd'].isnull().all()
    assert not df_out['signal'].isnull().all()
    # Check if MACD is EMA12 - EMA26
    assert np.allclose(df_out['macd'], df_out['ema12'] - df_out['ema26'], equal_nan=True)

def test_calculate_factors_bollinger_basic():
    df = create_sample_df(num_rows=50)
    df_out = calculate_factors(df.copy())
    assert 'boll_mid' in df_out.columns
    assert 'boll_up' in df_out.columns
    assert 'boll_down' in df_out.columns
    assert np.allclose(df_out['boll_mid'], df_out['close'].rolling(window=20, min_periods=5).mean(), equal_nan=True)
    boll_std_expected = df_out['close'].rolling(window=20, min_periods=5).std()
    assert np.allclose(df_out['boll_up'], df_out['boll_mid'] + 2 * boll_std_expected, equal_nan=True)
    assert np.allclose(df_out['boll_down'], df_out['boll_mid'] - 2 * boll_std_expected, equal_nan=True)

# Test for the case where std is 0 in clipping (defensive)
def test_calculate_factors_zero_std_clipping():
    df = pd.DataFrame({
        'close': [10.0] * 30, # All same prices, std will be 0
        'volume': [1000.0] * 30,
        'turn': [1.0] * 30
    })
    # calculate_factors should run without error
    df_out = calculate_factors(df.copy())
    assert df_out['close'].iloc[-1] == 10.0 # Original close should be unchanged in output
    # MA calculations should be based on 10.0
    assert df_out['ma5'].iloc[-1] == 10.0
    assert df_out['ma20'].iloc[-1] == 10.0
    # Volatility should be 0 or very close to 0
    assert df_out['volatility'].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    # RSI should be 50 (due to no change)
    assert df_out['rsi'].iloc[-1] == 50.0

    # Verify other columns exist
    expected_cols = ['ma5', 'ma20', 'vol_ratio', 'volatility', 'rsi', 'macd', 'signal', 
                     'boll_mid', 'boll_up', 'boll_down']
    for col in expected_cols:
        assert col in df_out.columns
        assert not df_out[col].isnull().all() # Ensure they are not all NaNs
        if col not in ['rsi', 'vol_ratio']: # rsi is 50, vol_ratio is 1.0
             if col not in ['volatility', 'macd', 'signal']: # these can be 0
                assert not (df_out[col] == 0).all() # Ensure not all zeros for MA etc.

    assert df_out['vol_ratio'].iloc[-1] == 1.0 # Since volume is constant, vol_ma = volume, ratio is 1
    assert df_out['macd'].iloc[-1] == 0.0
    assert df_out['signal'].iloc[-1] == 0.0
    assert df_out['boll_mid'].iloc[-1] == 10.0
    assert df_out['boll_up'].iloc[-1] == 10.0 # Since std is 0
    assert df_out['boll_down'].iloc[-1] == 10.0 # Since std is 0

    
# To run tests: pytest tests/core/test_indicators.py
