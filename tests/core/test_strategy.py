import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.core.strategy import QuantStrategy
# Assuming DataFetcher is imported in strategy.py for patching, or we might need to patch it where it's used.
# from src.core.data_fetcher import DataFetcher # Only if needed for type hinting or direct mock target

@pytest.fixture
def strategy_instance(silent_mode=False):
    """
    Provides a QuantStrategy instance.
    Mocks DataFetcher instantiation within QuantStrategy to avoid actual DataFetcher calls
    unless a specific method of DataFetcher is being tested (which would be mocked separately).
    """
    with patch('src.core.strategy.DataFetcher') as MockDataFetcher:
        mock_fetcher_instance = MockDataFetcher.return_value
        # Configure mock_fetcher_instance if needed for some tests, e.g.
        # mock_fetcher_instance.get_stock_list.return_value = pd.DataFrame(...)
        
        # Mock the cleanup method that gets registered with atexit
        mock_fetcher_instance.cleanup = MagicMock()

        strategy = QuantStrategy(silent_mode=silent_mode)
        strategy.data_fetcher = mock_fetcher_instance # Ensure it uses our mock
        return strategy

# === Tests for calculate_ma_for_stocks ===
# Note: The method in QuantStrategy is calculate_ma_for_stocks, not calculate_ma15.
# It calculates MA for a list of stocks provided in a DataFrame.
# The actual MA calculation for a single stock's daily data is in indicators.calculate_maN.
# So, we'll test calculate_ma_for_stocks by mocking get_stock_daily_data.

def test_calculate_ma_for_stocks_basic(strategy_instance):
    mock_daily_data_success = pd.DataFrame({
        'close': np.arange(1, 21) # 20 days of data
    })
    # Mock the data_fetcher's method that calculate_ma_for_stocks uses
    strategy_instance.data_fetcher.get_stock_daily_data = MagicMock(return_value=mock_daily_data_success)
    
    input_stocks_df = pd.DataFrame({'symbol': ['SH600000', 'SZ000001']})
    result_df = strategy_instance.calculate_ma_for_stocks(input_stocks_df.copy(), window=15)

    assert f'ma15' in result_df.columns
    assert len(result_df) == 2
    # Expected MA15 for np.arange(1, 21) is mean of 6 to 20 = (6+20)/2 = 13
    assert result_df[f'ma15'].iloc[0] == pytest.approx(13.0)
    assert result_df[f'ma15'].iloc[1] == pytest.approx(13.0)
    strategy_instance.data_fetcher.get_stock_daily_data.assert_any_call('SH600000')
    strategy_instance.data_fetcher.get_stock_daily_data.assert_any_call('SZ000001')

def test_calculate_ma_for_stocks_insufficient_data(strategy_instance):
    mock_daily_data_insufficient = pd.DataFrame({
        'close': np.arange(1, 10) # 9 days of data, less than 15
    })
    strategy_instance.data_fetcher.get_stock_daily_data = MagicMock(return_value=mock_daily_data_insufficient)
    
    input_stocks_df = pd.DataFrame({'symbol': ['SH600000']})
    result_df = strategy_instance.calculate_ma_for_stocks(input_stocks_df.copy(), window=15)
    
    assert pd.isna(result_df[f'ma15'].iloc[0]) # indicators.calculate_maN returns None -> np.nan

def test_calculate_ma_for_stocks_nans_in_close(strategy_instance):
    close_data = np.arange(1.0, 21.0)
    close_data[5:10] = np.nan # Add some NaNs
    mock_daily_data_with_nans = pd.DataFrame({'close': close_data})
    strategy_instance.data_fetcher.get_stock_daily_data = MagicMock(return_value=mock_daily_data_with_nans)

    input_stocks_df = pd.DataFrame({'symbol': ['SH600000']})
    result_df = strategy_instance.calculate_ma_for_stocks(input_stocks_df.copy(), window=15)
    
    # pandas rolling.mean() skips NaNs by default.
    # For window 15 on `close_data` ending at index 19:
    # Data from index 5 to 19. NaNs are from index 5 to 9.
    # Valid data points in window: close_data[0:5] (5 points) and close_data[10:20] (10 points)
    # The last 15 points are close_data[5:20]. 5 NaNs, 10 valid.
    # Valid points: 1,2,3,4,5,  11,12,13,14,15,16,17,18,19,20. sum = 145. count = 10. mean = 14.5
    # No, the window is fixed. The values are [NaN,NaN,NaN,NaN,NaN, 11,12,13,14,15,16,17,18,19,20]
    # Mean of (11..20) = (11+20)/2 = 15.5
    expected_ma = np.nanmean(close_data[5:20]) # This is how pandas rolling might do it
    assert result_df[f'ma15'].iloc[0] == pytest.approx(expected_ma)


def test_calculate_ma_for_stocks_fetcher_returns_empty(strategy_instance):
    strategy_instance.data_fetcher.get_stock_daily_data = MagicMock(return_value=pd.DataFrame()) # Empty df
    input_stocks_df = pd.DataFrame({'symbol': ['SH600000']})
    result_df = strategy_instance.calculate_ma_for_stocks(input_stocks_df.copy(), window=15)
    assert pd.isna(result_df[f'ma15'].iloc[0])

def test_calculate_ma_for_stocks_empty_input_df(strategy_instance):
    input_stocks_df = pd.DataFrame({'symbol': []}) # Empty input
    result_df = strategy_instance.calculate_ma_for_stocks(input_stocks_df.copy(), window=15)
    assert result_df.empty or f'ma15' not in result_df.columns or len(result_df[f'ma15']) == 0


# === Tests for analyze_short_term_explosion ===
def create_df_for_explosion_test(last_close=10, last_vol=100000, last_turn=2.0, 
                                 last_macd=0.1, last_signal=0.05, prev_macd=0.0, prev_signal=0.06,
                                 recent_prices=None, recent_volumes=None, recent_turns=None,
                                 num_rows=20):
    if num_rows < 20: # Method expects at least 20 rows for some calcs
        return pd.DataFrame()

    data = {'date': pd.date_range(end=pd.Timestamp.now(), periods=num_rows, freq='B')}
    
    base_prices = np.linspace(last_close - 5, last_close, num_rows) if recent_prices is None else recent_prices
    data['close'] = base_prices
    data['close'].iloc[-1] = last_close
    
    base_volumes = np.linspace(last_vol / 2, last_vol, num_rows) if recent_volumes is None else recent_volumes
    data['volume'] = base_volumes
    data['volume'].iloc[-1] = last_vol

    base_turns = np.linspace(last_turn / 2, last_turn, num_rows) if recent_turns is None else recent_turns
    data['turn'] = base_turns
    data['turn'].iloc[-1] = last_turn
    
    macds = np.linspace(prev_macd, last_macd, num_rows)
    signals = np.linspace(prev_signal, last_signal, num_rows)
    data['macd'] = macds
    data['signal'] = signals
    data['macd'].iloc[-1] = last_macd; data['macd'].iloc[-2] = prev_macd
    data['signal'].iloc[-1] = last_signal; data['signal'].iloc[-2] = prev_signal
    
    return pd.DataFrame(data)

def test_analyze_short_term_explosion_high_score(strategy_instance):
    # Conditions for high score:
    # - High volume_ratio (current vol >> recent mean vol)
    # - High turnover_ratio (current turn >> recent mean turn)
    # - MACD golden cross (last_macd > last_signal AND prev_macd < prev_signal)
    # - Price position low (current price near recent min)
    # - High recent_momentum (current price > price 3 days ago)
    
    prices = np.array([10,9,8,9,10,11,10,9.5,9,8.5, 8,8.2,8.1,8.3,8.5, 9,10,11,12,10.0]) # Ends low after a peak
    volumes = np.array([50000]*19 + [200000]) # Last volume spike
    turns = np.array([1.0]*19 + [5.0])       # Last turnover spike
    
    df = create_df_for_explosion_test(
        last_close=10.0, last_vol=200000, last_turn=5.0,
        last_macd=0.1, last_signal=0.0, prev_macd=-0.05, prev_signal=0.02, # Golden cross
        recent_prices=prices, recent_volumes=volumes, recent_turns=turns
    )
    df['close'].iloc[-3] = 9.0 # for recent_momentum: (10.0 / 9.0) - 1 = 0.111

    score = strategy_instance.analyze_short_term_explosion(df)
    # Manual calculation (approximate):
    # vol_ratio = 200000 / mean(volumes[-20:]) approx 200000 / ( (19*50000+200000)/20 ) = 200000 / 57500 ~ 3.47 -> clipped 3.47
    # turn_ratio = 5.0 / mean(turns[-20:]) approx 5.0 / ( (19*1.0+5.0)/20 ) = 5.0 / 1.2 ~ 4.16 -> clipped 4.16
    # macd_cross_score = 1
    # price_position: min=8, max=12, last=10. (10-8)/(12-8) = 2/4 = 0.5. (1 - price_position) = 0.5
    # recent_momentum = (10/9)-1 = 0.1111. (0.1111*100+20)/40 = (11.11+20)/40 = 31.11/40 ~ 0.77
    # score = 3.47*0.3 + 4.16*0.2 + 1*0.2 + 0.5*0.15 + 0.77*0.15
    #       = 1.041 + 0.832 + 0.2 + 0.075 + 0.1155 = ~2.26
    assert score > 2.0 # Expecting a relatively high score

def test_analyze_short_term_explosion_low_score(strategy_instance):
    prices = np.array([10,11,12,11,10,9,10,10.5,11,11.5, 12,11.8,11.9,11.7,11.5, 11,10,9,8,10.0])
    volumes = np.array([100000]*19 + [50000]) # Last volume drop
    turns = np.array([2.0]*19 + [0.5])       # Last turnover drop
    
    df = create_df_for_explosion_test(
        last_close=10.0, last_vol=50000, last_turn=0.5,
        last_macd=-0.1, last_signal=0.0, prev_macd=0.05, prev_signal=0.02, # Death cross
        recent_prices=prices, recent_volumes=volumes, recent_turns=turns
    )
    df['close'].iloc[-3] = 11.0 # for recent_momentum: (10.0 / 11.0) - 1 = -0.09

    score = strategy_instance.analyze_short_term_explosion(df)
    assert score < 1.0 # Expecting a low score

def test_analyze_short_term_explosion_insufficient_data(strategy_instance):
    df = create_sample_df(num_rows=19) # Less than 20 rows
    score = strategy_instance.analyze_short_term_explosion(df)
    assert score == 0.0

def test_analyze_short_term_explosion_with_nans(strategy_instance):
    df = create_df_for_explosion_test()
    df.loc[df.index[-5:], 'close'] = np.nan # Introduce NaNs in critical lookback period
    score = strategy_instance.analyze_short_term_explosion(df)
    assert score == 0.0 # Should handle NaNs gracefully and return 0

# === Tests for get_optimal_thread_count ===
@patch('src.core.strategy.psutil') # Mock the entire psutil module used in strategy.py
def test_get_optimal_thread_count_high_load(mock_psutil, strategy_instance):
    mock_psutil.cpu_percent.return_value = 90  # High CPU
    mock_psutil.virtual_memory.return_value = MagicMock(percent=80, available=1024*1024*500) # High RAM
    mock_psutil.swap_memory.return_value = MagicMock(percent=70)    # High Swap
    
    strategy_instance.min_workers = 2
    strategy_instance.max_workers = 10
    
    thread_count = strategy_instance.get_optimal_thread_count()
    assert thread_count == strategy_instance.min_workers # Expect min_workers due to high load (system_load > 0.8)

@patch('src.core.strategy.psutil')
def test_get_optimal_thread_count_low_load(mock_psutil, strategy_instance):
    mock_psutil.cpu_percent.return_value = 10  # Low CPU
    mock_psutil.virtual_memory.return_value = MagicMock(percent=20, available=1024*1024*8000) # Low RAM
    mock_psutil.swap_memory.return_value = MagicMock(percent=5)     # Low Swap
    
    strategy_instance.min_workers = 2
    strategy_instance.max_workers = 16 # Set higher for more range
    
    # system_load = max(0.1, 0.2, 0.05) = 0.2
    # optimal_threads = int(max_workers * (0.9 - system_load * 0.5))
    # optimal_threads = int(16 * (0.9 - 0.2 * 0.5)) = int(16 * (0.9 - 0.1)) = int(16 * 0.8) = 12.8 -> 12
    expected_threads = int(strategy_instance.max_workers * (0.9 - 0.2 * 0.5))
    expected_threads = min(max(expected_threads, strategy_instance.min_workers), strategy_instance.max_workers)

    thread_count = strategy_instance.get_optimal_thread_count()
    assert thread_count == expected_threads

@patch('src.core.strategy.psutil')
def test_get_optimal_thread_count_medium_load(mock_psutil, strategy_instance):
    mock_psutil.cpu_percent.return_value = 70  # Medium-High CPU
    mock_psutil.virtual_memory.return_value = MagicMock(percent=65, available=1024*1024*2000) # Medium RAM
    mock_psutil.swap_memory.return_value = MagicMock(percent=30)    # Medium Swap
    
    strategy_instance.min_workers = 4
    strategy_instance.max_workers = 10
    
    # system_load = max(0.7, 0.65, 0.30) = 0.7
    # optimal_threads = int(min_workers + (max_workers - min_workers) * (0.8 - system_load) / 0.2)
    # optimal_threads = int(4 + (10 - 4) * (0.8 - 0.7) / 0.2)
    # optimal_threads = int(4 + 6 * 0.1 / 0.2) = int(4 + 6 * 0.5) = int(4 + 3) = 7
    expected_threads = int(strategy_instance.min_workers + \
        (strategy_instance.max_workers - strategy_instance.min_workers) * \
        (0.8 - 0.7) / 0.2)
    expected_threads = min(max(expected_threads, strategy_instance.min_workers), strategy_instance.max_workers)
        
    thread_count = strategy_instance.get_optimal_thread_count()
    assert thread_count == expected_threads

@patch('src.core.strategy.psutil')
def test_get_optimal_thread_count_psutil_exception(mock_psutil, strategy_instance):
    mock_psutil.cpu_percent.side_effect = Exception("psutil error")
    strategy_instance.min_workers = 3
    
    thread_count = strategy_instance.get_optimal_thread_count()
    assert thread_count == strategy_instance.min_workers # Should default to min_workers on error
