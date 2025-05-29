import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Assuming DataFetcher is in src.core.data_fetcher
from src.core.data_fetcher import DataFetcher

# Define the expected columns for the output of get_financial_analysis_indicators
EXPECTED_OUTPUT_COLS = ['date', 'eps', 'bvps', 'roe', 'debt_to_asset_ratio', 
                        'gross_profit_margin', 'net_profit_margin']

@pytest.fixture
def data_fetcher_instance():
    """Pytest fixture to provide a DataFetcher instance."""
    # We can mock bs.login() and bs.logout() if they are called in __init__ or cleanup
    # and we don't want actual Baostock calls during these unit tests.
    with patch('baostock.login', return_value=True), \
         patch('baostock.logout', return_value=True):
        fetcher = DataFetcher()
    return fetcher

def create_mock_akshare_financial_df(data_dict):
    """
    Helper to create a DataFrame similar to what ak.stock_financial_analysis_indicator might return.
    '指标' is a column, other keys are date strings like '20221231'.
    """
    return pd.DataFrame(data_dict)

# --- Test Case 1: Successful Processing ---
def test_get_financial_indicators_successful_processing(data_fetcher_instance):
    raw_data = {
        '指标': ['每股收益', '每股净资产', '净资产收益率', '资产负债率', '销售毛利率', '销售净利率', '不相关指标', '其他指标'],
        '20221231': [1.0, 10.0, 0.10, 0.5, 0.3, 0.05, 999, 111],
        '20211231': [0.8, 9.0, 0.09, 0.45, 0.28, 0.04, 888, 222],
        '20201231': [0.7, 8.0, 0.08, 0.40, 0.25, 0.03, 777, 333],
    }
    mock_df = create_mock_akshare_financial_df(raw_data)

    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600000")

    assert not result_df.empty
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS
    assert len(result_df) == 3 # Three years of data

    # Check date conversion and sorting (should be descending as per akshare, then sorted ascending by function)
    # The function sorts the transposed df by index (dates) which are then reset.
    # So, the final 'date' column should be sorted ascending.
    assert pd.api.types.is_datetime64_any_dtype(result_df['date'])
    assert result_df['date'].is_monotonic_increasing

    # Check values for the latest year (2022-12-31)
    latest_year_data = result_df[result_df['date'] == pd.Timestamp('2022-12-31')]
    assert not latest_year_data.empty
    assert latest_year_data['eps'].iloc[0] == pytest.approx(1.0)
    assert latest_year_data['bvps'].iloc[0] == pytest.approx(10.0)
    assert latest_year_data['roe'].iloc[0] == pytest.approx(0.10)
    assert latest_year_data['debt_to_asset_ratio'].iloc[0] == pytest.approx(0.5)
    assert latest_year_data['gross_profit_margin'].iloc[0] == pytest.approx(0.3)
    assert latest_year_data['net_profit_margin'].iloc[0] == pytest.approx(0.05)

    # Check values for an older year (2020-12-31)
    older_year_data = result_df[result_df['date'] == pd.Timestamp('2020-12-31')]
    assert not older_year_data.empty
    assert older_year_data['eps'].iloc[0] == pytest.approx(0.7)


# --- Test Case 2: Missing some indicators in raw data ---
def test_get_financial_indicators_missing_indicators(data_fetcher_instance):
    raw_data = {
        '指标': ['每股收益', '资产负债率', '不相关指标'], # Missing bvps, roe, margins
        '20221231': [1.2, 0.55, 998],
        '20211231': [0.9, 0.48, 887],
    }
    mock_df = create_mock_akshare_financial_df(raw_data)

    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600001")
    
    assert not result_df.empty
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS
    assert len(result_df) == 2

    latest_year_data = result_df[result_df['date'] == pd.Timestamp('2022-12-31')]
    assert latest_year_data['eps'].iloc[0] == pytest.approx(1.2)
    assert latest_year_data['debt_to_asset_ratio'].iloc[0] == pytest.approx(0.55)
    
    # Check that missing indicators result in NaN columns
    assert pd.isna(latest_year_data['bvps'].iloc[0])
    assert pd.isna(latest_year_data['roe'].iloc[0])
    assert pd.isna(latest_year_data['gross_profit_margin'].iloc[0])
    assert pd.isna(latest_year_data['net_profit_margin'].iloc[0])

# --- Test Case 3: Empty raw data from akshare ---
def test_get_financial_indicators_empty_raw_data(data_fetcher_instance):
    mock_empty_df = pd.DataFrame() # Completely empty
    
    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_empty_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600002")
        
    assert result_df.empty # The function's logic for empty raw_financial_df
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS # Should still have expected columns

    mock_empty_with_indicator_col = pd.DataFrame({'指标': []}) # Empty but with '指标' column
    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_empty_with_indicator_col):
        result_df_2 = data_fetcher_instance.get_financial_analysis_indicators("SH600003")
    
    assert result_df_2.empty
    assert list(result_df_2.columns) == EXPECTED_OUTPUT_COLS


# --- Test Case 4: akshare call fails (simulated by exception) ---
def test_get_financial_indicators_akshare_exception(data_fetcher_instance):
    with patch('akshare.stock_financial_analysis_indicator', side_effect=Exception("AKShare API Error")):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600004")
        
    assert result_df.empty
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS

# --- Test Case 5: No valid year columns in raw data ---
def test_get_financial_indicators_no_year_columns(data_fetcher_instance):
    raw_data = {
        '指标': ['每股收益', '每股净资产'],
        'SomeOtherColumn': [1.0, 10.0], # No year-like columns
    }
    mock_df = create_mock_akshare_financial_df(raw_data)
    
    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600005")
        
    assert result_df.empty 
    # The current implementation might fail earlier if '指标' is not set as index correctly,
    # or if transpose fails. The goal is it should return an empty df with expected_cols.
    # After transpose, if no year columns, .index would be 'SomeOtherColumn'. 
    # pd.to_datetime will raise error or make NaT.
    # The function should catch this and return empty with expected_cols.
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS


# --- Test Case 6: Raw data has '指标' column but no actual indicator rows ---
def test_get_financial_indicators_no_indicator_rows(data_fetcher_instance):
    raw_data = {
        '指标': [], # No indicator names
        '20221231': [],
        '20211231': [],
    }
    mock_df = create_mock_akshare_financial_df(raw_data)
    
    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600006")
        
    assert result_df.empty
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS

# --- Test Case 7: Data with non-numeric values for indicators ---
def test_get_financial_indicators_non_numeric_values(data_fetcher_instance):
    raw_data = {
        '指标': ['每股收益', '每股净资产'],
        '20221231': ["invalid_eps", "10.5"], # Non-numeric EPS
        '20211231': ["0.8", "bad_bvps"],     # Non-numeric BVPS
    }
    mock_df = create_mock_akshare_financial_df(raw_data)

    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600007")

    assert not result_df.empty
    assert len(result_df) == 2
    
    data_2022 = result_df[result_df['date'] == pd.Timestamp('2022-12-31')]
    assert pd.isna(data_2022['eps'].iloc[0]) # "invalid_eps" should become NaN
    assert data_2022['bvps'].iloc[0] == pytest.approx(10.5)

    data_2021 = result_df[result_df['date'] == pd.Timestamp('2021-12-31')]
    assert data_2021['eps'].iloc[0] == pytest.approx(0.8)
    assert pd.isna(data_2021['bvps'].iloc[0]) # "bad_bvps" should become NaN

# --- Test Case 8: All required Chinese indicators are missing, but other indicators present ---
def test_get_financial_indicators_all_target_indicators_missing(data_fetcher_instance):
    raw_data = {
        '指标': ['不相关指标A', '不相关指标B'],
        '20221231': [100, 200],
        '20211231': [101, 201],
    }
    mock_df = create_mock_akshare_financial_df(raw_data)

    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600008")

    assert not result_df.empty # It will have rows for dates
    assert list(result_df.columns) == EXPECTED_OUTPUT_COLS
    assert len(result_df) == 2 # Still 2 years of data
    # All target indicator columns should be NaN because their Chinese names were not found
    for col in ['eps', 'bvps', 'roe', 'debt_to_asset_ratio', 'gross_profit_margin', 'net_profit_margin']:
        assert result_df[col].isnull().all()

# --- Test Case 9: Input DataFrame has dates not in YYYYMMDD format (should be handled by pd.to_datetime) ---
# The function converts index to datetime using pd.to_datetime(transposed_df.index, format='%Y%m%d')
# If akshare returns a different date format that's still convertible by pd.to_datetime without format,
# this test might be more involved. Assuming akshare is consistent or pd.to_datetime is robust.
# For this test, let's assume the format string in the function ensures only YYYYMMDD is processed.
# If the date column name itself is different, that's another case.
# Here, we test if a column is not YYYYMMDD, it's ignored by the transpose's index.
def test_get_financial_indicators_mixed_date_formats_in_columns(data_fetcher_instance):
    raw_data = {
        '指标': ['每股收益'],
        '20221231': [1.0],      # Valid
        '2021-12-31': [0.9],    # Invalid for current logic that filters columns by startswith('20') and endswith('1231') before transpose.
                                # Actually, the code `transposed_df.index = pd.to_datetime(transposed_df.index, format='%Y%m%d')`
                                # would fail if '2021-12-31' was a column name and became an index entry.
                                # The current code structure: set_index('指标').T means columns become index.
                                # So, '2021-12-31' would be an index value. pd.to_datetime('2021-12-31', format='%Y%m%d') would error.
        'InvalidDate': [0.8]    # Invalid
    }
    # The function will try to convert '20221231', '2021-12-31', 'InvalidDate' as index of transposed_df
    # to datetime using format='%Y%m%d'. '2021-12-31' and 'InvalidDate' will become NaT.
    # These NaT rows will then be dropped if 'date' is critical, or just be NaT.
    mock_df = create_mock_akshare_financial_df(raw_data)
    
    with patch('akshare.stock_financial_analysis_indicator', return_value=mock_df):
        result_df = data_fetcher_instance.get_financial_analysis_indicators("SH600009")
        
    assert not result_df.empty
    assert len(result_df) == 1 # Only the row from '20221231' should be valid
    assert result_df['date'].iloc[0] == pd.Timestamp('2022-12-31')
    assert result_df['eps'].iloc[0] == 1.0

# (Optional) Add more tests for specific data type conversions if necessary.
# The current tests cover the main logic paths and error handling for the processing part.
