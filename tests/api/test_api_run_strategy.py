import pytest
import requests
import json

# This is an integration test and requires the Next.js development server 
# (and the Python backend it calls) to be running on http://localhost:3000.
# You can mark it or skip it in environments where the server isn't available.
# Example using a custom marker (requires pytest.ini configuration):
# @pytest.mark.integration 
# Example of skipping explicitly:
# @pytest.mark.skip(reason="Requires Next.js server to be running on localhost:3000")

def test_run_strategy_endpoint():
    """
    Integration test for the /api/run-strategy endpoint.
    Assumes the Next.js server is running on localhost:3000.
    """
    api_url = "http://localhost:3000/api/run-strategy"
    
    print(f"Attempting to connect to API endpoint: {api_url}") # For visibility during test runs

    try:
        response = requests.get(api_url, timeout=300) # 5 minutes timeout
        
        # Check for HTTP errors (4xx or 5xx)
        response.raise_for_status() 

        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        # print(f"Response content (first 500 chars): {response.text[:500]}") # For debugging if needed

        # Assert Content-Type
        assert 'application/json' in response.headers.get('Content-Type', ''), \
            f"Expected 'application/json' in Content-Type, got '{response.headers.get('Content-Type')}'"

        # Try to parse JSON
        data = response.json()

        # Assert Data Structure (Basic)
        assert isinstance(data, list), f"Expected response data to be a list, got {type(data)}"
        
        if data: # If the list is not empty
            print(f"Received {len(data)} items in the list. First item: {data[0]}")
            assert isinstance(data[0], dict), f"Expected list items to be dictionaries, got {type(data[0])}"
            
            # Check for a few expected keys based on app.py's JSON output
            # These keys are from the DataFrame columns after processing in app.py,
            # before they are mapped to Chinese names for console display in app.py.
            # The JSON output from app.py (when --json-output is used)
            # uses the DataFrame's original column names.
            # Let's refer to the keys as they would be in the DataFrame:
            # 'name', 'symbol', 'prediction', 'pe', 'roe', 'ma15', etc.
            
            # Expected keys in the JSON output (DataFrame column names)
            # These should match the keys in the `result_dict` in `QuantStrategy.analyze_stock`
            # and also columns added by `calculate_ma_for_stocks` like 'ma15'.
            expected_keys = [
                'name', 'symbol', 'market', 'prediction', 'momentum_score', 'rsi',
                'volatility', 'macd', 'macd_status', 'close', 'volume', 'turn',
                'explosion_score', 'eps', 'bvps', 'roe', 'debt_to_asset_ratio',
                'gross_profit_margin', 'net_profit_margin', 'ma15' # 'ma15' is added by calculate_ma_for_stocks
            ]
            
            for key in expected_keys:
                assert key in data[0], f"Expected key '{key}' not found in the first item of the response data."
            
            # Example of checking a few specific values if you have expected ranges or types
            assert isinstance(data[0]['name'], str), "Stock name should be a string"
            assert isinstance(data[0]['prediction'], float), "Prediction score should be a float"
            if data[0].get('pe') is not None: # PE can be NaN/null
                 assert isinstance(data[0]['pe'], (float, int)), "PE should be a number or null"

        else:
            print("API returned an empty list. This might be valid depending on strategy results.")
            # No further structural checks if the list is empty.

    except requests.exceptions.ConnectionError:
        pytest.skip("Next.js server not running or not reachable on localhost:3000. Skipping integration test.")
    except requests.exceptions.Timeout:
        pytest.fail(f"Request to {api_url} timed out after 5 minutes.")
    except requests.exceptions.HTTPError as e:
        # Attempt to get more details from the response if it's an HTTP error
        error_details = response.text # Or response.json() if the error is JSON
        pytest.fail(f"API endpoint test failed with HTTPError: {e}. Response body: {error_details}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse JSON response: {e}. Response text: {response.text}")
    except Exception as e:
        pytest.fail(f"API endpoint test failed with an unexpected error: {e}")

# To run this test, ensure the Next.js server is running:
# 1. Navigate to the 'webapp' directory: `cd webapp`
# 2. Start the dev server: `npm run dev`
# 3. In a separate terminal, navigate to the project root and run pytest:
#    `pytest tests/api/test_api_run_strategy.py`
#
# If you use custom markers like @pytest.mark.integration, configure pytest.ini:
# [pytest]
# markers =
#     integration: marks tests as integration tests (requires running server)
