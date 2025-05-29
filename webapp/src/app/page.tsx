'use client';

import { useState, useEffect } from 'react';

// Define an interface for the expected stock data structure
interface StockData {
  name: string;
  symbol: string;
  prediction: number;
  market?: string;
  momentum_score?: number | null;
  rsi?: number | null;
  close?: number | null;
  ma15?: number | null;
  explosion_score?: number | null;
  macd_status?: string | null;
  pe?: number | null;
  pb?: number | null;
  roe?: number | null;
  debt_to_asset_ratio?: number | null;
  eps?: number | null;
  bvps?: number | null;
  gross_profit_margin?: number | null;
  net_profit_margin?: number | null;
  volume?: number | null;
  turn?: number | null;
  volatility?: number | null;
  macd?: number | null;
  // The [key: string]: any; is generally discouraged if specific keys are known.
  // We'll rely on the specific optional properties above.
}

export default function Home() {
  const [data, setData] = useState<StockData[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/run-strategy');
        if (response.ok) {
          const result = await response.json();
          const result = await response.json();
          // Type assertion for the result
          if (Array.isArray(result) && result.every(item => typeof item === 'object' && item !== null)) {
            setData(result as StockData[]);
          } else if (typeof result === 'object' && result !== null && (result as any).error) {
            // Handle cases where the API itself returns an error JSON
            const apiError = result as { error: string; details?: string };
            setError(apiError.error + (apiError.details ? `: ${apiError.details}` : ''));
            setData([]); 
          } else {
            console.error('Fetched data is not an array of objects:', result);
            setError('Received unexpected data format from API.');
            setData([]); 
          }
        } else {
          let errorDetails = `Status: ${response.status}`;
          try {
            const errData = await response.json();
            errorDetails = (errData as any).error || (errData as any).details || JSON.stringify(errData);
          } catch {
            // If parsing error JSON fails, use status text or generic message
            errorDetails = response.statusText || 'Server returned an error.';
          }
          setError(`Failed to fetch data. ${errorDetails}`);
          setData([]); 
        }
      } catch (e: unknown) { // Changed from e: any to e: unknown
        console.error('Fetch error:', e);
        if (e instanceof Error) {
          setError(e.message);
        } else {
          setError('An unexpected error occurred during fetch.');
        }
        setData([]); 
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []); // Empty dependency array means this effect runs once on mount

  const displayColumns = [
    { key: 'name', label: '股票名称' },
    { key: 'symbol', label: '股票代码' },
    { key: 'prediction', label: '上涨概率 (%)' },
    { key: 'pe', label: '市盈率(PE)' },
    { key: 'roe', label: '净资产收益率(ROE) (%)' },
    { key: 'momentum_score', label: '动量得分' },
    { key: 'explosion_score', label: '爆发潜力值' },
    { key: 'debt_to_asset_ratio', label: '资产负债率 (%)' },
    { key: 'eps', label: '每股收益(年)'},
    { key: 'bvps', label: '每股净资产(年)'},
    { key: 'gross_profit_margin', label: '销售毛利率(年) (%)'},
    { key: 'net_profit_margin', label: '销售净利率(年) (%)'}
  ];

  // Type for value in formatValue
  type StockValue = string | number | null | undefined;

  const formatValue = (value: StockValue, key: string) => {
    if (value === null || typeof value === 'undefined' || (typeof value === 'number' && isNaN(value))) {
      return 'N/A';
    }
    if (typeof value === 'number') {
      // Keys that should be displayed as percentages
      const percentageKeys = ['prediction', 'roe', 'debt_to_asset_ratio', 'gross_profit_margin', 'net_profit_margin'];
      if (percentageKeys.includes(key)) {
        return (value * 100).toFixed(2) + '%';
      }
      return value.toFixed(2); // Default to 2 decimal places for other numbers
    }
    return String(value);
  };


  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 sm:p-8 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <div className="w-full max-w-6xl p-4 sm:p-6 md:p-8 bg-white dark:bg-gray-800 shadow-xl rounded-lg">
        <h1 className="text-2xl sm:text-3xl font-bold text-center mb-6 sm:mb-8 text-gray-700 dark:text-gray-200">
          量化策略分析结果
        </h1>

        {loading && (
          <p className="text-center text-lg text-blue-500 dark:text-blue-400">
            Loading strategy results...
          </p>
        )}

        {error && (
          <p className="text-center text-lg text-red-500 dark:text-red-400 p-4 bg-red-100 dark:bg-red-900 rounded-md">
            Error: {error}
          </p>
        )}

        {!loading && !error && data && data.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  {displayColumns.map((col) => (
                    <th
                      key={col.key}
                      scope="col"
                      className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                    >
                      {col.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {data.map((stock, index) => (
                  <tr key={stock.symbol || index} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                    {displayColumns.map((col) => (
                      <td
                        key={col.key}
                        className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300"
                      >
                        {formatValue(stock[col.key], col.key)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!loading && !error && (!data || data.length === 0) && (
          <p className="text-center text-lg text-gray-500 dark:text-gray-400">
            No data available.
          </p>
        )}
      </div>
    </main>
  );
}
