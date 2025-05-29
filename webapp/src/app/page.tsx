'use client';

import { useState, useEffect } from 'react';

// Define an interface for the expected stock data structure
interface StockData {
  // Using English keys as they are likely from the DataFrame before mapping in app.py for JSON
  // These should match the keys in the JSON output from the API
  name: string; // 股票名称
  symbol: string; // 股票代码
  prediction: number; // 上涨概率
  pe?: number | null; // 市盈率(PE) - Optional and can be null
  roe?: number | null; // 净资产收益率(ROE) - Optional and can be null
  momentum_score?: number | null; // 动量得分
  explosion_score?: number | null; // 爆发潜力值
  // Add other relevant fields as needed
  [key: string]: any; // Allow other properties
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
          if (Array.isArray(result)) {
            setData(result);
          } else if (typeof result === 'object' && result !== null && result.error) {
            // Handle cases where the API itself returns an error JSON
            setError(result.error + (result.details ? `: ${result.details}` : ''));
            setData([]); // Clear data
          } 
          else {
            // If the result is not an array (e.g. a single object, or unexpected format)
            console.error('Fetched data is not an array:', result);
            setError('Received unexpected data format from API.');
            setData([]); // Clear data
          }
        } else {
          const errData = await response.json().catch(() => ({ error: 'Failed to fetch data. Server returned an error.' }));
          setError(errData.error || `Failed to fetch data. Status: ${response.status}`);
          setData([]); // Clear data
        }
      } catch (e: any) {
        console.error('Fetch error:', e);
        setError(e.message || 'An unexpected error occurred.');
        setData([]); // Clear data
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
  ];

  const formatValue = (value: any, key: string) => {
    if (value === null || typeof value === 'undefined' || (typeof value === 'number' && isNaN(value))) {
      return 'N/A';
    }
    if (typeof value === 'number') {
      if (key === 'prediction' || key === 'roe') {
        return (value * 100).toFixed(2) + '%';
      }
      return value.toFixed(2); // Default to 2 decimal places for numbers
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
