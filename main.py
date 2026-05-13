#!/usr/bin/env python3
"""
QSSS - 量化选股系统主程序
兼容旧版本调用方式
"""

from src.qsss.core.strategy import QuantStrategy

if __name__ == "__main__":
    # 兼容旧版本直接运行
    strategy = QuantStrategy()
    selected_stocks = strategy.run_analysis()

    if not selected_stocks.empty:
        # 计算15日均线
        selected_stocks['ma15'] = selected_stocks['symbol'].apply(
            lambda x: strategy.calculate_ma15(x)
        )

        # 重命名列
        columns_map = {
            'name': '股票名称',
            'symbol': '股票代码',
            'market': '交易所-板块',
            'prediction': '上涨概率',
            'momentum_score': '动量得分',
            'rsi': 'RSI指标',
            'close': '收盘价',
            'ma15': '15日均线价格',
            'explosion_score': '爆发潜力值',
            'macd_status': 'MACD状态'
        }

        display_df = selected_stocks[list(columns_map.keys())].copy()
        display_df.columns = [columns_map[col] for col in display_df.columns]

        print("\n=== 选出的标的（前20名） ===")
        print(display_df.head(20).to_string(index=False))

        # 筛选15日均线在15元以内的股票
        low_price_stocks = display_df[display_df['15日均线价格'] <= 15]
        if not low_price_stocks.empty:
            print("\n=== 15日均线在15元以内的标的 ===")
            print(low_price_stocks.to_string(index=False))

        # 显示超短线爆发潜力股票
        explosion_stocks = selected_stocks[
            selected_stocks['explosion_score'] > 1.5
        ].sort_values('explosion_score', ascending=False)

        if not explosion_stocks.empty:
            explosion_display = explosion_stocks[list(columns_map.keys())].copy()
            explosion_display.columns = [columns_map[col] for col in explosion_display.columns]

            print("\n=== 超短线爆发潜力股票（前20名） ===")
            print(explosion_display.head(20).to_string(index=False))

            # 筛选15日均线在15元以内的爆发潜力股票
            low_price_explosion = explosion_display[explosion_display['15日均线价格'] <= 15]
            if not low_price_explosion.empty:
                print("\n=== 15日均线在15元以内的爆发潜力股票 ===")
                print(low_price_explosion.to_string(index=False))
    else:
        print("未找到符合条件的标的")
