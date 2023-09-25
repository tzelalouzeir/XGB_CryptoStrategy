# XGBoost Bitcoin Trading Strategy Analysis

This repository contains a Python script that fetches historical Bitcoin (BTC) price data, calculates technical indicators, and applies a trading strategy based on machine learning. The purpose of the script is to assess the performance of the strategy using the XGBoost algorithm, which predicts long, short, or neutral positions. 

## Key Features

1. **Data Fetching**: Fetches historical BTC-USD price data from Yahoo Finance.
2. **Technical Indicators**: Calculates 12 technical indicators such as SMA, EMA, MACD, RSI, and more.
3. **Signal Generation**: Determines trading signals (long/short/neutral) based on the change in closing price.
4. **Machine Learning**: Uses XGBoost to predict trading signals based on feature importance.
5. **Trading Strategy**: Implements a simulated trading strategy, recording entry and exit points and evaluating overall return.
6. **Visualization**: Displays feature importance, balance over time, and BTC-USD close prices with trading points.

## Libraries Used
- pandas
- pandas_ta
- yfinance
- numpy
- matplotlib
- xgboost
- scikit-learn

## Instructions

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/tzelalouzeir/XGB_CryptoStrategy.git
   ```

2. **Install Required Libraries**:
   Ensure you have the required libraries. You can install them using:
   ```bash
   pip install pandas pandas_ta yfinance numpy matplotlib xgboost scikit-learn
   ```

3. **Run the Script**:
   ```bash
   python xgb_backtrade.py
   ```

4. **Analyze the Results**: 
   After running the script, it will display various metrics, visualizations, and the final trading balance.

## Notes

- If you wish to fetch fresh data, uncomment the lines related to the `yf.download` function and adjust the period and interval according to your preference.
- Adjust parameters such as `max_trades_per_day`, `leverage`, and the window size for signal generation to tune the strategy.
- Ensure to interpret the results in a historical context and not as financial advice.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer

This code is intended for educational purposes only. Past performance is not indicative of future results. Always do your research before making any investment.

## License

[MIT](https://choosealicense.com/licenses/mit/)
