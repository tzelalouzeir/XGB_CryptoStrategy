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
   ![Result](https://github.com/tzelalouzeir/XGB_CryptoStrategy/blob/main/img/results.png)
   ![Features](https://github.com/tzelalouzeir/XGB_CryptoStrategy/blob/main/img/features.png)
   ![ROC](https://github.com/tzelalouzeir/XGB_CryptoStrategy/blob/main/img/roc.png)
   ![Performance](https://github.com/tzelalouzeir/XGB_CryptoStrategy/blob/main/img/per.PNG)

## Notes

- If you wish to fetch fresh data, uncomment the lines related to the `yf.download` function and adjust the period and interval according to your preference.
- Adjust parameters such as `max_trades_per_day`, `leverage`, and the window size for signal generation to tune the strategy.
- Ensure to interpret the results in a historical context and not as financial advice.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer

This code is intended for educational purposes only. Past performance is not indicative of future results. Always do your research before making any investment.

## Related Projects
- [Technical Analysis Repository](<https://github.com/tzelalouzeir/XGBoost_Indicators>): This repository fetches 120 days of hourly Bitcoin price data, calculates technical indicators, and analyzes the relations between these indicators.
- [Finding Features with XGBoost](https://github.com/tzelalouzeir/XGBoost_Indicators_2): Training and evaluating an XGBoost classifier on the Bitcoin technical indicators dataset. It aims to predict trading signals (like 'long', 'short', or 'neutral') based on the values of various indicators.
- [XGBoost Model Optimization](https://github.com/tzelalouzeir/XGBoost_Indicators_3): Optimizing the hyperparameters of an XGBoost classifier using the hyperopt library.

## 🤝 Let's Connect!
Connect with me on [LinkedIn](https://www.linkedin.com/in/tzelalouzeir/).

For more insights into my work, check out my latest project: [tafou.io](https://tafou.io).

I'm always eager to learn, share, and collaborate. If you have experiences, insights, or thoughts about RL, Prophet, XGBoost, SARIMA, ARIMA, or even simple Linear Regression in the domain of forecasting, please create an issue, drop a comment, or even better, submit a PR! 

_Let's learn and grow together!_ 🌱

## License

[MIT](https://choosealicense.com/licenses/mit/)
