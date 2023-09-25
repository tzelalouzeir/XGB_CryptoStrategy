import pandas as pd
import pandas_ta as ta
import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, log_loss

# # Fetching data and desire timeframe and save as csv, currently i want to use 30day (max) and 5min timeframe
# df= yf.download("BTC-USD",period="30d",interval="5m")
# df.to_csv('btc30d.csv')

# Load your data from CSV
df = pd.read_csv('btc30d.csv', parse_dates=['Datetime']) # For day timeframe using 'Date' column and for Hours and Minutes is 'Datetime'
df.set_index('Datetime', inplace=True)

# Calculate 12 indicators using pandas_ta
df.ta.sma(length=50, append=True)
df.ta.ema(length=50, append=True)
df.ta.wma(length=50, append=True)
df.ta.macd(fast=12, slow=26, append=True)
df.ta.rsi(length=14, append=True)
df.ta.bbands(length=20, append=True)
df.ta.adx(length=14, append=True)
df.ta.stoch(length=14, append=True)
df.ta.willr(length=14, append=True)
df.ta.roc(length=10, append=True)
df.ta.cci(length=20, append=True)
df.ta.atr(length=14, append=True)


# Create an empty column for long/short/neutral signals
df['Signal'] = None

# Set the window size for calculating price differences.
# This is the number of bars (or candles) we'll look back to compare the current Close price against.
# For instance, a window_size of 2 will compare the current Close price against the Close price from 2 bars ago.
window_size = 2

# Loop through the DataFrame starting from the position after the window_size.
# This ensures we always have a prior bar (or candle) to compare against.
for i in range(window_size, len(df)):
    
    # Calculate the difference between the current Close price and the Close price from window_size bars ago.
    close_diff = df['Close'].iloc[i] - df['Close'].iloc[i - window_size]
    
    # Based on the calculated difference, label the current position:
    # 1. 'long' if the Close price has increased.
    # 2. 'short' if the Close price has decreased.
    # 3. 'neutral' if there's no change in the Close price.
    if close_diff > 0:
        df.at[df.index[i], 'Signal'] = 'long'
    elif close_diff < 0:
        df.at[df.index[i], 'Signal'] = 'short'
    else:
        df.at[df.index[i], 'Signal'] = 'neutral'


# Drop OHLC and Adj Close cause we need only indicators (If we need it cause for now we need Close price for backtesting our strategy)
#df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close'], axis=1, inplace=True)

# Now you can analyze indicator values at each signal point
long_condition = df['Signal'] == 'long'
short_condition = df['Signal'] == 'short'
neutral_condition = df['Signal'] == 'neutral'

long_indicators = df.loc[long_condition].select_dtypes(include=['float64']).mean()
short_indicators = df.loc[short_condition].select_dtypes(include=['float64']).mean()
neutral_indicators = df.loc[neutral_condition].select_dtypes(include=['float64']).mean()


# Finding relations between indicators, for example correlation matrix
correlation_matrix = df.select_dtypes(include=['float64']).corr()

# Mostly I care about Entering Position Long and Short
print(f'\nCandle Look: {window_size}\n')
print('Long Indicators:\n', long_indicators)
print('Short Indicators:\n', short_indicators)
# print('Neutral Indicators:\n', neutral_indicators)
# print('Correlation Matrix:\n', correlation_matrix)

############################################ XGB MODEL ############################################

# Remove NaN rows
df.dropna(inplace=True)

# Label encode the target column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Signal'])

# Prepare data
X = df.drop(['Signal'], axis=1)
y = y_encoded

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the XGBoost model
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order and match the feature names
sorted_indices = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_indices]

# Plot the feature importances
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), sorted_names, rotation=90)
plt.show()

# Calculate Performance of Model 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
matrix = confusion_matrix(y_test, y_pred)
probabilities = clf.predict_proba(X_test)[:, 1]  # get the probability of the positive class
auc = roc_auc_score(y_test, probabilities)

# Visualize Performance
print(matrix)
print(report)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {auc:.2f}")
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

############################################ SELECTING STRATEGY ############################################
# Take top 5 indicators from XGBoost feature importance
top_5 = sorted_names[:5]

def strategy_signal(row):
    """
    This function determines a trading signal based on certain conditions using
    the top 5 indicators (as determined by the feature importance from the XGBoost model).

    Parameters:
    - row: A single row from a DataFrame, expected to have the indicators as its columns.

    Returns:
    - 'long', 'short', or 'neutral' based on the evaluation of the conditions.

    Details:
    1. long_conditions: Checks if ALL the values of the top 5 indicators in the current row
       are greater than the average values of these indicators when the signal was 'long' 
       in the past.
    
    2. short_conditions: Checks if ALL the values of the top 5 indicators in the current row 
       are less than the average values of these indicators when the signal was 'short' 
       in the past.

    If all the long_conditions are met, the function returns 'long'.
    If all the short_conditions are met, it returns 'short'.
    If neither set of conditions is fully met, it returns 'neutral'.
    """

    # Check if all values of top 5 indicators for the row are greater than the average
    # values for those indicators during 'long' signals.
    long_conditions = all(row[indicator] > long_indicators[indicator] for indicator in top_5)
    
    # Check if all values of top 5 indicators for the row are less than the average
    # values for those indicators during 'short' signals.
    short_conditions = all(row[indicator] < short_indicators[indicator] for indicator in top_5)
    
    if long_conditions:
        return 'long'
    elif short_conditions:
        return 'short'
    else:
        return 'neutral'


# Apply strategy
df['Strategy_Signal'] = df.apply(strategy_signal, axis=1)

# Set a maximum number of trades allowed in a 24-hour period. Given 5min data, it's set to 10 trades.
# This means the strategy will attempt to make a trade approximately every 2 hours (24 data points of 5min each).
max_trades_per_day = 10
hours_passed = 0
trade_count_today = 0

# Initialize trading parameters
initial_balance = 1000
balance = initial_balance
position = None  # Current position, either 'long', 'short' or None
entry_price = 0
take_profit = 0
stop_loss = 0

long_tp_pct = 1.01  # Take-profit threshold for long trades (1% above entry)
long_sl_pct = 0.99  # Stop-loss threshold for long trades (1% below entry)

short_tp_pct = 0.99  # Take-profit threshold for short trades (1% below entry)
short_sl_pct = 1.01  # Stop-loss threshold for short trades (1% above entry)

entry_points = {'long': [], 'short': []}
exit_points = {'long': [], 'short': []}

# Leverage setting, indicating potential amplification of profits or losses.
leverage = 24
balance_over_time = [initial_balance]  # Track balance over time for analysis
trade_dates = []  # Track dates of trades for analysis

# Loop through the DataFrame to evaluate trading conditions
for i in range(1, len(df)):
    hours_passed += 1

    # After 24 data points (2 hours with 5min data), reset trade counter.
    if hours_passed == 24:
        trade_count_today = 0
        hours_passed = 0

    # Check exit conditions if there's an open position
    if position:
        if position == 'long':
            # Check if price hit stop loss or take profit for long position
            if df['Close'].iloc[i] <= stop_loss or df['Close'].iloc[i] >= take_profit:
                pct_change = (df['Close'].iloc[i] - entry_price) / entry_price
                balance *= (1 + pct_change * leverage)
                position = None
                exit_points['long'].append(df.index[i])
                balance_over_time.append(balance)
                trade_dates.append(df.index[i])
        else:  # position == 'short'
            # Check if price hit stop loss or take profit for short position
            if df['Close'].iloc[i] >= stop_loss or df['Close'].iloc[i] <= take_profit:
                pct_change = (entry_price - df['Close'].iloc[i]) / entry_price
                balance *= (1 + pct_change * leverage)
                position = None
                exit_points['short'].append(df.index[i])
                balance_over_time.append(balance)
                trade_dates.append(df.index[i])

    # Check entry conditions if there's no open position and trade count hasn't reached the daily limit
    else:
        if trade_count_today < max_trades_per_day:
            if df['Strategy_Signal'].iloc[i] == 'long':
                position = 'long'
                entry_price = df['Close'].iloc[i]
                take_profit = entry_price * long_tp_pct
                stop_loss = entry_price * long_sl_pct
                entry_points['long'].append(df.index[i])
                trade_count_today += 1
                balance_over_time.append(balance)
                trade_dates.append(df.index[i])
            elif df['Strategy_Signal'].iloc[i] == 'short':
                position = 'short'
                entry_price = df['Close'].iloc[i]
                take_profit = entry_price * short_tp_pct
                stop_loss = entry_price * short_sl_pct
                entry_points['short'].append(df.index[i])
                trade_count_today += 1
                balance_over_time.append(balance)
                trade_dates.append(df.index[i])

# After looping through the data, close any open positions at the final price
if position == 'long':
    pct_change = (df['Close'].iloc[-1] - entry_price) / entry_price
    balance *= (1 + pct_change * leverage)
    balance_over_time.append(balance)
    trade_dates.append(df.index[i])
elif position == 'short':
    pct_change = (entry_price - df['Close'].iloc[-1]) / entry_price
    balance *= (1 + pct_change * leverage)
    balance_over_time.append(balance)
    trade_dates.append(df.index[i])

# Display trading results
print(f"Final Balance: ${balance:.2f}")
total_return = (balance - initial_balance) / initial_balance * 100
print(f"Total Return: {total_return:.2f}%")


# Prepare parameters' details
parameters_details = ", ".join(top_5)
params_title = "Trading Parameters"
indicators_title = "Top Indicators"

# Create a figure
plt.figure(figsize=(15, 15))

# First plot for displaying the parameters
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st plot
plt.axis('off')

# Add titles and details with various font sizes and weights
plt.text(0.5, 0.75, params_title, ha='center', va='center', fontsize=16, fontweight='bold', color='black')
plt.text(0.5, 0.6, f"Max Trades Per Day: {max_trades_per_day}", ha='center', va='center', fontsize=12, color='#555555')
plt.text(0.5, 0.5, f"Leverage: {leverage}", ha='center', va='center', fontsize=12, color='#555555')
plt.text(0.5, 0.35, indicators_title, ha='center', va='center', fontsize=16, fontweight='bold', color='black')
plt.text(0.5, 0.15, parameters_details, ha='center', va='center', fontsize=12, color='#555555')

# Second plot for Balance Over Time
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd plot
plt.plot(trade_dates, balance_over_time[:len(trade_dates)], color='magenta', alpha=0.6, label='Balance')
plt.title('Balance Over Time')
plt.xlabel('Datetime')
plt.ylabel('Balance')
plt.legend()
plt.grid(True)

# Third plot for Close Price with Entry and Exit Points
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd plot
plt.plot(df['Close'], label='Close Price', color='blue', alpha=0.6)

# Plot long entry points
long_dates = entry_points['long']
plt.scatter(long_dates, df['Close'][long_dates], color='green', marker='^', alpha=1, label='Long Entry')

# Plot short entry points
short_dates = entry_points['short']
plt.scatter(short_dates, df['Close'][short_dates], color='red', marker='v', alpha=1, label='Short Entry')

# Plot long exit points
long_exit_dates = exit_points['long']
plt.scatter(long_exit_dates, df['Close'][long_exit_dates], color='green', marker='v', alpha=1, label='Long Exit')

# Plot short exit points
short_exit_dates = exit_points['short']
plt.scatter(short_exit_dates, df['Close'][short_exit_dates], color='red', marker='^', alpha=1, label='Short Exit')

plt.title('BTC-USD Close Price with Entry and Exit Points')
plt.xlabel('Datetime')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

# Adjust the layout to make sure everything fits well
plt.tight_layout()
plt.subplots_adjust(hspace=0.4) 
plt.show()
