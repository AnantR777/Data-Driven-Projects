import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Downloaded from yahoo finance, read in
df = pd.read_csv('SPTL.csv')
# Downloaded from newyorkfed.org
df_rates = pd.read_excel('EFFR.xlsx')
df_rates.set_index('Effective Date', inplace=True)


# Convert the 'Date' column to datetime format and set it as the index for first df
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
# df_rates has 'Effective Date' as its index but not in datetime format
df_rates.index = pd.to_datetime(df_rates.index)

# Calculate the daily risk-free rate
dc = 1/252
df_rates['Daily Rate (%)'] = df_rates['Rate (%)'] * dc

df = df[['Adj Close']]
df_rates = df_rates[['Daily Rate (%)']]

print(df.describe())
print(df_rates.describe())

merged_df = df.merge(df_rates, left_index=True, right_index=True, how='inner').sort_index()

print(f"# missing values: {merged_df.isnull().sum().sum()}") # there are no missing values

# Check for duplicate rows in the DataFrame
duplicate_rows = merged_df.duplicated()

# Count the number of duplicate rows
num_duplicate_rows = duplicate_rows.sum()

print(f"# duplicate rows: {num_duplicate_rows}")

# Remove duplicate rows, keeping the first occurrence
merged_df = merged_df.drop_duplicates()

print(merged_df.describe())
print(merged_df)


# Plotting both columns using subplots
fig, axs = plt.subplots(1, 2, figsize=(17, 8))  # 2 Rows, 1 Column

# Plotting 'Adj Close'
axs[0].plot(merged_df.index, merged_df['Adj Close'], label='Adj Close')
axs[0].set_title('SPTL ETF Adjusted Close Prices')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Adjusted Close Price')
axs[0].legend()
axs[0].grid(True)

# Plotting 'Daily Rate (%)'
axs[1].plot(merged_df.index, merged_df['Daily Rate (%)'], label='Daily Risk-Free Rate', color='orange')
axs[1].set_title('Effective Fed Funds Rate (Daily)')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Daily Rate (%)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("ETF and Rates plot.png")
plt.show()


# Calculate daily returns for SPTL ETF (pct_change finds the return as a decimal)
merged_df['SPTL Return'] = merged_df['Adj Close'].pct_change()

# Calculate the daily excess return per unit of SPTL
merged_df['Excess Return'] = merged_df['SPTL Return'] - (merged_df['Daily Rate (%)']/100)
# converted percentage daily rate to decimal, note since the rf rate is so small
# it barely has an impact on the return

# Plotting in a 1x3 subplot layout
fig, axs = plt.subplots(1, 3, figsize=(21, 5), gridspec_kw={'wspace': 0.35})  # Increase horizontal spacing

# Plotting the SPTL ETF Return Time Series
axs[0].plot(merged_df.index, merged_df['SPTL Return'], color='blue', label='SPTL Return')
axs[0].set_title('SPTL ETF Return')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Return')
axs[0].legend()
axs[0].grid(True)

# Plotting the EFFR
axs[1].plot(merged_df.index, merged_df['Daily Rate (%)'], color='orange', label='EFFR')
axs[1].set_title('Effective Fed Funds Rate (Daily)')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Rate (%)')
axs[1].legend()
axs[1].grid(True)

# Plotting the Excess Return per unit of SPTL
axs[2].plot(merged_df.index, merged_df['Excess Return'], color='green', label='Excess Return')
axs[2].set_title('Excess Return per unit of SPTL')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Excess Return')
axs[2].legend()
axs[2].grid(True)

plt.savefig("ETF_Rates_Excess_Returns_plot.png")
plt.show()

initial_capital = 200000
initial_cash = 130000

# Convert to NumPy array
adj_close_array = merged_df['Adj Close'].values
# Train-test split - using 7/10 for training
train_size = int(len(adj_close_array) * 0.7) # use 7/10 for training
train_adj_close_array = adj_close_array[:train_size]
test_adj_close_array = adj_close_array[train_size:] # remaining for test


#########TRAINING SET ops########################

# Define a range of time windows to test
time_windows = [i for i in range(10, 95, 5)]  # For example, from 10 to 90, stepping by 5

# Dictionary to store final returns for each time window
returns_by_window = {}

for time_window in time_windows:
    cumsum = [0]
    # cumulative sum of the ETF price up to the point, more efficient than summing all and dividing

    ma = np.zeros(np.shape(train_adj_close_array))
    # storing the moving average values of the security's price changes
    w = np.zeros(np.shape(train_adj_close_array))
    # storing quantity of security held in portfolio
    cash = np.zeros(np.shape(train_adj_close_array))
    # tracking the cash held in the portfolio over time.
    cash[0] = initial_cash

    for i, x in enumerate(train_adj_close_array[:-1],0):
        # iterate through each price except the last one, starts at first index
        cumsum.append(cumsum[i] + x)  # keeping a running total (cumulative sum) of the prices
        ma[i] = x  # mavg is just current price when window not big enough
        if i >= time_window:  # check if there are enough datapoints
            moving_ave = (cumsum[i] - cumsum[i - time_window]) / (time_window)
            ma[i] = moving_ave  # efficient way to find mavg

        if ma[i] == x:
            w[i + 1] = w[i]  # same quantity as previously, just hold
            cash[i + 1] = cash[
                i]  # same cash as previously (no extra/less amount invested)

        if ma[i] < x:
            w[i + 1] = cash[i] / x + w[
                i]  # convert all cash to security - fractional trading allowed
            cash[i + 1] = 0  # no cash left over

        if ma[i] > x:
            cash[i + 1] = w[i] * x + cash[i]  # convert all security to cash
            w[i + 1] = 0

    ma[i + 1] = train_adj_close_array[len(train_adj_close_array) - 1]

    strategy_value = [a * b for a, b in zip(w, train_adj_close_array)] + cash
    total_return_strategy = (strategy_value[-1] - strategy_value[0]) / \
                            strategy_value[0]

    # Store the final return for this time window
    returns_by_window[time_window] = total_return_strategy

    # What is the total return of my strategy? (can check by uncommenting below for current strat)
    #print((strategy_value[-1]-strategy_value[0])/strategy_value[0])
    # Total return of buying and holding

# Plotting Time Window vs Total Return
fig, ax = plt.subplots(1, 2, figsize=(24, 6))

# Plotting Time Window vs Total Return
ax[0].plot(list(returns_by_window.keys()), list(returns_by_window.values()), marker='o')
ax[0].set_title('Strat 1: Time Window vs Total Return (validation on train set)')
ax[0].set_xlabel('Time Window')
ax[0].set_ylabel('Total Return (decimal)')
ax[0].grid(True)

# Find the best performing time window
best_window = max(returns_by_window, key=returns_by_window.get) # use the one that gives the best return

# Rerun the strategy for the best performing time window
time_window = best_window
leverage = 10


cumsum = [0]  # cumulative sum of the ETF price up to the point, more efficient than summing all and dividing

ma = np.zeros(np.shape(
    train_adj_close_array))  # storing the moving average values of the security's price changes
w = np.zeros(np.shape(
    train_adj_close_array))  # storing amount of security held in portfolio
cash = np.zeros(np.shape(
    train_adj_close_array))  # tracking the cash held in the portfolio over time.
amount_borrowed = np.zeros(np.shape(train_adj_close_array))
cash[0] = initial_cash*leverage

for i, x in enumerate(train_adj_close_array[:-1], 0):
    # iterate through each price except the last one, starts at first index
    cumsum.append(cumsum[i] + x)  # keeping a running total (cumulative sum) of the prices
    ma[i] = x  # mavg is just current price when window not big enough
    if i >= time_window:  # check if there are enough datapoints
        moving_ave = (cumsum[i] - cumsum[i - time_window]) / (time_window)
        ma[i] = moving_ave  # efficient way to find mavg

    if ma[i] == x:
        w[i + 1] = w[i]  # same amount as previously, just hold
        cash[i + 1] = cash[i]  # same cash as previously (no extra/less amount invested)
        amount_borrowed[i + 1] = 0

    if ma[i] < x:
        w[i + 1] = cash[i] / x + w[i]  # convert all cash to security - fractional trading allowed
        cash[i + 1] = 0  # no cash left over
        amount_borrowed[i + 1] = cash[i] - cash[i] / leverage

    if ma[i] > x:
        cash[i + 1] = w[i] * x + cash[i]  # convert all security to cash
        w[i + 1] = 0
        amount_borrowed[i + 1] = 0

ma[i + 1] = train_adj_close_array[len(train_adj_close_array) - 1]

borrowing_cost = np.zeros_like(w)
for i in range(len(w)):
    borrowing_cost[i] = amount_borrowed[i]*(merged_df['Daily Rate (%)']/100).iloc[i]

strategy_value = [a * b for a, b in zip(w, train_adj_close_array)] + cash - borrowing_cost
total_return_strategy = (strategy_value[-1] - strategy_value[0]) / \
                        strategy_value[0]

# What is the total return of the best strategy?
print(f"Best mavg timewindow ({best_window} days) strat return (found using train set):"
      f"{(strategy_value[-1] - strategy_value[0]) / strategy_value[0]}")
# Total return of buying and holding
print(f"Buy and hold return (train set): {(train_adj_close_array[-1] - train_adj_close_array[0]) / train_adj_close_array[0]}")

# After running the strategy for the best window, plot the results:
# 1. Moving Average and Adjusted Close Prices for Best Window
ax[1].plot(merged_df.iloc[:train_size].index, ma, label='Moving Average', color='orange')
ax[1].plot(merged_df.iloc[:train_size].index, train_adj_close_array, label='Adjusted Close', color='blue')
ax[1].set_title(f"Best Time Window ({best_window} days): Moving Average vs Adjusted Close Prices (train set)")
ax[1].set_xlabel("Date")
ax[1].set_ylabel("ETF Price in Dollars")
ax[1].legend()
plt.savefig("Valdation and mavg for strat1.png")
plt.show()

# 2. Strategy Performance vs Buy and Hold for Best Window
strategy_value = [a * b for a, b in zip(w, train_adj_close_array)] + cash - borrowing_cost
#buy_and_hold = initial_cash * train_adj_close_array / train_adj_close_array[0]

plt.figure(figsize=(12, 6))
plt.plot(merged_df.iloc[:train_size].index, strategy_value, label='Strategy', color='red')
#plt.plot(merged_df.iloc[:train_size].index, buy_and_hold, label='Buy and Hold', color='cyan')
plt.title(f"Portfolio value for strat (including cash) on Train")
plt.xlabel("Date")
plt.ylabel("Portfolio value in Dollars")
plt.legend()
plt.show()


theta_train = np.zeros_like(w)
for t in range(len(w)):
    theta_train[t] = w[t] * train_adj_close_array[t]  # Calculate the dollar value of the position at time t

# Daily trading PnL for training set
deltaV = np.zeros_like(w)
for i in range(len(w)):
    deltaV[i] = merged_df['Excess Return'].iloc[i]*theta_train[i]
deltaV[0] = initial_capital  # Set the first value to initial_capital for the sake of cumsum below to find Vt
# actually has undefined first value since it's day on day change
# all values after the first one are the actual deltaV

# Plotting V_t
Vt = deltaV.cumsum()
plt.plot(merged_df.iloc[:train_size].index, Vt, label='V_t', color='blue')
plt.title(f"V_t = Excess Return * Theta (training set)")
plt.xlabel("Date")
plt.ylabel("V_t in Dollars")
plt.legend()
plt.show()

# Calculate upper and lower bounds
upper_bound = Vt * leverage
lower_bound = -upper_bound

# Calculate turnover
turnover_dollars = np.abs(np.diff(theta_train)).sum()

# Plotting the strategy position and bounds
plt.figure(figsize=(14, 7))
plt.plot(merged_df.iloc[:train_size].index, theta_train, label=r'$\theta_t$ - Strategy Position')
plt.plot(merged_df.iloc[:train_size].index, upper_bound, label=r'Upper Bound $V_t \cdot L$')
plt.plot(merged_df.iloc[:train_size].index, lower_bound, label=r'Lower Bound $-V_t \cdot L$')

plt.fill_between(merged_df.iloc[:train_size].index, upper_bound, lower_bound, color='gray', alpha=0.1)

plt.title('Strat 1: Value of position (theta) with Upper and Lower Bounds (train set)')
plt.xlabel('Date')
plt.ylabel('Dollar Value')
plt.axhline(y=2000000, color='r', linestyle='-', linewidth=0.5)
plt.axhline(y=-2000000, color='r', linestyle='-', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.savefig("Strat1 train theta vs bounds.png")
plt.show()


# Initialize turnover arrays
turnover_dollars = np.zeros_like(w)
turnover_units = np.zeros_like(w)

# Calculate turnovers for each day (except the first because it has no previous day)
for t in range(1, len(theta_train)):
    turnover_dollars[t] = abs(theta_train[t] - theta_train[t-1])
    turnover_units[t] = abs(theta_train[t] / train_adj_close_array[t] - theta_train[t-1] / train_adj_close_array[t-1])

total_turnover_dollars = np.sum(turnover_dollars)
total_turnover_units = np.sum(turnover_units)
print(f"Total turnover in dollars (train set): {total_turnover_dollars}")
print(f"Total turnover in units (train set): {total_turnover_units}")

# Define the moving average window, e.g., 75 days
moving_average_window = 75

# Calculate moving averages using a simple rolling window approach
moving_average_turnover_dollars = np.convolve(turnover_dollars, np.ones(moving_average_window), 'valid') / moving_average_window
moving_average_turnover_units = np.convolve(turnover_units, np.ones(moving_average_window), 'valid') / moving_average_window

# Plotting the moving averages
plt.figure(figsize=(14, 7))

# Plot moving average of dollar turnover
plt.plot(merged_df.iloc[moving_average_window-1:train_size].index, moving_average_turnover_dollars, label='Moving Average Dollar Turnover')
plt.title('Strat 1: 75 Day Moving Average of Dollar Turnover (train set)')
plt.xlabel('Date')
plt.ylabel('Turnover in Dollars')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 train mavg dollar turnover.png")
plt.show()

# Plotting the moving averages
plt.figure(figsize=(14, 7))

# Plot moving average of unit turnover
plt.plot(merged_df.iloc[moving_average_window-1:train_size].index, moving_average_turnover_units, label='Moving Average Unit Turnover')

plt.title('Strat 1: 75 Day Moving Average of Unit Turnover (train set)')
plt.xlabel('Date')
plt.ylabel('Turnover in Units')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 train mavg unit turnover.png")
plt.show()

# Initialize arrays to store margin used and unused capital
Mt = np.zeros_like(theta_train)  # Total margin used
unused_capital = np.zeros_like(theta_train)  # Unused capital

# Convert risk-free rate from percent to decimal
risk_free_rate = merged_df['Daily Rate (%)'][:train_size] / 100

for t in range(1, len(theta_train)):  # We start at 1 since we cannot calculate the change for the first entry
    # Calculate total margin used
    Mt[t] = abs(theta_train[t]) / leverage # for this strat theta is positive anyway
    # Calculate unused capital for each day
    unused_capital[t] = strategy_value[t] - Mt[t]

# Now, we can also calculate the change in total value ΔVt_total for each day
deltaV[0] = 0 # actually undefined, but assign to 0 for the sake of the plot
delta_V_cap = np.zeros_like(theta_train)
delta_V_total = np.zeros_like(theta_train)
for t in range(1, len(delta_V_total)):
    # The change in the investment value ΔVt was defined earlier
    # The change in unused capital is ΔVt^cap
    delta_V_cap[t] = (unused_capital[t]) * risk_free_rate[t]
    delta_V_total[t] = deltaV[t] + delta_V_cap[t]

# Plot delta's
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[:train_size], delta_V_cap, label='Unused Capital')
plt.plot(merged_df.index[:train_size], deltaV, label='delta Vt')
plt.plot(merged_df.index[:train_size], delta_V_total, label='delta_V_total')
plt.title('Strat 1: delta V\'s Over Time (train set)')
plt.xlabel('Date')
plt.ylabel('Capital changes day on day in dollars')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 train delta V.png")
plt.show()

# Cumsum (V plots)
deltaV[0] = initial_capital
delta_V_total[0] = initial_capital
delta_V_cap[0] = initial_capital - initial_cash
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[:train_size], delta_V_cap.cumsum(), label='Cumulative Unused Capital')
plt.plot(merged_df.index[:train_size], deltaV.cumsum(), label='Vt')
plt.plot(merged_df.index[:train_size], delta_V_total.cumsum(), label='Vt_total')
plt.title('Strat 1: V\'s Over Time (train set)')
plt.xlabel('Date')
plt.ylabel('Capital in dollars')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 train V.png")
plt.show()

# Maximum Drawdown Calculation
Vt = np.cumsum(deltaV)
peak = np.maximum.accumulate(Vt*leverage) # computes the max cumulative return value seen so far
drawdowns = peak-(Vt*leverage)  # use this to find most -ve drawdown below
max_drawdown = np.max(drawdowns) / peak[np.argmax(drawdowns)] if peak[np.argmax(drawdowns)] != 0 else 0
# We've defined it as a positive drawdown

np.delete(deltaV, 0) # deletes first element because not defined in reality

# Calmar Ratio Calculation
avg_annual_return = np.mean(deltaV) * 252  # Assuming 252 trading days in a year
# above is unlevered
calmar_ratio = avg_annual_return*leverage / np.max(drawdowns) if max_drawdown != 0 else 0

port_return = np.zeros_like(theta_train)

# Calculate portfolio return for each day (except the first because it has no previous day)
# Excludes risk free rate
for t in range(1, len(theta_train)):
    port_return[t] = merged_df['SPTL Return'].iloc[t]*(theta_train[t])

# Sharpe Ratio Calculation
mean_daily_excess_return = np.mean(deltaV)*leverage
port_return_vol = np.std(port_return)
sharpe_ratio = mean_daily_excess_return / port_return_vol if port_return_vol != 0 else 0

# Sortino Ratio Calculation
negative_port_returns = port_return[port_return < 0]
downside_volatility = np.std(negative_port_returns)
sortino_ratio = mean_daily_excess_return / downside_volatility if downside_volatility != 0 else 0

print(f"Sharpe Ratio (train set): {sharpe_ratio}")
print(f"Sortino Ratio (train set): {sortino_ratio}")
print(f"Maximum Drawdown (train set): {max_drawdown}")
print(f"Calmar Ratio (train set): {calmar_ratio}")

# Convert deltaV to a pandas Series for ease of rolling calculations
deltaV_series = pd.Series(deltaV)
port_return_series = pd.Series(port_return)

# Calculate daily returns for the Sharpe Ratio
daily_excess_returns = deltaV_series * leverage

# Calculate rolling mean and standard deviation over a 75-day window
rolling_mean_excess_returns = daily_excess_returns.rolling(window=75).mean()
rolling_std_returns = port_return_series.rolling(window=75).std()

# Calculate the rolling Sharpe Ratio
# Assuming the risk-free rate is negligible for this calculation
rolling_sharpe_ratio = rolling_mean_excess_returns / rolling_std_returns

# Plot the 75-day rolling Sharpe Ratio
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[:train_size], rolling_sharpe_ratio.fillna(0), label='75-Day Rolling Sharpe Ratio')
plt.title('Strat 1: 75-Day Rolling Sharpe Ratio (train set)')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 train roll sharpe.png")
plt.show()

# Calculate rolling 90-day volatility (standard deviation of returns)
rolling_volatility = merged_df['Adj Close'].rolling(window=90).std()

# If you want to overlay the two plots, you'll need a second y-axis
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Drawdown', color=color)
ax1.plot(merged_df.iloc[:train_size].index, drawdowns, label='Drawdown', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('90-Day Rolling Volatility', color=color)
ax2.plot(merged_df.iloc[89:train_size].index, rolling_volatility.iloc[89:train_size], label='90-Day Rolling Volatility', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Strat 1: Drawdown and 90-Day Rolling Volatility (train set)')
plt.savefig("Strat 1 train drawdown vol.png")
plt.show()

########TEST SET ops#############################

initial_capital = 200000
initial_cash = 130000

# Convert to NumPy array
adj_close_array = merged_df['Adj Close'].values
# Train-test split - using 7/10 for training
train_size = int(len(adj_close_array) * 0.7)
test_adj_close_array = adj_close_array[train_size:]

# Rerun the strategy for the best performing time window
time_window = best_window
cumsum = [0]
ma = np.zeros(np.shape(test_adj_close_array))
w = np.zeros(np.shape(test_adj_close_array))
cash = np.zeros(np.shape(test_adj_close_array))

leverage = 10


# strategy loop for the best time window


cumsum = [0]  # cumulative sum of the ETF price up to the point, more efficient than summing all and dividing

ma = np.zeros(np.shape(
    test_adj_close_array))  # storing the moving average values of the security's price changes
w = np.zeros(np.shape(
    test_adj_close_array))  # storing amount of security held in portfolio
cash = np.zeros(np.shape(
    test_adj_close_array))  # tracking the cash held in the portfolio over time.
amount_borrowed = np.zeros(np.shape(test_adj_close_array))
cash[0] = initial_cash*leverage

for i, x in enumerate(test_adj_close_array[:-1], 0):
    # iterate through each price except the last one, starts at first index
    cumsum.append(cumsum[i] + x)  # keeping a running total (cumulative sum) of the prices
    ma[i] = x  # mavg is just current price when window not big enough
    if i >= time_window:  # check if there are enough datapoints
        moving_ave = (cumsum[i] - cumsum[i - time_window]) / (time_window)
        ma[i] = moving_ave  # efficient way to find mavg

    if ma[i] == x:
        w[i + 1] = w[i]  # same amount as previously, just hold
        cash[i + 1] = cash[i]  # same cash as previously (no extra/less amount invested)
        amount_borrowed[i + 1] = 0

    if ma[i] < x:
        w[i + 1] = cash[i] / x + w[i]  # convert all cash to security - fractional trading allowed
        cash[i + 1] = 0  # no cash left over
        amount_borrowed[i + 1] = cash[i] - cash[i]/leverage

    if ma[i] > x:
        cash[i + 1] = w[i] * x + cash[i]  # convert all security to cash
        w[i + 1] = 0
        amount_borrowed[i + 1] = 0

ma[i + 1] = test_adj_close_array[len(test_adj_close_array) - 1]

borrowing_cost = np.zeros_like(w)
for i in range(len(w)):
    borrowing_cost[i] = amount_borrowed[i]*(merged_df['Daily Rate (%)']/100).iloc[train_size+i]

strategy_value = [a * b for a, b in zip(w, test_adj_close_array)] + cash - borrowing_cost
total_return_strategy = (strategy_value[-1] - strategy_value[0]) / \
                        strategy_value[0]

# What is the total return of the best strategy?
print(f"Strat return for best window on test set: {(strategy_value[-1] - strategy_value[0]) / strategy_value[0]}")
# Total return of buying and holding
print(f"Buy and hold return: {(test_adj_close_array[-1] - test_adj_close_array[0]) / test_adj_close_array[0]}")

# After running the strategy for the best window, plot the results:
# 1. Moving Average and Adjusted Close Prices for Best Window on test data
plt.figure(figsize=(12, 6))
plt.plot(merged_df.iloc[train_size:].index, ma, label='Moving Average', color='orange')
plt.plot(merged_df.iloc[train_size:].index, test_adj_close_array, label='Adjusted Close', color='blue')
plt.title(f"Test Set Moving Average vs Adjusted Close Prices")
plt.xlabel("Date")
plt.ylabel("ETF Price in Dollars")
plt.legend()
plt.show()

# 2. Strategy Performance vs Buy and Hold for Best Window
strategy_value = [a * b for a, b in zip(w, test_adj_close_array)] + cash - borrowing_cost
#buy_and_hold = initial_cash * train_adj_close_array / train_adj_close_array[0]

plt.figure(figsize=(12, 6))
plt.plot(merged_df.iloc[train_size:].index, strategy_value, label='Strategy', color='blue')
#plt.plot(merged_df.iloc[train_size:].index, buy_and_hold, label='Buy and Hold', color='cyan')
plt.title(f"Portfolio value for strat (including cash) on Test")
plt.xlabel("Date")
plt.ylabel("Portfolio value in Dollars")
plt.legend()
plt.show()

theta_test = np.zeros_like(w)
for t in range(len(w)):
    theta_test[t] = w[t] * test_adj_close_array[t]  # Calculate the dollar value of the position at time t

# Daily trading PnL for training set
deltaV = np.zeros_like(w)
for i in range(len(w)):
    deltaV[i] = merged_df['Excess Return'].iloc[train_size+i]*theta_test[i]
deltaV[0] = initial_capital  # Set the first value to initial_capital for the sake of cumsum below to find Vt
# actually has undefined first value since it's day on day change
# all values after the first one are the actual deltaV

# Plotting V_t
Vt = deltaV.cumsum()
plt.plot(merged_df.iloc[train_size:].index, Vt, label='V_t', color='blue')
plt.title(f"V_t = Excess Return * Theta (test set)")
plt.xlabel("Date")
plt.ylabel("V_t in Dollars")
plt.legend()
plt.show()

# Calculate upper and lower bounds
upper_bound = Vt * leverage
lower_bound = -upper_bound

# Calculate turnover
turnover_dollars = np.abs(np.diff(theta_test)).sum()

# Plotting the strategy position and bounds
plt.figure(figsize=(14, 7))
plt.plot(merged_df.iloc[train_size:].index, theta_test, label=r'$\theta_t$ - Strategy Position')
plt.plot(merged_df.iloc[train_size:].index, upper_bound, label=r'Upper Bound $V_t \cdot L$')
plt.plot(merged_df.iloc[train_size:].index, lower_bound, label=r'Lower Bound $-V_t \cdot L$')

plt.fill_between(merged_df.iloc[train_size:].index, upper_bound, lower_bound, color='gray', alpha=0.1)

plt.title('Strat 1: Value of position (theta) with Upper and Lower Bounds (test set)')
plt.xlabel('Date')
plt.ylabel('Dollar Value')
plt.axhline(y=2000000, color='r', linestyle='-', linewidth = 0.5)
plt.axhline(y=-2000000, color='r', linestyle='-', linewidth = 0.5)
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 test theta vs bounds.png")
plt.show()


# Initialize turnover arrays
turnover_dollars = np.zeros_like(w)
turnover_units = np.zeros_like(w)

# Calculate turnovers for each day (except the first because it has no previous day)
for t in range(1, len(theta_test)):
    turnover_dollars[t] = abs(theta_test[t] - theta_test[t-1])
    turnover_units[t] = abs(theta_test[t] / test_adj_close_array[t] - theta_test[t-1] / test_adj_close_array[t-1])

total_turnover_dollars = np.sum(turnover_dollars)
total_turnover_units = np.sum(turnover_units)
print(f"Total turnover in dollars (test set): {total_turnover_dollars}")
print(f"Total turnover in units (test set): {total_turnover_units}")

# Define the moving average window
moving_average_window = 75

# Calculate moving averages using a simple rolling window approach
moving_average_turnover_dollars = np.convolve(turnover_dollars, np.ones(moving_average_window), 'valid') / moving_average_window
moving_average_turnover_units = np.convolve(turnover_units, np.ones(moving_average_window), 'valid') / moving_average_window

# Plotting the moving averages
plt.figure(figsize=(14, 7))

# Plot moving average of dollar turnover
plt.plot(merged_df.iloc[train_size+moving_average_window-1:].index, moving_average_turnover_dollars, label='Moving Average Dollar Turnover')
plt.title('Strat 1: 75-Day Moving Average of Dollar Turnover (test set)')
plt.xlabel('Date')
plt.ylabel('Turnover in Dollars')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 test mavg dollar turnover.png")
plt.show()

# Plotting the moving averages
plt.figure(figsize=(14, 7))

# Plot moving average of unit turnover
plt.plot(merged_df.iloc[train_size+moving_average_window-1:].index, moving_average_turnover_units, label='Moving Average Unit Turnover')

plt.title('Strat 1: 75-Day Moving Average of Unit Turnover (test set)')
plt.xlabel('Date')
plt.ylabel('Turnover in Units')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 test mavg unit turnover.png")
plt.show()

# Initialize arrays to store margin used and unused capital
Mt = np.zeros_like(theta_test)  # Total margin used
unused_capital = np.zeros_like(theta_test)  # Unused capital

# Convert risk-free rate from percent to decimal
risk_free_rate = merged_df['Daily Rate (%)'][train_size:] / 100

for t in range(1, len(theta_test)):  # We start at 1 since we cannot calculate the change for the first entry
    # Calculate total margin used
    Mt[t] = abs(theta_test[t]) / leverage # for this strat theta is positive anyway
    # Calculate unused capital for each day
    unused_capital[t] = strategy_value[t] - Mt[t]

# Now, we can also calculate the change in total value ΔVt_total for each day
deltaV[0] = 0 # actually undefined, but assign to 0 for the sake of the plot
delta_V_cap = np.zeros_like(theta_test)
delta_V_total = np.zeros_like(theta_test)
for t in range(1, len(delta_V_total)):
    # The change in the investment value ΔVt was defined earlier
    # The change in unused capital is ΔVt^cap
    delta_V_cap[t] = (unused_capital[t]) * risk_free_rate[t]
    delta_V_total[t] = deltaV[t] + delta_V_cap[t]

# Plot delta's
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[train_size:], delta_V_cap, label='Unused Capital')
plt.plot(merged_df.index[train_size:], deltaV, label='delta Vt')
plt.plot(merged_df.index[train_size:], delta_V_total, label='delta_V_total')
plt.title('Strat 1: delta V\'s Over Time (test set)')
plt.xlabel('Date')
plt.ylabel('Capital changes day on day in dollars')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 test delta V.png")
plt.show()

# Cumsum (V plots)
deltaV[0] = initial_capital
delta_V_total[0] = initial_capital
delta_V_cap[0] = initial_capital - initial_cash
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[train_size:], delta_V_cap.cumsum(), label='Cumulative Unused Capital')
plt.plot(merged_df.index[train_size:], deltaV.cumsum(), label='Vt')
plt.plot(merged_df.index[train_size:], delta_V_total.cumsum(), label='Vt_total')
plt.title('Strat 1: V\'s Over Time (test set)')
plt.xlabel('Date')
plt.ylabel('Capital changes day on day in dollars')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 test V.png")
plt.show()

# Maximum Drawdown Calculation
Vt = np.cumsum(deltaV)
peak = np.maximum.accumulate(Vt*leverage) # computes the max cumulative return value seen so far
drawdowns = peak-(Vt*leverage)  # use this to find most -ve drawdown below
max_drawdown = np.max(drawdowns) / peak[np.argmax(drawdowns)] if peak[np.argmax(drawdowns)] != 0 else 0
# We've defined it as a positive drawdown

np.delete(deltaV, 0) # deletes first element because not defined in reality

# Calmar Ratio Calculation
avg_annual_return = np.mean(deltaV) * 252  # Assuming 252 trading days in a year
# above is unlevered
calmar_ratio = avg_annual_return*leverage / np.max(drawdowns) if max_drawdown != 0 else 0

port_return = np.zeros_like(theta_test)

# Calculate portfolio return for each day (except the first because it has no previous day)
# Excludes risk free rate
for t in range(1, len(theta_test)):
    port_return[t] = merged_df['SPTL Return'].iloc[train_size+t]*(theta_test[t])

# Sharpe Ratio Calculation
mean_daily_excess_return = np.mean(deltaV)*leverage
port_return_vol = np.std(port_return)
sharpe_ratio = mean_daily_excess_return / port_return_vol if port_return_vol != 0 else 0

# Sortino Ratio Calculation
negative_port_returns = port_return[port_return < 0]
downside_volatility = np.std(negative_port_returns)
sortino_ratio = mean_daily_excess_return / downside_volatility if downside_volatility != 0 else 0

print(f"Sharpe Ratio (test set): {sharpe_ratio}")
print(f"Sortino Ratio (test set): {sortino_ratio}")
print(f"Maximum Drawdown (test set): {max_drawdown}")
print(f"Calmar Ratio (test set): {calmar_ratio}")

# Convert deltaV to a pandas Series for ease of rolling calculations
deltaV_series = pd.Series(deltaV)
port_return_series = pd.Series(port_return)

# Calculate daily returns for the Sharpe Ratio
daily_excess_returns = deltaV_series * leverage

# Calculate rolling mean and standard deviation over a 75-day window
rolling_mean_excess_returns = daily_excess_returns.rolling(window=75).mean()
rolling_std_returns = port_return_series.rolling(window=75).std()

# Calculate the rolling Sharpe Ratio
# Assuming the risk-free rate is negligible for this calculation
rolling_sharpe_ratio = rolling_mean_excess_returns / rolling_std_returns

# Plot the 75-day rolling Sharpe Ratio
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[train_size:], rolling_sharpe_ratio.fillna(0), label='75-Day Rolling Sharpe Ratio')
plt.title(f'Strat 1: 75-Day Rolling Sharpe Ratio (test set)')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.savefig("Strat 1 test roll sharpe.png")
plt.show()

# Calculate rolling 90-day volatility (standard deviation of returns)
rolling_volatility = merged_df['Adj Close'].rolling(window=90).std()

# If you want to overlay the two plots, you'll need a second y-axis
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Drawdown', color=color)
ax1.plot(merged_df.iloc[train_size:].index, drawdowns, label='Drawdown', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('90-Day Rolling Volatility', color=color)
ax2.plot(merged_df.iloc[train_size+89:].index, rolling_volatility.iloc[train_size+89:], label='90-Day Rolling Volatility', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Strat 1: Drawdown and 90-Day Rolling Volatility (test set)')
plt.savefig("Strat 1 test drawdown vol.png")
plt.show()
