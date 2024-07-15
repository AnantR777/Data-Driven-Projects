import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from alpha_vantage.timeseries import TimeSeries
import statsmodels
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from numpy.random import seed
from numpy.random import randn
from scipy.stats import kstest, norm


## Section 1: Time Series
#"""
# Store API key
api_key = 'VC9SRBBQK54Y6ASV'

# Using the API from alpha vantage to retrieve data
# our time series, default is json - but we want to use pandas to analyse the time series
ts = TimeSeries(key=api_key, output_format = 'pandas') # time series object
""" (In case we don't want to use an API use the below)
SPY_data = pd.read_csv('SPY.csv')
QQQ_data = pd.read_csv('QQQ.csv')
# Ticker symbols of the chosen ETFs
etf_symbols = ['SPY', 'QQQ']
etf_data = {'SPY':SPY_data, 'QQQ':QQQ_data}
"""
# Dictionary to hold data
etf_data = {}

etf_symbols = ['SPY', 'QQQ']
# Fetch and store data for each ETF
for symbol in etf_symbols:
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full') # gets daily data

    # Ensure the index is in datetime format
    data.index = pd.to_datetime(data.index)

    # Sort the data by date to ensure it's in chronological order
    data_sorted = data.sort_index()

    # Filter data up until end of jan 2024
    end_date = datetime(2024, 1, 31)
    filtered_data = data_sorted[data_sorted.index <= end_date] # want data points 31st jan 2024 and before

    # Select the last 300 business days
    etf_data[symbol] = filtered_data.tail(300) # gets the last 300 data points
#"""

# Etf_data contains the last 300 data points for 'SPY' and 'QQQ' (most recent being 31/02/2024)
spy_close = etf_data['SPY']['4. close']  # Closing prices of SPY
qqq_close = etf_data['QQQ']['4. close']  # Closing prices of QQQ

# Assuming etf_data is your dictionary containing pandas DataFrames for 'SPY' and 'QQQ'
spy_close_description = etf_data['SPY']['4. close'].describe() # gets statistics
qqq_close_description = etf_data['QQQ']['4. close'].describe()

# Print the descriptive statistics
print("SPY Closing Prices Descriptive Statistics:")
print(spy_close_description)
print("\nQQQ Closing Prices Descriptive Statistics:")
print(qqq_close_description)

# Set the plot style
plt.style.use('seaborn-darkgrid')  # Use seaborn-darkgrid style for a modern and stylish plot

# Create a plot
plt.figure(figsize=(12, 6))

# Plot the data of closing prices
plt.plot(spy_close.index, spy_close, label='SPY', color='royalblue', linewidth=2, marker='o', markersize=4)
plt.plot(qqq_close.index, qqq_close, label='QQQ', color='darkorange', linewidth=2, marker='^', markersize=4)

# Adding titles and labels with enhanced formatting
plt.title('SPY vs QQQ - Last 300 Days', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)

# Customize the legend
plt.legend(frameon=True, loc='upper left', fontsize=12)

# Format the x-axis to show dates more clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)  # Rotate date labels for better readability

# Add grid for better readability, but lighter than default
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a tight layout to ensure everything fits without overlap
plt.tight_layout()

# Save the figure
plt.savefig('ETFplots.png')

# Show the plot
plt.show()


## Section 2: Moving averages
# 3. Define mathematically the moving average of the price time series with an arbitrary timewindow τ.

# Compute moving averages for different windows (in days)
time_windows = [5, 20, 60]

# Iterate through the different time windows for both ETFs
for window in time_windows:
    for symbol in etf_symbols:
        etf_data[symbol]['MA_' + str(window)] = etf_data[symbol]['4. close'].rolling(window=window).mean()
        print(etf_data[symbol]['MA_' + str(window)])
# Plotting
plt.figure(figsize=(12, 6))

# Plot for SPY
plt.subplot(2, 1, 1)  # Two rows, one column, first subplot
plt.plot(etf_data['SPY']['4. close'], label='SPY Close')
for window in time_windows:
    plt.plot(etf_data['SPY']['MA_' + str(window)], label=f'SPY MA {window} Days')
plt.title('SPY Closing Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot for QQQ
plt.subplot(2, 1, 2)  # Two rows, one column, second subplot
plt.plot(etf_data['QQQ']['4. close'], label='QQQ Close')
for window in time_windows:
    plt.plot(etf_data['QQQ']['MA_' + str(window)], label=f'QQQ MA {window} Days')
plt.title('QQQ Closing Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.savefig('MovingAvgPlots.png')
plt.show()


# Function to calculate linear and log return
## 6. Computing the linear and log returns

def calculate_returns(df, price_column):
    df['Linear_Return'] = df[price_column][1:]/df[price_column][:-1].values - 1 # uses linear return formula as in the doc but divided through
    df['Log_Return'] = np.log(df[price_column] / df[price_column].shift(1)) # we want to divide prices by their 1-shift
    df['Cumulative_Linear_Return'] = (1 + df['Linear_Return']).cumprod() - 1
    df['Cumulative_Log_Return'] = df['Log_Return'].cumsum() # log returns are additive
    return df

# Calculating returns for each ETF
for symbol in etf_symbols:
    etf_data[symbol] = calculate_returns(etf_data[symbol], '4. close')
    print(etf_data[symbol][['Linear_Return', 'Log_Return', 'Cumulative_Linear_Return', 'Cumulative_Log_Return']])
    # displays returns, just to see there are no abnormalities

## 7. Plotting the linear and log returns against time

# Clear, distinguishable colour palette
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Colours for lines
lineStyles = ['-', '--', '-.', ':']  # Line styles for differentiation

plt.figure(figsize=(14, 10))  # Big figure size for better readability

# Linear Return vs Log Return for each ETF
for i, symbol in enumerate(etf_symbols):
    ax = plt.subplot(2, 2, i + 1)  # 2 rows, 2 columns, subplot i+1
    ax.plot(etf_data[symbol]['Linear_Return'], label=f'{symbol} Linear Return', color=colors[0], linestyle=lineStyles[0])
    ax.plot(etf_data[symbol]['Log_Return'], label=f'{symbol} Log Return', color=colors[1], linestyle=lineStyles[1])
    ax.set_title(f'{symbol} Linear vs Log Return', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.legend(frameon=True, framealpha=0.8, loc='best')

# Cumulative Linear Return vs Cumulative Log Return for each ETF
for i, symbol in enumerate(etf_symbols):
    ax = plt.subplot(2, 2, i + 3)  # 2 rows, 2 columns, subplot i+3
    ax.plot(etf_data[symbol]['Cumulative_Linear_Return'], label=f'{symbol} Cumulative Linear Return', color=colors[2], linestyle=lineStyles[2])
    ax.plot(etf_data[symbol]['Cumulative_Log_Return'], label=f'{symbol} Cumulative Log Return', color=colors[3], linestyle=lineStyles[3])
    ax.set_title(f'{symbol} Cumulative Linear vs Cumulative Log Return', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(frameon=True, framealpha=0.8, loc='best')

plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust the spacing for better readability
plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
plt.savefig('ReturnsPlots.png')
plt.show()


## 8-16 Compute and Plot the ACFs and PACFs of the Price and Return Time Series

# Define a colour palette for differentiation
colors = plt.cm.viridis(np.linspace(0, 1, len(etf_symbols) * 2))  # Using a colourmap for a variety of colours

# Function to add zero-line and customize the plot
def customize_plot(ax, title, xlabel, ylabel):
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)  # Add a horizontal line at y=0 for reference
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend()

# Compute and plot ACF for the price of each ETF
fig, ax = plt.subplots(figsize=(14, 6))
for i, symbol in enumerate(etf_symbols):
    acf_values = stattools.acf(etf_data[symbol]['4. close'], nlags=80)
    ax.plot(acf_values, label=f'{symbol} Price ACF', color=colors[i], marker='o', markersize=5)
customize_plot(ax, 'Price Autocorrelation Function (ACF) for ETFs', 'Lag', 'Autocorrelation')

plt.savefig('ACFprice.png')

# Compute and plot PACF for the price of each ETF
fig, ax = plt.subplots(figsize=(14, 6))
for i, symbol in enumerate(etf_symbols):
    pacf_values = stattools.pacf(etf_data[symbol]['4. close'], method='ols', nlags=80)
    ax.plot(pacf_values, label=f'{symbol} Price PACF', color=colors[len(etf_symbols) + i], marker='^', markersize=5)
customize_plot(ax, 'Price Partial Autocorrelation Function (PACF) for ETFs', 'Lag', 'Partial Autocorrelation')

plt.savefig('PACFprice.png')

# Compute and plot ACF for the returns of each ETF
fig, ax = plt.subplots(figsize=(14, 6))
for i, symbol in enumerate(etf_symbols):
    acf_values = stattools.acf(etf_data[symbol]['Log_Return'].dropna(), nlags=80)
    ax.plot(acf_values, label=f'{symbol} Return ACF', color=colors[i], marker='s', markersize=5)
customize_plot(ax, 'Return Autocorrelation Function (ACF) for ETFs', 'Lag', 'Autocorrelation')

plt.savefig('ACFreturns.png')

# Compute and plot PACF for the returns of each ETF
fig, ax = plt.subplots(figsize=(14, 6))
for i, symbol in enumerate(etf_symbols):
    pacf_values = stattools.pacf(etf_data[symbol]['Log_Return'].dropna(), method='ols', nlags=80)
    ax.plot(pacf_values, label=f'{symbol} Return PACF', color=colors[len(etf_symbols) + i], marker='x', markersize=5)
customize_plot(ax, 'Return Partial Autocorrelation Function (PACF) for ETFs', 'Lag', 'Partial Autocorrelation')

plt.savefig('PACFreturns.png')

plt.show()

## 17 - 18 Performing a Gaussianity test of the return time series

# seed the random number generator
seed(1)


for symbol in etf_symbols:
    # Drop NaN values from the returns
    data = etf_data[symbol]['Log_Return'].dropna()

    # Calculate the mean and standard deviation of the returns
    mean, std = data.mean(), data.std()

    # Generate a normal distribution with the same mean and std as the data
    norm_dist = norm(loc=mean, scale=std)

    # Perform the K-S test
    stat, p = kstest(data, norm_dist.cdf)
    # compared with ETF returns distribution function found empirically
    # note norm_dist cdf is a highly accurate numerical representation of the
    # normal cdf

    print(f'{symbol}: Statistics={stat:.3f}, p={p:.3f}')

    # Interpret the results
    confidence_level = 0.95
    if p > 1 - confidence_level:
        print(f'{symbol} Sample looks Gaussian (fail to reject H0)')
    else:
        print(f'{symbol} Sample does not look Gaussian (reject H0)')

## 19-20 Stationary test


# We use the adfuller function from stattools to do the ADF test:

for symbol in etf_symbols:
    # Perform ADF test on the price time series
    result = stattools.adfuller(etf_data[symbol]['4. close'].dropna())

    print(f'{symbol} ADF Statistic: {result[0]:.3f}')
    print(f'{symbol} p-value: {result[1]:.3f}')

    # Interpret the results
    alpha = 0.95
    if result[1] > 1 - alpha:
        print(f'{symbol} price time series is non-stationary (fail to reject H0) since there is a unit root')
    else:
        print(f'{symbol} price series is stationary (reject H0) since there is no unit root')

for symbol in etf_symbols:
    # Perform ADF test on the return time series
    result = stattools.adfuller(etf_data[symbol]['Log_Return'].dropna())

    print(f'{symbol} ADF Statistic: {result[0]:.3f}')
    print(f'{symbol} p-value: {result[1]:.3f}')

    # Interpret the results
    alpha = 0.95
    if result[1] > 1 - alpha:
        print(f'{symbol} return time series is non-stationary (fail to reject H0) since there is a unit root')
    else:
        print(f'{symbol} return series is stationary (reject H0) since there is no unit root')

## 21-23 Cointegration test for price and return series

# y1 and y2 are pandas Series of ETF prices SPY and QQQ respectively
y1price = etf_data['SPY']['4. close']
y2price = etf_data['QQQ']['4. close']
y1return = etf_data['SPY']['Log_Return'].dropna()
y2return = etf_data['QQQ']['Log_Return'].dropna()

# Add a constant term for the intercept
y2price_const = sm.add_constant(y2price)
y2return_const = sm.add_constant(y2return)

model_for_price = sm.OLS(y1price, y2price_const)
results_price = model_for_price.fit()
model_for_return = sm.OLS(y1return, y2return_const)
results_return = model_for_return.fit()

# The coefficient θ
theta_for_price = results_price.params[1]
theta_for_return = results_return.params[1]

zt_for_price = y1price - theta_for_price * y2price
zt_for_return = y1return - theta_for_return * y2return

# Perform ADF test on the return time series
adf_result_price = stattools.adfuller(zt_for_price)
adf_result_return = stattools.adfuller(zt_for_return)

print(f'Price Series Residual ADF Statistic: {adf_result_price[0]:.3f}')
print(f'p-value of Price Series Residual: {adf_result_price[1]:.3f}')
print(f'Return Series Residual ADF Statistic: {adf_result_return[0]:.3f}')
print(f'p-value of Return Series Residual: {adf_result_return[1]:.3f}')

# Interpret the results
confidence_level = 0.95
if adf_result_price[1] > 1 - confidence_level:
    print(f'Price Series Residual is non-stationary (fail to reject H0) so SPY and QQQ prices are not cointegrated')
else:
    print(f'Price Series Residual is stationary (reject H0) so SPY and QQQ prices are cointegrated')

if adf_result_return[1] > 1 - confidence_level:
    print(f'Return Series Residual is non-stationary (fail to reject H0) so SPY and QQQ returns are not cointegrated')
else:
    print(f'Return Series Residual is stationary (reject H0) so SPY and QQQ returns are cointegrated')
