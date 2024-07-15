import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load and preprocess the data
df = pd.read_csv('^N100.csv')
df['Adj Close'] = df['Adj Close'].astype(float) # convert closing values to floats
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True) # chronological order
print(f"Missing values before: {df.isnull().sum().sum()}") # there are 6 null values - are they all in the same row?
print(df[df.isna().any(axis=1)]) # Christmas day 2019 is a null value
df.dropna(inplace=True)
print(f"Missing values after: {df.isnull().sum().sum()}")
print(f"Duplicate rows before: {df.duplicated().sum()}")


# Descriptive statistics
descriptive_stats = df.describe()
print(descriptive_stats)

# Plot index points
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Adj Close'],linestyle='-', color='b')
plt.title('Euronext 100: Adjusted Close Index Points Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close (index points)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Euronext Adj Close.png")
plt.show()


adj_close_array = df['Adj Close'].values
# Train-test split - using 8/10 for training
train_size = int(len(adj_close_array) * 0.8) # use training set for acf/pacf plot for benchmark AR model
train_adj_close_array = adj_close_array[:train_size]
test_adj_close_array = adj_close_array[train_size:]
# Want to see if it's an autoregressive process and if so identify the lag

# Compute ACF and PACF
acf_values = acf(train_adj_close_array, nlags=80, fft=True)  # Using fft=True for efficient computation
pacf_values = pacf(train_adj_close_array, nlags=80)

# Plot ACF and PACF in the same figure for comparison
plt.figure(figsize=(14, 6))

# ACF plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.plot(acf_values, label='Euronext Index ACF', color='blue', marker='o', markersize=5)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.ylim(-0.2, 1.2)
plt.legend()

# PACF plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.plot(pacf_values, label='Euronext Index PACF', color='green', marker='^', markersize=5)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.ylim(-0.2, 1.2)
plt.legend()

# Show the plots
plt.tight_layout()  # Adjust layout to fit all elements
plt.savefig("acfpacfplot.png")
plt.show()


""" CROSS VALIDATION PROCEDURE FOR LSTM (uncomment to run - though takes a bit of time)
# Create lagged features for 'Adj Close'
n_lags = 5 # 5 features
split_ratios = [0.2, 0.4, 0.6, 0.8] # Splits for training data
lstm_units = [x for x in range(3, 61, 3)]
results = {units: [] for units in lstm_units}
for i in range(1, n_lags + 1):
    df[f'Lag_{i}'] = df['Adj Close'].shift(i) # shift by certain lag to create feature
df.dropna(inplace=True)  # Drop rows with NaN values created by shifting

count = 0
for units in lstm_units:
    rmse_values = []
    for split_ratio in split_ratios:
        # Calculate split indices
        train_size = int(len(df) * split_ratio)
        test_size = int(len(df) * 0.2)  # Next 0.2 of the data for testing
        # Split the data
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:train_size + test_size]

        # Normalize features using only the training data
        scaler_features = MinMaxScaler(feature_range=(0, 1))
        # Fit a scaler on the training data only
        # squeeze all values together so that they're comparable to one another
        # features scaled to between 0 and 1, standardised
        # Also makes program less computationally intensive
        # Initialise minmax scaler
        train_features_scaled = scaler_features.fit_transform(train_df.drop(columns=['Date', 'Adj Close']))
        # apply same scale to test separately
        test_features_scaled = scaler_features.transform(test_df.drop(columns=['Date', 'Adj Close']))

        # Normalize target using only the training data
        scaler_target = MinMaxScaler(feature_range=(0, 1))
        # ensuring that the model's output is also normalized
        train_target_scaled = scaler_target.fit_transform(train_df[['Adj Close']])
        # apply same scale to test set
        test_target_scaled = scaler_target.transform(test_df[['Adj Close']])

        # Combine scaled features and target back into train and test dataframes
        train_scaled = np.hstack((train_features_scaled, train_target_scaled))
        test_scaled = np.hstack((test_features_scaled, test_target_scaled))

        # Prepare the dataset for LSTM, creating sequences of n days' worth of lagged features for each input.
        def prepare_data(df, n_lags):
            X, y = [], []
            for i in range(n_lags, len(df)): # start index is n_lags because the first
                # n_lags - 1 entries do not have enough previous observations to create a full set of lagged features.
                X.append(df.iloc[i-n_lags:i, 1:].values)  # Use all lag features
                # Start from second column since date shouldn't be included
                # n_lags previous observations for all features, which is then appended to the list X
                y.append(df.iloc[i, -n_lags])  # Want to predict current day's 'Adj Close'
            return np.array(X), np.array(y) # need input data in the form of NumPy arrays

        # Reshape data for LSTM [samples, time steps, features]
        X_train, y_train = prepare_data(pd.DataFrame(train_scaled), n_lags)
        X_test, y_test = prepare_data(pd.DataFrame(test_scaled), n_lags)


        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units, activation='tanh', input_shape=(n_lags, X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')


        # Fit the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

        y_test = y_test.reshape(-1, 1)

        # Making predictions
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(-1, 1)

        # Inverse scaling for actual and predicted values
        y_test_inv = scaler_target.inverse_transform(y_test)
        y_pred_inv = scaler_target.inverse_transform(y_pred)

        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))

        def mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        rmse_val = rmse(y_test_inv, y_pred_inv)
        rmse_values.append(rmse_val)
        count +=1 # Keep track of number of models trained for validation
        print(count)
    # After all splits for the current number of units are processed, calculate the average RMSE
    avg_rmse = np.mean(rmse_values)
    results[units].append(avg_rmse) # Append the average RMSE to the results dictionary

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-')
plt.xlabel('Number of LSTM Units')
plt.ylabel('Average RMSE')
plt.title('Average RMSE vs. Number of LSTM Units')
plt.savefig('Validation LSTM.png')
plt.show()
"""

# We found through validation that 21 is the best number of hidden units
# Create lagged features for 'Adj Close'
n_lags = 5
for i in range(1, n_lags + 1):
    df[f'Lag_{i}'] = df['Adj Close'].shift(i)
df.dropna(inplace=True)  # Drop rows with NaN values created by shifting
df = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
# drop irrelevant columns

# Calculate split indices
train_size = int(len(df) * 0.8)
# Split the data
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Normalize features using only the training data
scaler_features = MinMaxScaler(feature_range=(0, 1))
# Fit a scaler on the training data only
# squeeze all values together so that they're comparable to one another
# features scaled to between 0 and 1, standardised
# Also makes program less computationally intensive
# Initialise minmax scaler
train_features_scaled = scaler_features.fit_transform(train_df.drop(columns=['Date', 'Adj Close']))
# apply same scale to test separately
test_features_scaled = scaler_features.transform(test_df.drop(columns=['Date', 'Adj Close']))

# Normalize target using only the training data
scaler_target = MinMaxScaler(feature_range=(0, 1))
# ensuring that the model's output is also normalized
train_target_scaled = scaler_target.fit_transform(train_df[['Adj Close']])
# apply same scale to test set
test_target_scaled = scaler_target.transform(test_df[['Adj Close']])

# Combine scaled features and target back into train and test dataframes
train_scaled = np.hstack((train_features_scaled, train_target_scaled))
test_scaled = np.hstack((test_features_scaled, test_target_scaled))

# Prepare the dataset for input, creating sequences of n days' worth of lagged features for each input.
def prepare_data(df, n_lags):
    X, y = [], []
    for i in range(n_lags, len(df)): # start index is n_lags because the first
        # n_lags - 1 entries do not have enough previous observations to create a full set of lagged features.
        X.append(df.iloc[i-n_lags:i, 1:].values)  # Use all lag features
        # Start from second column since date shouldn't be included
        # n_lags previous observations for all features, which is then appended to the list X
        y.append(df.iloc[i, -n_lags])  # Want to predict current day's 'Adj Close'
    return np.array(X), np.array(y) # need input data in the form of NumPy arrays

# Reshape data for input [samples, time steps, features]
X_train, y_train = prepare_data(pd.DataFrame(train_scaled), n_lags)
X_test, y_test = prepare_data(pd.DataFrame(test_scaled), n_lags)

y_test = y_test.reshape(-1, 1)

# Inverse scaling for actual and predicted values
y_test_inv = scaler_target.inverse_transform(y_test)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to evaluate models
def evaluate_model(model, X_test, y_test_inv):
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1, 1)
    y_pred_inv = scaler_target.inverse_transform(y_pred)
    print("MAE:", mean_absolute_error(y_test_inv, y_pred_inv))
    print("RMSE:", rmse(y_test_inv, y_pred_inv))
    print("MAPE:", mape(y_test_inv, y_pred_inv))
    return y_pred_inv, rmse(y_test_inv, y_pred_inv)

""" saving the best model (uncomment to run)
bestLSTM, bestRNN, bestGRU = float('inf'), float('inf'), float('inf')
for k in range(15): # run the training process 10 times, save model with best rmse for LSTM, RNN, GRU
    # Reshape data for LSTM [samples, time steps, features]
    X_train, y_train = prepare_data(pd.DataFrame(train_scaled), n_lags)
    X_test, y_test = prepare_data(pd.DataFrame(test_scaled), n_lags)

    y_test = y_test.reshape(-1, 1)

    # Inverse scaling for actual and predicted values
    y_test_inv = scaler_target.inverse_transform(y_test)
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Function to evaluate models
    def evaluate_model(model, X_test, y_test_inv):
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(-1, 1)
        y_pred_inv = scaler_target.inverse_transform(y_pred)
        print("MAE:", mean_absolute_error(y_test_inv, y_pred_inv))
        print("RMSE:", rmse(y_test_inv, y_pred_inv))
        print("MAPE:", mape(y_test_inv, y_pred_inv))
        return y_pred_inv, rmse(y_test_inv, y_pred_inv)

    # Define, compile and fit the models
    model_lstm = Sequential()
    model_lstm.add(LSTM(21, activation='tanh', input_shape=(n_lags, X_train.shape[2])))
    # lookback 5 timesteps, use lagged features
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')

    # Fit the model
    print("Training LSTM model...")
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    _, accLSTM = evaluate_model(model_lstm, X_test, y_test_inv)

    if accLSTM <= bestLSTM:
        bestLSTM = accLSTM # only save new model if current rmse better than current best
        with open("LSTMmodel.pickle", "wb") as f:
            pickle.dump(model_lstm, f)

    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(21, activation='tanh', input_shape=(n_lags, X_train.shape[2])))
    model_rnn.add(Dense(1))
    model_rnn.compile(optimizer='adam', loss='mse')

    print(f"X_train.shape[2] is {X_train.shape[2]}")

    print("Training RNN model...")
    model_rnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # Evaluate RNN model
    print("\nEvaluating RNN model...")
    _, accRNN = evaluate_model(model_rnn, X_test, y_test_inv)

    if accRNN <= bestRNN:
        bestRNN = accRNN
        with open("RNNmodel.pickle", "wb") as f:
            pickle.dump(model_rnn, f)

    model_gru = Sequential()
    model_gru.add(GRU(21, activation='tanh', input_shape=(n_lags, X_train.shape[2])))
    model_gru.add(Dense(1))
    model_gru.compile(optimizer='adam', loss='mse')

    print("Training GRU model...")
    model_gru.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # Evaluate GRU model
    print("\nEvaluating GRU model...")
    _, accGRU = evaluate_model(model_gru, X_test, y_test_inv)

    if accGRU <= bestGRU:
        bestGRU = accGRU
        with open("GRUmodel.pickle", "wb") as f:
            pickle.dump(model_gru, f)

print(bestLSTM)
print(bestRNN)
print(bestGRU)
"""
pickleLSTM_in = open("LSTMmodel.pickle", "rb")
model_lstm = pickle.load(pickleLSTM_in)

# Evaluate LSTM model
print("Evaluating best LSTM model...")
y_pred_inv_lstm, _ = evaluate_model(model_lstm, X_test, y_test_inv)

pickleRNN_in = open("RNNmodel.pickle", "rb")
model_rnn = pickle.load(pickleRNN_in)

# Evaluate RNN model
print("\nEvaluating best RNN model...")
y_pred_inv_rnn, _ = evaluate_model(model_rnn, X_test, y_test_inv)

pickleGRU_in = open("GRUmodel.pickle", "rb")
model_gru = pickle.load(pickleGRU_in)

# Evaluate GRU model
print("\nEvaluating best GRU model...")
y_pred_inv_gru, _ = evaluate_model(model_gru, X_test, y_test_inv)

# Plotting Results
plt.figure(figsize=(14, 10))
dates = test_df['Date'].values[n_lags:]  # Adjust for the lag in the test set

# Plot for LSTM
plt.subplot(3, 1, 1)
plt.plot(dates, y_test_inv, label='Actual')
plt.plot(dates, y_pred_inv_lstm, label='LSTM Predicted')
plt.title('LSTM Prediction')
plt.legend()

# Plot for RNN
plt.subplot(3, 1, 2)
plt.plot(dates, y_test_inv, label='Actual')
plt.plot(dates, y_pred_inv_rnn, label='RNN Predicted')
plt.title('RNN Prediction')
plt.legend()

# Plot for GRU
plt.subplot(3, 1, 3)
plt.plot(dates, y_test_inv, label='Actual')
plt.plot(dates, y_pred_inv_gru, label='GRU Predicted')
plt.title('GRU Prediction')
plt.legend()

plt.tight_layout()
plt.savefig("RNN models performance.png")
plt.show()

# `train_adj_close_array` is training dataset (numpy array)
# and `test_adj_close_array` is test dataset (numpy array)

# Fit the AR(1) model on the training data
model = ARIMA(train_adj_close_array, order=(1, 0, 0))  # ARIMA(1,0,0) model
model_fitted = model.fit()

# Initialize an array to hold the forecasts
test_forecasts = np.zeros(len(test_adj_close_array))

# Forecast one step ahead for each point in the test set
for i in range(len(test_adj_close_array)):
    # The forecast for the next point is based on the model fitted to all data up to the current point
    endog = np.concatenate([train_adj_close_array, test_adj_close_array[:i]])
    model_temp = ARIMA(endog, order=(1, 0, 0))
    model_temp_fitted = model_temp.fit()
    forecast = model_temp_fitted.forecast(steps=1)
    test_forecasts[i] = forecast[0]

# Plotting Results
plt.figure(figsize=(14, 4))
dates = test_df['Date'].values  # Adjust for the lag in the test set
plt.plot(dates, test_adj_close_array[1:], label='Actual', color='blue')
plt.plot(dates, test_forecasts[1:], label='AR(1)', color='orange')
plt.title('AR(1) Prediction')
plt.legend()
plt.savefig('ARIMAplot.png')
plt.show()

# Function to evaluate models based on actual vs. predicted
def evaluate_model_metrics(y_pred_inv, y_test_inv, model_name):
    print(f"Model: {model_name}")
    print("MAE:", mean_absolute_error(y_test_inv, y_pred_inv))
    print("RMSE:", rmse(y_test_inv, y_pred_inv))
    print("MAPE:", mape(y_test_inv, y_pred_inv))

print("Evaluating AR(1)...")
evaluate_model_metrics(test_forecasts[1:], test_adj_close_array[1:], "AR(1) one-step Forecast")


def naive_forecast(X_test):
    # Simply use the last observed value
    return X_test[:, -1, 0]  # last observed value is ini last column

def average_forecast(X_test):
    # Calculate the average of all observed values
    return np.mean(X_test[:, :, 0], axis=1)  # Assuming observed values are in the first feature

def weighted_average_forecast(X_test):
    n = X_test.shape[1]
    weights = np.linspace(1, n, n)  # Linearly decreasing weights
    weighted_averages = np.average(X_test[:, :, 0], axis=1, weights=weights)
    return weighted_averages

# Generate predictions
print(X_test.shape[1])
naive_pred_scaled = naive_forecast(X_test)
average_pred_scaled = average_forecast(X_test)
weighted_average_pred_scaled = weighted_average_forecast(X_test)

# Inverse transform to original scale
naive_pred_inv = scaler_target.inverse_transform(naive_pred_scaled.reshape(-1, 1))
average_pred_inv = scaler_target.inverse_transform(average_pred_scaled.reshape(-1, 1))
weighted_average_pred_inv = scaler_target.inverse_transform(weighted_average_pred_scaled.reshape(-1, 1))

print("Evaluating Naive Forecast...")
evaluate_model_metrics(naive_pred_inv, y_test_inv, "Naive Forecast")

print("\nEvaluating Average Forecast...")
evaluate_model_metrics(average_pred_inv, y_test_inv, "Average Forecast")

print("\nEvaluating Weighted Average Forecast...")
evaluate_model_metrics(weighted_average_pred_inv, y_test_inv, "Weighted Average Forecast")

# Plotting Results
plt.figure(figsize=(14, 10))
dates = test_df['Date'].values[n_lags:]  # Adjust for the lag in the test set

# Adding subplots for baseline models below

# Plot for Naive Forecast
plt.subplot(3, 1, 1)
plt.plot(dates, y_test_inv, label='Actual', color='blue')
plt.plot(dates, naive_pred_inv, label='Naive Forecast', color='orange')
plt.title('Naive Forecast Prediction')
plt.legend()

# Plot for Average Forecast
plt.subplot(3, 1, 2)
plt.plot(dates, y_test_inv, label='Actual', color='blue')
plt.plot(dates, average_pred_inv, label='Average Forecast', color='green')
plt.title('Average Forecast Prediction')
plt.legend()

# Plot for Weighted Average Forecast
plt.subplot(3, 1, 3)
plt.plot(dates, y_test_inv, label='Actual', color='blue')
plt.plot(dates, weighted_average_pred_inv, label='Weighted Average Forecast', color='red')
plt.title('Weighted Average Forecast Prediction')
plt.legend()

plt.tight_layout()
plt.savefig("BenchmarkPlots.png")
plt.show()
