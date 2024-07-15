import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Path to your Excel file
file_path = '48_Industry_Portfolios_daily.csv'


start_row = 24103  # Adjust this based on the actual starting row of data frame

# Read the file
df = pd.read_csv(file_path, skiprows=start_row)

# Convert the 'Unnamed: 0' column to datetime
df['Date'] = pd.to_datetime(df['Unnamed: 0'], format='%Y%m%d')

# Set the converted column as the index
df.set_index('Date', inplace=True)
df.drop(['Unnamed: 0'], axis = 1, inplace=True)

# Filter the DataFrame for years 2008 and 2009
df = df[(df.index.year == 2008) | (df.index.year == 2009)]

# Check the result for the first few rows to verify
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate values: {df.duplicated().sum()}")

# Define a function to calculate the covariance based on the formula provided
def calculate_covariance(data):
    # Ensure all data is numeric and handle non-numeric entries
    data = data.apply(pd.to_numeric, errors='coerce')

    # Calculate the mean returns for each asset
    mean_returns = data.mean()

    # Normalize data by subtracting the mean returns
    normalized_data = data - mean_returns

    # Compute the covariance matrix
    covariance_matrix = normalized_data.T.dot(normalized_data) / (len(data))
    # the covariance formula using pairs of assets, done in matrix form

    return covariance_matrix

# Function to calculate the optimal weights using ridge regularization
def calculate_optimal_weights(cov_matrix, lambda_):
    regularized_cov_inv = np.linalg.inv(cov_matrix + lambda_ * np.identity(len(cov_matrix)))
    weights_numerators = regularized_cov_inv.sum(axis=1)
    weights_denominator = regularized_cov_inv.sum()
    optimal_weights = weights_numerators / weights_denominator
    return optimal_weights


def calculate_optimal_weights_no_reg_no_short(cov_matrix):
    n_assets = len(cov_matrix)

    # Objective function for portfolio variance
    def objective(weights):
        return weights.T @ cov_matrix @ weights

    # Initial guess
    initial_weights = np.ones(n_assets) / n_assets

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]  # Sum to 1
    bounds = [(0, 1) for _ in range(n_assets)]  # No shorting

    # Optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                      constraints=constraints)

    if not result.success:
        raise ValueError('Optimisation Failed')

    return result.x

# Assume df is loaded dataframe containing the equity returns
# Calculate the indices for splitting the data
total_length = len(df)
split_indices = [int(total_length * 0.25), int(total_length * 0.5),
                 int(total_length * 0.75)]

# Set up the lambda range
lambda_range = np.arange(0, 10.01, 0.01)  # Lambda values from 0 to 10 in increments of 0.01


# Loop over each fold
for i, split_index in enumerate(split_indices):
    # size of the test set should be 0.25 of the total data length
    test_size = split_indices[0] if i == 0 else split_index - split_indices[
        i - 1]

    # Split the data into training and test sets
    train_set = df.iloc[:split_index]
    test_set = df.iloc[split_index:split_index + test_size]

    # Calculate covariance matrix for the training set
    train_cov_matrix = calculate_covariance(train_set)
    train_cov_matrix = train_cov_matrix.astype(float)

    # Calculate covariance matrix for the test set
    test_cov_matrix = calculate_covariance(test_set)
    test_cov_matrix = test_cov_matrix.astype(float)

    # Calculate the optimal weights for the unregularized portfolio (lambda = 0)
    unreg_optimal_weights = calculate_optimal_weights(train_cov_matrix.values, 0)
    unreg_in_sample_risk = unreg_optimal_weights.T.dot(train_cov_matrix).dot(
            unreg_optimal_weights)
    unreg_out_of_sample_risk = unreg_optimal_weights.T.dot(test_cov_matrix).dot(
        unreg_optimal_weights)
    # Optimising weights without regularization and with no shorting allowed
    optimal_weights_no_reg_no_shorting = calculate_optimal_weights_no_reg_no_short(
        train_cov_matrix.values)
    in_sample_risk_no_reg_no_shorting = optimal_weights_no_reg_no_shorting.T.dot(train_cov_matrix).dot(
            optimal_weights_no_reg_no_shorting)
    out_of_sample_risk_no_reg_no_shorting = optimal_weights_no_reg_no_shorting.T.dot(test_cov_matrix).dot(
            optimal_weights_no_reg_no_shorting)

    # Initialize lists to store risks and optimal lambda values for each fold
    in_sample_risks = []
    out_of_sample_risks = []
    equally_weighted_variances_in = []
    # Initialise a list to store the in-sample risks for the equally weighted portfolio
    equally_weighted_variances_out = []

    # Iterate over lambda values
    for lambda_value in lambda_range:
        # Calculate the optimal weights for the current lambda
        optimal_weights = calculate_optimal_weights(train_cov_matrix.values,
                                                    lambda_value)
        # Calculate the in-sample risk for the training set
        in_sample_risk = optimal_weights.T.dot(train_cov_matrix).dot(
            optimal_weights)
        # Calculate the out-of-sample risk for the test set
        out_of_sample_risk = optimal_weights.T.dot(test_cov_matrix).dot(
            optimal_weights)

        # Store the risks
        in_sample_risks.append(in_sample_risk)
        out_of_sample_risks.append(out_of_sample_risk)

    # Calculate the variance for an equally weighted portfolio using the training set covariance matrix
    n_assets = train_cov_matrix.shape[0]
    equal_weights = np.full(n_assets, 1 / n_assets)
    eq_weighted_in_sample_risk = equal_weights.T.dot(train_cov_matrix).dot(
        equal_weights)
    eq_weighted_out_sample_risk = equal_weights.T.dot(test_cov_matrix).dot(
        equal_weights)


    # Find the optimal lambda for the minimum out-of-sample risk
    min_risk_index = np.argmin(out_of_sample_risks)
    optimal_lambda = lambda_range[min_risk_index]
    min_out_of_sample_risk = out_of_sample_risks[min_risk_index]

    # Calculate the optimal weights using the optimal lambda
    optimal_weights = calculate_optimal_weights(train_cov_matrix.values,
                                                optimal_lambda)

    # Plot the optimal weights for each asset as a two-way bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    assets = train_cov_matrix.columns
    plt.bar(assets, optimal_weights,
            color=['red' if x < 0 else 'green' for x in optimal_weights])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylabel('Optimal Weights')
    plt.title(f'Ridge: Optimal Weights for Fold {i + 1} with Lambda {optimal_lambda:.2f}')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.savefig(f"RidgeOptimalWeightsFold{i+1}.png")
    plt.show()

    # Plot the optimal weights for each asset as a two-way bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    assets = train_cov_matrix.columns
    plt.bar(assets, unreg_optimal_weights,
            color=['red' if x < 0 else 'green' for x in unreg_optimal_weights])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylabel('Weights')
    plt.title(
        f'No regularisation: Weights for Fold {i + 1}')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.savefig(f"NoRegWeightsFold{i + 1}.png")
    plt.show()

    # Plot the optimal weights for each asset as a two-way bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    assets = train_cov_matrix.columns
    plt.bar(assets, optimal_weights_no_reg_no_shorting,
            color=['green' for x in optimal_weights_no_reg_no_shorting])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylabel('Weights')
    plt.title(
        f'No regularisation, no shorting: Weights for Fold {i + 1}')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.savefig(f"NoRegNoShortWeightsFold{i + 1}.png")
    plt.show()

    # Print the results for the current fold
    print(f"Fold {i + 1}:")
    print(f"Unregularised In-sample risk: {unreg_in_sample_risk}")
    print(f"Unregularised Out-of-sample risk: {unreg_out_of_sample_risk}")
    print(f"Ridge: Optimal lambda for minimum out-of-sample risk: {optimal_lambda}")
    print(f"Ridge: Minimum out-of-sample risk: {min_out_of_sample_risk}")
    # Print the equally weighted portfolio variance for the current fold
    print(f"Unregularised, No Shorting, In-sample Risk: {in_sample_risk_no_reg_no_shorting}")
    print(f"Unregularised, No Shorting, Out-of-sample Risk: {out_of_sample_risk_no_reg_no_shorting}")
    print(f"Equally weighted portfolio In-sample risk: {eq_weighted_in_sample_risk}")
    print(f"Equally weighted portfolio Out-of-sample risk: {eq_weighted_out_sample_risk}")

    # Plot in-sample risks vs lambda values
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, in_sample_risks, label='In-sample risk')
    # Mark the first point
    plt.scatter(lambda_range[0], in_sample_risks[0], color='red',
                label='Unregularised', zorder=5)
    plt.xlabel('Lambda')
    plt.ylabel('Risk')
    plt.legend()
    plt.title(f'Ridge: In-Sample Risks for Fold {i + 1}')
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.savefig(f"RidgeISRisksFold{i + 1}.png")
    plt.show()

    # Plot out-of-sample risks vs lambda values
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, out_of_sample_risks, label='Out-of-sample risk')
    # Mark the first point
    plt.scatter(lambda_range[0], out_of_sample_risks[0], color='red',
                label='Unregularised', zorder=5)
    plt.xlabel('Lambda')
    plt.ylabel('Risk')
    plt.legend()
    plt.title(f'Ridge: Out-of-Sample Risks for Fold {i + 1}')
    plt.xlim(0,10)
    plt.tight_layout()
    plt.savefig(f"RidgeOOSRisksFold{i + 1}.png")
    plt.show()
