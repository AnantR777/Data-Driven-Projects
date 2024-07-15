import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score

df = pd.read_csv("bankPortfolios.csv")
print(df.columns)

# Define the new column names based on the provided image
new_column_names = [
    'Loans for construction and land development',
    'Loans secured by farmland',
    'Loans secured by 1-4 family residential properties',
    'Loans secured by multi-family (> 5) residential properties',
    'Loans secured by non-farm non-residential properties',
    'Agricultural loans',
    'Commercial and industrial loans',
    'Loans to individuals',
    'All other loans (excluding consumer loans)',
    'Obligations (other than securities and leases) of states and political subdivision in the U.S.',
    'Held-to-maturity securities',
    'Available-for-sale securities, total',
    'Premises and fixed assets including capitalized lease',
    'Cash',
    'Bank\'s debt',
    'Default'
]

# Rename the columns of the dataframe
df.columns = new_column_names

# Checking for duplicate rows
duplicate_rows = df[df.duplicated()]

duplicate_rows_count = duplicate_rows.shape[0]  # Gives the count of duplicate rows

# Check the number of duplicate rows
print(f"Number of duplicate rows before: {duplicate_rows_count}")
print(f"The duplicate row is {duplicate_rows}")
df = df.drop_duplicates()
# Checking for duplicate rows after
duplicate_rows = df[df.duplicated()]
duplicate_rows_count = duplicate_rows.shape[0]
print(f"Number of duplicate rows after: {duplicate_rows_count}")

# None missing
print(f"Number of missing values: {df.isnull().sum().sum()}")

# First, we calculate the sum of all asset categories (columns 1 to 14)
df['Total Assets'] = df.iloc[:, 0:14].sum(axis=1) # not to be used as feature

# Now we calculate each of the required ratios

# Loan-to-Asset Ratio: sum of all loan categories (columns 1 to 9) divided by the sum of all asset categories
df['Loan-to-Asset Ratio'] = df.iloc[:, 0:9].sum(axis=1) / df['Total Assets']

# Debt-to-Assets Ratio: bank debt (column 15) divided by the sum of all asset categories
df["Debt-to-Assets Ratio"] = df.iloc[:, 14] / df['Total Assets']

# Securities to Assets Ratio: sum of held-to-maturity securities (column 11) and available-for-sale securities (column 12),
# then divided by the sum of all asset categories
df['Securities-to-Assets Ratio'] = (df.iloc[:, 10] + df.iloc[:, 11]) / df['Total Assets']

# Liquidity Ratio: Cash (column 14) divided by the sum of all asset categories
df['Liquidity Ratio'] = df.iloc[:, 13] / df['Total Assets']

# Display the first few rows to confirm the new columns
print(df[['Loan-to-Asset Ratio', 'Debt-to-Assets Ratio', 'Securities-to-Assets Ratio', 'Liquidity Ratio']].head())

selected_columns = df[['Loan-to-Asset Ratio', 'Debt-to-Assets Ratio', 'Securities-to-Assets Ratio', 'Liquidity Ratio']]
# Add a constant column for the intercept
X = selected_columns.assign(const=1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
# drop column with highest VIF
df = df.drop('Loan-to-Asset Ratio', axis=1)

# No longer has loan-to-asset ratio
selected_columns = df[['Debt-to-Assets Ratio', 'Securities-to-Assets Ratio', 'Liquidity Ratio']]

# Add a constant column for the intercept
X = selected_columns.assign(const=1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# Next, undersample the majority class
# First, count the instances of the majority and minority classes
count_class_0, count_class_1 = df.Default.value_counts()

# Divide the dataframe by class
df_class_0 = df[df['Default'] == 0]
df_class_1 = df[df['Default'] == 1]

# Perform undersampling of the majority class
df_class_0_under = df_class_0.sample(count_class_1)
df = pd.concat([df_class_0_under, df_class_1], axis=0)

# Check the balance of the classes now
print(df.Default.value_counts())
print(df.columns)

# Prepare the feature matrix (X) and the target variable (y)
X = df[['Debt-to-Assets Ratio', 'Securities-to-Assets Ratio', 'Liquidity Ratio']]
y = df['Default']

# Define the scoring metrics
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'auc': make_scorer(roc_auc_score, needs_proba=True)
}

# Define the parameter grid for C (regularization strength)
# np.logspace generates numbers evenly spaced on a log scale, a common practice for regularization strength
param_grid = {
    'C': np.logspace(-3, 3, 20)  # Exploring a wide range of C
}

# Create a scorer for AUC
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Set up the grid search with 6-fold cross-validation
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=6, scoring=auc_scorer)

# Fit the grid search model
grid_search.fit(X, y)

# Best C value
best_C = grid_search.best_params_['C']
print(f"Best C: {best_C}")

# Get the results into a DataFrame for plotting
results = pd.DataFrame(grid_search.cv_results_)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.semilogx(results['param_C'], results['mean_test_score'], '-o')  # Use a log scale for C
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Mean Test AUC')
plt.title('Logistic Regression: Tuning C')
plt.grid(True)
#plt.savefig("LRhyperparam tuning.png")
plt.show()

# Define the best Logistic Regression model
best_lr = LogisticRegression(solver='liblinear', C=best_C)

# Perform 6-fold cross-validation using cross_validate with the defined scoring metrics
cv_results = cross_validate(best_lr, X, y, cv=6, scoring=scoring_metrics)

# Calculate the average scores for each metric
average_scores = {metric: np.mean(scores) for metric, scores in cv_results.items() if 'test_' in metric}

# Print the average scores from 6-fold cross-validation for the best model
print("Average scores from 6-fold cross-validation for the best Logistic Regression model:")
for metric, average_score in average_scores.items():
    print(f"{metric}: {average_score}")

# Specify the kernels for the SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Dictionary to store the best model results for each kernel
best_models = {}

# Loop through each kernel and perform grid search
for kernel in kernels:
    print(f"Processing {kernel} kernel")
    svm = SVC(kernel=kernel, probability=True)
    param_grid = {'C': np.logspace(-3, 3, 20)}

    grid_search = GridSearchCV(svm, param_grid, cv=6, scoring='roc_auc')
    grid_search.fit(X, y)

    # Plotting the hyperparameter tuning results
    plt.figure(figsize=(8, 6))
    plt.semilogx(grid_search.cv_results_['param_C'].data.astype(float),
                 grid_search.cv_results_['mean_test_score'], '-o')
    plt.title(f'Hyperparameter Tuning for SVM ({kernel} kernel)')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Mean Test AUC')
    plt.grid(True)
    #plt.savefig(f"{kernel}-hyperparam tuning.png")
    plt.show()

    best_C = grid_search.best_params_['C']
    best_score = grid_search.best_score_
    print(f"Best C for {kernel} kernel: {best_C}, with score: {best_score}")

    # Store the best model
    best_models[kernel] = grid_search.best_estimator_

# Assuming 'rbf' kernel's model was the best, just as an example
best_kernel = 'rbf'  # replace with actual best kernel from your results
best_model = best_models[best_kernel]

# Re-evaluate the selected best model using 6-fold cross-validation to get detailed metrics
cv_results = cross_validate(best_model, X, y, cv=6, scoring=scoring_metrics)

# Calculate the average scores for each metric
average_scores = {metric: np.mean(scores) for metric, scores in cv_results.items() if 'test_' in metric}

# Print the average scores for the best SVM model
print(f"Average scores from 6-fold cross-validation for the best SVM model ({best_kernel} kernel):")
for metric, average_score in average_scores.items():
    print(f"{metric}: {average_score}")

# Define the model
dt = DecisionTreeClassifier()

# Define the parameter grid to search over for 'max_depth'
param_grid = {
    'max_depth': np.arange(1, 13),  # Exploring depths from 1 to 20
}

# Create a scorer for AUC
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Set up the grid search with 6-fold cross-validation
grid_search = GridSearchCV(dt, param_grid, cv=6, scoring=auc_scorer)

# Fit the grid search model
grid_search.fit(X, y)

# Get the results into a DataFrame
best_depth = grid_search.best_params_['max_depth']
print(f"Best max_depth: {best_depth}")
results = pd.DataFrame(grid_search.cv_results_)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(results['param_max_depth'], results['mean_test_score'], '-o')
plt.xlabel('Max Depth')
plt.ylabel('Mean Test AUC')
plt.title('Decision Tree: Tuning Max Depth')
plt.xticks(np.arange(1, 13))
plt.grid(True)
#plt.savefig("hyperparamtuningDT.png")
plt.show()

# Define the best model
best_dt = DecisionTreeClassifier(max_depth=best_depth)

# Perform 6-fold cross-validation using cross_validate with the defined scoring metrics
cv_results = cross_validate(best_dt, X, y, cv=6, scoring=scoring_metrics)

# Calculate the average scores for each metric
average_scores = {metric: np.mean(scores) for metric, scores in cv_results.items() if 'test_' in metric}

# Print the average scores from 6-fold cross-validation for the best model
print("Average scores from 6-fold cross-validation for the best Decision Tree model:")
for metric, average_score in average_scores.items():
    print(f"{metric}: {average_score}")

# Define the parameter grid
param_grid = {
    'n_neighbors': np.arange(1, 20)  # Exploring a range of neighbors
}

# Create a scorer for AUC
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Set up the grid search with 6-fold cross-validation
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=6, scoring=auc_scorer)

# Fit the grid search model
grid_search.fit(X, y)

# Best number of neighbors
best_n_neighbors = grid_search.best_params_['n_neighbors']
print(f"Best n_neighbors: {best_n_neighbors}")

# Get the results into a DataFrame for plotting
results = pd.DataFrame(grid_search.cv_results_)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], '-o')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Test AUC')
plt.title('k-NN: Tuning n_neighbors')
plt.xticks(np.arange(1, 21, 2))
plt.grid(True)
#plt.savefig("kNNhyperparamtuning.png")
plt.show()

# Define the best k-NN model
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)

# Perform 6-fold cross-validation using cross_validate with the defined scoring metrics
cv_results = cross_validate(best_knn, X, y, cv=6, scoring=scoring_metrics)

# Calculate the average scores for each metric
average_scores = {metric: np.mean(scores) for metric, scores in cv_results.items() if 'test_' in metric}

# Print the average scores from 6-fold cross-validation for the best model
print("Average scores from 6-fold cross-validation for the best k-NN model:")
for metric, average_score in average_scores.items():
    print(f"{metric}: {average_score}")
