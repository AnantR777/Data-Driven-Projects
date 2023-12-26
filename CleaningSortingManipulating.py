import os
import pandas as pd
import matplotlib.pyplot as plt

folder_path = 'C:/Users/Anantajit/PycharmProjects/ElecDemandPrediction'
df_concatenated = pd.read_csv("historic_demand_year_2009.csv")

# concatenates excel sheets (lining up column headings)
for file in os.listdir(folder_path):
    if file.startswith('historic_demand_year_201') or file.startswith('historic_demand_year_202'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.upper() # column heading concatenation is case sensitive
        df_concatenated = pd.concat([df_concatenated, df], ignore_index=True)

# cleaning
# replace all missing values in these columns with 0, since they should be 0
missing_values_nsl = df_concatenated['NSL_FLOW'].isnull()
missing_values_eleclinkflow = df_concatenated['ELECLINK_FLOW'].isnull()

df_concatenated.loc[missing_values_nsl, 'NSL_FLOW'] = 0
df_concatenated.loc[missing_values_eleclinkflow, 'ELECLINK_FLOW'] = 0

# df_concatenated.isnull().any() can be used to check for any remaining NaN

selected_columns = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']

print(df_concatenated)

df_concatenated['SETTLEMENT_DATE'] = pd.to_datetime(df_concatenated['SETTLEMENT_DATE'])

# Combine 'SETTLEMENT_DATE' and 'SETTLEMENT_PERIOD' into a single datetime column
df_concatenated['SETTLEMENT_TIME'] = df_concatenated['SETTLEMENT_DATE'] + pd.to_timedelta(df_concatenated['SETTLEMENT_PERIOD'] - 1, unit='h')

# Drop the original 'SETTLEMENT_DATE' and 'SETTLEMENT_PERIOD' columns if no longer needed
#df_concatenated.drop(['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'], axis=1, inplace=True)

# Move 'SETTLEMENT_TIME' column to the first column
df_concatenated.insert(0, 'SETTLEMENT_TIME', df_concatenated.pop('SETTLEMENT_TIME'))

# Convert 'READING_TIME' column to datetime (if not already done)
df_concatenated['SETTLEMENT_TIME'] = pd.to_datetime(df_concatenated['SETTLEMENT_TIME'])

# Define a function to plot data
"""def plot_column(df, column_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['SETTLEMENT_TIME'], df[column_name])
    plt.xlabel('Settlement Time')
    plt.ylabel(f'{column_name} Values')
    plt.title(f'{column_name} over Settlement Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

# checking how demand metrics vary with time
# Plot 'ND' against 'READING_TIME'
plot_column(df_concatenated, 'ND')

# Plot 'TSD' against 'READING_TIME'
plot_column(df_concatenated, 'TSD')

# Plot 'ENGLAND_WALES_DEMAND' against 'READING_TIME'
plot_column(df_concatenated, 'ENGLAND_WALES_DEMAND')"""
# Note this plot looks odd so we choose not to use this

# feature engineering
df_concatenated['EMBEDDED_RENEWABLE_GENERATION'] = df_concatenated['EMBEDDED_WIND_GENERATION'] + df_concatenated['EMBEDDED_SOLAR_GENERATION']
df_concatenated['EMBEDDED_RENEWABLE_CAPACITY'] = df_concatenated['EMBEDDED_WIND_CAPACITY'] + df_concatenated['EMBEDDED_SOLAR_CAPACITY']
interconnector_flows = ['IFA_FLOW', 'IFA2_FLOW', 'BRITNED_FLOW', 'MOYLE_FLOW', 'EAST_WEST_FLOW', 'NEMO_FLOW']
df_concatenated['TOTAL_FLOW'] = sum(df_concatenated[i] for i in interconnector_flows)

print(df_concatenated.columns)



