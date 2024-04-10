from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load mortality data from a CSV file
mortdf = pd.read_csv(r"C:\Users\hsain\Downloads\death.csv", encoding='latin1')
# Load incidence data from a CSV file
incddf = pd.read_csv(r"C:\Users\hsain\Downloads\incd.csv", encoding='latin1')

# Drop rows containing the specified values
mortdf = mortdf[~mortdf.isin(['*', '**', '']).any(axis=1)]
incddf = incddf[~incddf.isin(['*', '**', '']).any(axis=1)]
# Reset the index after dropping rows
mortdf.reset_index(drop=True, inplace=True)
incddf.reset_index(drop=True, inplace=True)
# Convert 'FIPS' column to integers
mortdf['FIPS'] = mortdf['FIPS'].astype(int)
incddf['FIPS'] = incddf['FIPS'].astype(int)
# Convert integers to strings and pad with leading zeros
mortdf['FIPS'] = mortdf['FIPS'].apply(lambda x: str(x).zfill(5))
incddf['FIPS'] = incddf['FIPS'].apply(lambda x: str(x).zfill(5))

# Drop columns and rename them
incddf.drop(incddf.columns[[0, 3, 4, 7, 8, 9]].values, axis=1, inplace=True)
mortdf.drop(mortdf.columns[[0, 2, 4, 5, 7, 8, 9, 10]], axis=1, inplace=True)

mortdf.rename(columns={mortdf.columns[1]: 'Mortality_Rate',
                       mortdf.columns[2]: 'Avg_Ann_Deaths'}, inplace=True)
incddf.rename(columns={incddf.columns[1]: 'Incidence_Rate',
                       incddf.columns[2]: 'Avg_Ann_Incidence'}, inplace=True)

# Create new column for merge
incddf['StateFIPS'] = incddf['FIPS'].str[:2]

# POVERTY DATA
# Specify the columns to be read from the CSV file
colspov = ['State', 'StateFIPS', 'AreaName',
           'B17001_002', 'B17001_003', 'B17001_017']

# Load the dataset
povdf = pd.read_csv(
    r"C:\Users\hsain\Downloads\poverty.csv", usecols=colspov)

# Remove rows with StateFIPS code '72'
povdf = povdf.drop(povdf[povdf.StateFIPS > 70].index)

# Add leading zeros to the state FIPS codes
povdf['StateFIPS'] = povdf.StateFIPS.astype(str).str.zfill(2)

# Rename columns
povdf.rename(columns={'B17001_002': 'All_Poverty', 'B17001_003': 'M_Poverty', 'B17001_017': 'F_Poverty'},
             inplace=True)

# INCOME DATA
# Specify the columns to be read from the CSV file
colsinc = ['StateFIPS', 'B19013_001', 'B19013A_001',
           'B19013B_001', 'B19013C_001', 'B19013D_001', 'B19013I_001']

# Load the dataset
incomedf = pd.read_csv(
    r"C:\Users\hsain\Downloads\income.csv", usecols=colsinc)

# Remove rows with StateFIPS code '72'
incomedf = incomedf.drop(incomedf[incomedf.StateFIPS > 70].index)

# Add leading zeros to the state FIPS codes
incomedf['StateFIPS'] = incomedf.StateFIPS.astype(str).str.zfill(2)

# Rename columns
incomedf.rename(columns={'B19013_001': 'Med_Income', 'B19013A_001': 'Med_Income_White',
                         'B19013B_001': 'Med_Income_Black', 'B19013C_001': 'Med_Income_Nat_Am',
                         'B19013D_001': 'Med_Income_Asian', 'B19013I_001': 'Hispanic'}, inplace=True)

# HEALTH INSURANCE DATA
colshins = ['StateFIPS', 'B27001_004', 'B27001_005', 'B27001_007', 'B27001_008',
            'B27001_010', 'B27001_011', 'B27001_013', 'B27001_014',
            'B27001_016', 'B27001_017', 'B27001_019', 'B27001_020',
            'B27001_022', 'B27001_023', 'B27001_025', 'B27001_026',
            'B27001_028', 'B27001_029', 'B27001_032', 'B27001_033',
            'B27001_035', 'B27001_036', 'B27001_038', 'B27001_039',
            'B27001_041', 'B27001_042', 'B27001_044', 'B27001_045',
            'B27001_047', 'B27001_048', 'B27001_050', 'B27001_051',
            'B27001_053', 'B27001_054', 'B27001_056', 'B27001_057']

hinsdf = pd.read_csv(
    r"C:\Users\hsain\Downloads\insurance.csv", usecols=colshins)

# Remove rows with StateFIPS code '72'
hinsdf = hinsdf.drop(hinsdf[hinsdf.StateFIPS > 70].index)

# Add leading zeros to the state FIPS codes
hinsdf['StateFIPS'] = hinsdf.StateFIPS.astype(str).str.zfill(2)

colspop = ['StateFIPS', 'POPESTIMATE2015']
popdf = pd.read_csv(
    r"C:\Users\hsain\Downloads\population.csv", usecols=colspop)
popdf['StateFIPS'] = popdf.StateFIPS.astype(str).str.zfill(2)

# columns representing males' health insurance statistics
males = ['B27001_004', 'B27001_005', 'B27001_007', 'B27001_008',
         'B27001_010', 'B27001_011', 'B27001_013', 'B27001_014',
         'B27001_016', 'B27001_017', 'B27001_019', 'B27001_020',
         'B27001_022', 'B27001_023', 'B27001_025', 'B27001_026',
         'B27001_028', 'B27001_029']

# females' health insurance statistics
females = ['B27001_032', 'B27001_033', 'B27001_035', 'B27001_036',
           'B27001_038', 'B27001_039', 'B27001_041', 'B27001_042',
           'B27001_044', 'B27001_045', 'B27001_047', 'B27001_048',
           'B27001_050', 'B27001_051', 'B27001_053', 'B27001_054',
           'B27001_056', 'B27001_057']

# separate the "with" and "without" health insurance columns
males_with = []
males_without = []
females_with = []
females_without = []

# strip the backticks
# i would be the index of the column
# j would be the actual value
for i, j in enumerate(males):
    if i % 2 == 0:
        males_with.append(j)
    else:
        males_without.append(j)

for i, j in enumerate(females):
    if i % 2 == 0:
        females_with.append(j)
    else:
        females_without.append(j)

# Create features that sum all the individual age group
newcols = ['M_With', 'M_Without', 'F_With', 'F_Without']

# create the new columns in the dataset and add the values
for col in newcols:
    hinsdf[col] = 0

for i in males_with:
    hinsdf['M_With'] += hinsdf[i]
for i in males_without:
    hinsdf['M_Without'] += hinsdf[i]
for i in females_with:
    hinsdf['F_With'] += hinsdf[i]
for i in females_without:
    hinsdf['F_Without'] += hinsdf[i]
# create two new columns adding the genders divided by with and without
hinsdf['All_With'] = hinsdf.M_With + hinsdf.F_With
hinsdf['All_Without'] = hinsdf.M_Without + hinsdf.F_Without

# Remove all the individual age group variables
# but, save them as a df called hinsdf_extra (just in case)
hinsdf_extra = hinsdf.loc[:,
                          hinsdf.columns[hinsdf.columns.str.contains('B27001')].values]
hinsdf.drop(hinsdf.columns[hinsdf.columns.str.contains(
    'B27001')].values, axis=1, inplace=True)

# Merge incddf and mortdf on StateFIPS
mergdf = incddf.merge(mortdf, how='outer', on='FIPS')

# Merge merged_inc_mort_df with povdf, incomedf, and hinsdf
fulldf = mergdf.merge(povdf, how='outer', on='StateFIPS') \
    .merge(incomedf, how='outer', on='StateFIPS') \
    .merge(hinsdf, how='outer', on='StateFIPS') \
    .merge(popdf, how='outer', on='StateFIPS')

# Replace 'stable' with 1, 'falling' with 2, and 'rising' with 3
fulldf['Recent Trend'] = fulldf['Recent Trend'].replace(
    {'stable': 1, 'falling': 2, 'rising': 3})

# Drop columns cause of missing values
fulldf.drop(['Med_Income_White', 'Med_Income_Black', 'Med_Income_Nat_Am',
             'Med_Income_Asian', 'Hispanic', 'State', 'AreaName'], axis=1, inplace=True)
fulldf.dropna(inplace=True)

# Filter out rows containing a number followed by " # "
fulldf = fulldf[~fulldf['Incidence_Rate'].str.contains(
    r' #  ', na=False)]

# Create a boolean mask indicating whether any value in each row contains "_"
fulldf = fulldf[~fulldf.isin(['_', '__']).any(axis=1)]

# Remove commas from numeric columns
fulldf.replace(',', '', regex=True, inplace=True)

# Convert numeric columns to float
fulldf = fulldf.astype(float)

# PART 2
# Define the columns for X and y
cols = ['Incidence_Rate', 'All_Poverty', 'Med_Income',
        'All_With', 'All_Without', 'POPESTIMATE2015',]

# Extract X and y from the dataframe
X = fulldf[cols]
y = fulldf['Mortality_Rate']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Instantiate the linear regression model
linreg = LinearRegression()

# Train the model
linreg.fit(X_train, y_train)

# Make predictions
y_pred = linreg.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Linear Regression):", mse)

# Random Forest: Instantiate the Random Forest Regressor
# Adjust hyperparameters as needed
rf_regressor = RandomForestRegressor(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 55, 60],  # Number of trees in the forest
    'max_depth': [18, 20, 22],      # Maximum depth of the tree
    # Minimum number of samples required to split an internal node
    'min_samples_split': [2, 3, 4],
    # Add more parameters as needed
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
print(best_params)
# Retrain the model on the complete training data using the best parameters
best_estimator.fit(X_train, y_train)

# Make Predictions
y_pred_rf = best_estimator.predict(X_test)

# Evaluate the Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Mean Squared Error (Random Forest):", mse_rf)

# Linear Regression
# Perform cross-validation for linear regression
cv_scores_linear = cross_val_score(linreg, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared (Linear Regression):", cv_scores_linear)

# Random Forest
# Perform cross-validation for random forest
cv_scores_rf = cross_val_score(best_estimator, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared (Random Forest):", cv_scores_rf)

coefficients = linreg.coef_
feature_names = X.columns
coefficients_df = pd.DataFrame(
    {'Feature': feature_names, 'Coefficient': coefficients})
# Sort the coefficients by absolute value to identify the most influential features
coefficients_df = coefficients_df.reindex(
    coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
print(coefficients_df)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue',
            label='Actual vs Predicted', alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
         y_test.max()], color='red', linestyle='--', lw=2)
plt.xlabel('Actual Mortality Rate')
plt.ylabel('Predicted Mortality Rate')
plt.title('Actual vs Predicted Mortality Rate')
plt.legend()
plt.show()

# Create a DataFrame containing the actual and predicted values
predictions_df = pd.DataFrame({
    'Actual_Mortality_Rate': y_test,
    'Predicted_Mortality_Rate_LinearRegression': y_pred,
    'Predicted_Mortality_Rate_RandomForest': y_pred_rf
})

# Specify the path where you want to save the Excel file
# output_file = 'predictions.xlsx'

# Export predictions DataFrame to Excel
# predictions_df.to_excel(output_file, index=False)

# print("Predictions exported to", output_file)
