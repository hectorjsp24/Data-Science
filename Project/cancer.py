from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Load data
incddf = pd.read_excel(
    r'C:\Users\hsain\Downloads\NEW DATA\Incidence.xlsx', sheet_name='Incidence', usecols=['County', 'FIPS', 'Age-Adjusted Incidence Rate([rate note]) - cases per 100,000', 'Average Annual Count'], dtype={'FIPS': str})
alcdf = pd.read_excel(
    r'C:\Users\hsain\Downloads\NEW DATA\Alcohol.xlsx', sheet_name='Heavy', usecols=['Location', '2012 Both Sexes', 'State'])
airdf = pd.read_excel(
    r'C:\Users\hsain\Downloads\NEW DATA\Air Quality.xlsx', sheet_name='County Factbook 2022', usecols=['FIPS', 'PM2.5'], dtype={'FIPS': str})
empdf = pd.read_excel(r'C:\Users\hsain\Downloads\NEW DATA\Unemployment.xlsx',
                      sheet_name='UnemploymentMedianIncome', usecols=['FIPS', 'Employed_2021', 'Unemployed_2021'], dtype={'FIPS': str})
povdf = pd.read_excel(r'C:\Users\hsain\Downloads\NEW DATA\PovertyEstimates.xlsx', sheet_name='PovertyEstimates', usecols=[
                      'FIPS_Code', 'POVALL_2021', 'MEDHHINC_2021'], dtype={'FIPS_Code': str})
edudf = pd.read_excel(r'C:\Users\hsain\Downloads\NEW DATA\Education.xlsx', sheet_name='Education 1970 to 2021', usecols=[
                      'FIPS', 'Less than a high school diploma, 2017-21', 'High school diploma only, 2017-21', 'Some college or associates degree, 2017-21', 'Bachelors degree or higher, 2017-21'], dtype={'FIPS': str})
popdf = pd.read_excel(
    r'C:\Users\hsain\Downloads\NEW DATA\PopulationEstimates.xlsx', sheet_name='Population', usecols=['FIPS', 'POP_ESTIMATE_2021'], dtype={'FIPS': str})

# INC DATA SET
# Drop rows containing the specified values
incddf = incddf[~incddf.isin(['*', '* ', '**', '', '_', '__']).any(axis=1)]
# Reset the index after dropping rows
incddf.reset_index(drop=True, inplace=True)

incddf.rename(columns={incddf.columns[0]: 'Location',
                       incddf.columns[2]: 'Incidence_Rate',
                       incddf.columns[3]: 'Avg_Ann_Incidence'}, inplace=True)

# ALCOHOL DATA SET
# Rename column for alcohol data
alcdf.rename(columns={'2012 Both Sexes': 'Heavy Drinking Percentage'},
             inplace=True)
# Eliminate States data from the dataset
alcdf = alcdf[alcdf['Location'] != alcdf['State']]
alcdf = alcdf.drop(columns=['State'])
alcdf.sort_values(by='Heavy Drinking Percentage',
                  ascending=False, inplace=True)
alcdf.drop_duplicates(subset='Location', keep='first', inplace=True)

# AIR QUALITY DATA SET
airdf.rename(columns={'PM2.5': 'Air Pollution'}, inplace=True)

# UNEMPLOYMENT DATA SET
empdf.rename(columns={'Employed_2021': 'Employed',
             'Unemployed_2021': 'Unemployed'}, inplace=True)

# POVERTY DATA SET
povdf.rename(columns={'POVALL_2021': 'POVERTY',
             'MEDHHINC_2021': 'MEDIAN INCOME', 'FIPS_Code': 'FIPS'}, inplace=True)

# EDUCATION DATA SET
edudf.rename(columns={'Less than a high school diploma, 2017-21': 'Less than High School', 'High school diploma only, 2017-21': 'Only High School',
             'Some college or associates degree, 2017-21': 'Some College', 'Bachelors degree or higher, 2017-21': 'Bachelors or higher'}, inplace=True)

# POPULATION DATA SET
popdf.rename(columns={'POP_ESTIMATE_2021': 'Population'}, inplace=True)

mergdf = incddf.merge(alcdf, how='inner', on='Location')

# MERGE THE DATA
fulldf = mergdf.merge(airdf, how='outer', on='FIPS') \
    .merge(popdf, how='outer', on='FIPS') \
    .merge(empdf, how='outer', on='FIPS') \
    .merge(povdf, how='outer', on='FIPS') \
    .merge(edudf, how='outer', on='FIPS')

fulldf.dropna(inplace=True)

# PART 2
# Define the columns for X and y

# Convert the 'Incident Rate' column to float type
fulldf['Incidence_Rate'] = fulldf['Incidence_Rate'].astype(float)

# Load data (Assuming data loading code remains unchanged)
# Assuming fulldf is already defined

# Define the columns for X and y
cols = ['Heavy Drinking Percentage',
        'Air Pollution', 'MEDIAN INCOME', 'Population']

# Extract X and y from the dataframe
X = fulldf[cols]
y = fulldf['Incidence_Rate']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Instantiate the Gradient Boosting Regression model
gb_regressor = GradientBoostingRegressor(random_state=42)

# Train the model
gb_regressor.fit(X_train, y_train)

# Make predictions
y_pred = gb_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Gradient Boosting Regression):", mse)

# Create a DataFrame containing the actual and predicted values
predictions_df = pd.DataFrame({
    'Actual_Incidence_Rate': y_test,
    'Predicted_Incidence_Rate_GradientBoosting': y_pred
})
# Filter out data points where both actual and predicted incidence rates are below 350
filtered_predictions_df = predictions_df[(predictions_df['Actual_Incidence_Rate'] >= 350) & (
    predictions_df['Predicted_Incidence_Rate_GradientBoosting'] >= 350)]

# Plot the actual vs predicted values after filtering
plt.figure(figsize=(10, 6))
plt.scatter(filtered_predictions_df['Actual_Incidence_Rate'], filtered_predictions_df['Predicted_Incidence_Rate_GradientBoosting'], color='blue',
            label='Actual vs Predicted (Filtered)', alpha=0.3)
# Adjust the range of x and y values as needed
plt.plot([350, 550], [350, 550], color='red', linestyle='--', lw=2)
plt.xlabel('Actual Incidence Rate')
plt.ylabel('Predicted Incidence Rate')
plt.title(
    'Actual vs Predicted Incidence Rate (Gradient Boosting Regression) - Filtered')
plt.legend()
plt.show()


# Specify the path where you want to save the Excel file
output_file = 'predictions_gradient_boosting.xlsx'

# Export predictions DataFrame to Excel
# predictions_df.to_excel(output_file, index=False)
