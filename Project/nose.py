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

# Load data
incddf = pd.read_csv(r"C:\Users\hsain\Downloads\incd.csv", encoding='latin1')
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
incddf = incddf[~incddf.isin(['*', '**', '', '_', '__']).any(axis=1)]
# Reset the index after dropping rows
incddf.reset_index(drop=True, inplace=True)
# Convert 'FIPS' column to integers
incddf['FIPS'] = incddf['FIPS'].astype(int)
# Convert integers to strings and pad with leading zeros
incddf['FIPS'] = incddf['FIPS'].apply(lambda x: str(x).zfill(5))

# Drop columns and rename them
incddf.drop(incddf.columns[[3, 4, 6, 7, 8, 9]].values, axis=1, inplace=True)

incddf.rename(columns={incddf.columns[0]: 'Location',
                       incddf.columns[2]: 'Incidence_Rate',
                       incddf.columns[3]: 'Avg_Ann_Incidence'}, inplace=True)
# Remove " #" suffix from the values in the specified column
incddf['Incidence_Rate'] = incddf['Incidence_Rate'].str.replace(' #', '')

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
# Drop rows with missing or non-numeric values in the target variable 'Incidence_Rate'
fulldf.dropna(subset=['Incidence_Rate'], inplace=True)
fulldf = fulldf[fulldf['Incidence_Rate'].str.replace('.', '', 1).str.isdigit()]
# Convert the 'Incident Rate' column to float type
fulldf['Incidence_Rate'] = fulldf['Incidence_Rate'].astype(float)

# print(fulldf.head())
# fulldf.to_excel('merged_data.xlsx', index=False)

# Define the columns for X and y
cols = ['Heavy Drinking Percentage', 'Air Pollution', 'Population',
        'MEDIAN INCOME', 'Employed', 'Bachelors or higher']

# Extract X and y from the dataframe
X = fulldf[cols]
y = fulldf['Incidence_Rate']

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

# Linear Regression
# Perform cross-validation for linear regression
cv_scores_linear = cross_val_score(linreg, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared (Linear Regression):", cv_scores_linear)

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
plt.xlabel('Actual Incidence Rate')
plt.ylabel('Predicted Incidence Rate')
plt.title('Actual vs Predicted Incidence Rate')
plt.legend()
plt.show()


# Create a DataFrame containing the actual and predicted values
predictions_df = pd.DataFrame({
    'Actual_Mortality_Rate': y_test,
    'Predicted_Mortality_Rate_LinearRegression': y_pred})

# Specify the path where you want to save the Excel file
output_file = 'predictions.xlsx'

# Export predictions DataFrame to Excel
# predictions_df.to_excel(output_file, index=False)
