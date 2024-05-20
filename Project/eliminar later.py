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
incddf = incddf[~incddf.isin(['*', '**', '', '_', '__']).any(axis=1)]
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
print(mergdf.head())
print(mergdf['FIPS'].dtype)
