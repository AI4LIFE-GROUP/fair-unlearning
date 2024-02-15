import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file =  'datasets/HSLS/hsls_df_knn_impute_past_v2.pkl'

df = pd.read_pickle(file)

## Setting NaNs to out-of-range entries
## entries with values smaller than -7 are set as NaNs
df[df <= -7] = np.nan

## Dropping all rows or columns with missing values
## this step significantly reduces the number of samples
df = df.dropna()

## Creating racebin & gradebin & sexbin variables
## X1SEX: 1 -- Male, 2 -- Female, -9 -- NaN -> Preprocess it to: 0 -- Female, 1 -- Male, drop NaN
## X1RACE: 0 -- BHN, 1 -- WA
df['gradebin'] = df['grade9thbin']
df['racebin'] = np.logical_or(((df['studentrace']*7).astype(int)==7).values, ((df['studentrace']*7).astype(int)==1).values).astype(int)
df['sexbin'] = df['studentgender'].astype(int)


## Dropping race and 12th grade data just to focus on the 9th grade prediction ##
df = df.drop(columns=['studentgender', 'grade9thbin', 'grade12thbin', 'studentrace'])

## Scaling ##
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

## Balancing data to have roughly equal race=0 and race =1 ##
# df = balance_data(df, group_feature)

df.to_csv("datasets/HSLS/hsls_cleaned.csv")