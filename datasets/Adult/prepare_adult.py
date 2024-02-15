import pandas as pd
import numpy as np

## Prepare Adult dataset
# After cleaning, 30940 train samples, 15507 test samples

file = 'datasets/Adult/adult.data'
fileTest = 'datasets/Adult/adult.test'

df = pd.read_csv(file, header=None,sep=',\s+',engine='python')
dfTest = pd.read_csv(fileTest,header=None,skiprows=1,sep=',\s+',engine='python') 


columnNames = ["age", "workclass", "fnlwgt", "education", "education-num",
"marital-status", "occupation", "relationship", "race", "gender",
"capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

df.columns = columnNames
dfTest.columns = columnNames

df = df.append(dfTest)

# drop columns that won't be used
dropCol = ["fnlwgt","workclass","occupation"]
df.drop(dropCol,inplace=True,axis=1)

# keep only entries marked as ``White'' or ``Black''
ix = df['race'].isin(['White','Black'])
df = df.loc[ix,:]

# binarize race
# Black = 0; White = 1
df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='White' else 0)

# binarize gender
# Female = 0; Male = 1
df.loc[:,'gender'] = df['gender'].apply(lambda x: 1 if x=='Male' else 0)

# binarize income
# '>50k' = 1; '<=50k' = 0
df.loc[:,'income'] = df['income'].apply(lambda x: 1 if x[0]=='>' else 0)


# drop "education" and native-country (education already encoded in education-num)
features_to_drop = ["education","native-country"]
df.drop(features_to_drop,inplace=True,axis=1)



# create one-hot encoding
categorical_features = list(set(df)-set(df._get_numeric_data().columns))
df = pd.concat([df,pd.get_dummies(df[categorical_features])],axis=1,sort=False)
df.drop(categorical_features,inplace=True,axis=1)

print(df.shape)
# reset index
df.reset_index(inplace=True,drop=True)
df.to_csv("datasets/Adult/adult_cleaned.csv")