# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


file = 'datasets/COMPAS/compas-scores-two-years.csv'
df = pd.read_csv(file,index_col=0)

# select features for analysis
df = df[['age', 'c_charge_degree', 'race',  'sex', 'priors_count', 
            'days_b_screening_arrest',  'is_recid',  'c_jail_in', 'c_jail_out']]

# drop missing/bad features (following ProPublica's analysis)
# ix is the index of variables we want to keep.

# Remove entries with inconsistent arrest information.
ix = df['days_b_screening_arrest'] <= 30
ix = (df['days_b_screening_arrest'] >= -30) & ix

# remove entries entries where compas case could not be found.
ix = (df['is_recid'] != -1) & ix

# remove traffic offenses.
ix = (df['c_charge_degree'] != "O") & ix


# trim dataset
df = df.loc[ix,:]

# create new attribute "length of stay" with total jail time.
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)


# drop 'c_jail_in' and 'c_jail_out'
# drop columns that won't be used
dropCol = ['c_jail_in', 'c_jail_out','days_b_screening_arrest']
df.drop(dropCol,inplace=True,axis=1)

# keep only African-American and Caucasian
df = df.loc[df['race'].isin(['African-American','Caucasian']),:]

# binarize race 
# African-American: 0, Caucasian: 1
df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='Caucasian' else 0)

# binarize gender
# Female: 1, Male: 0
df.loc[:,'sex'] = df['sex'].apply(lambda x: 1 if x=='Male' else 0)

# rename columns 'sex' to 'gender'
df.rename(index=str, columns={"sex": "gender"},inplace=True)

# binarize degree charged
# Misd. = -1, Felony = 1
df.loc[:,'c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 1 if x=='F' else -1)
        
# reset index
df.reset_index(inplace=True,drop=True)

train_idx, test_idx = train_test_split(df.index, test_size=0.2)

df["dataset"] = "train"
df.loc[test_idx, "dataset"] = "test"

df.to_csv("datasets/COMPAS/compas_cleaned_labeled.csv")

## generate unlearning requests. minority = race == 1, which is African American, race == 0 is Caucasian
df_train = df.loc[df["dataset"] == "train"].reset_index()
minority_indices = np.random.permutation(df_train.loc[df_train["race"] == 0].index.to_numpy())
majority_indices = np.random.permutation(df_train.loc[df_train["race"] == 1].index.to_numpy())

np.save("./minority_requests.npy", minority_indices)
np.save("./majority_requests.npy", majority_indices)