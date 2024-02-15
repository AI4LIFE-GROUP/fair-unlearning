import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def load_data(name, returndf=False, generate_val=False, private_embeddings=False):
    if name == "COMPAS":
        if private_embeddings:
            embeddings_file = 'datasets/COMPAS/lbfgs_mlp_extracted.pth'
            data = torch.load(embeddings_file)
            X_train = data['X_train'].squeeze().cpu()
            y_train = data['y_train'].cpu()
            X_test = data['X_test'].squeeze().cpu()
            y_test = data['y_test'].cpu()
            S_train = data["S_train"].cpu().reshape(-1,1)
            S_test = data["S_test"].cpu().reshape(-1,1)

            X_train = X_train/np.max(np.linalg.norm(X_train, axis=1))
            X_test = X_test/np.max(np.linalg.norm(X_train, axis=1))

            traindf = pd.DataFrame(X_train)
            traindf['race'] = S_train
            traindf['is_recid'] = y_train

            testdf = pd.DataFrame(X_test)
            testdf['race'] = S_test
            testdf['is_recid'] = y_test
            return traindf, testdf, 'is_recid'
        #make loaders for compas
        file = 'datasets/COMPAS/compas_cleaned_labeled.csv'
        # file = '../../datasets/COMPAS/compas_cleaned_labeled.csv'

        df = pd.read_csv(file,index_col=0)

        if generate_val:
            traintestdf, valdf = train_test_split(df.drop(["dataset"],axis=1), test_size=0.2, random_state=123)
            traindf, testdf = train_test_split(traintestdf, test_size=0.25)
        else:
            traindf, testdf = train_test_split(df.drop(["dataset"],axis=1), test_size=0.2)


        X_train = traindf.drop(['is_recid'], axis=1).values.astype(np.float32)
        y_train = traindf['is_recid'].values.astype(np.int64)
        X_test = testdf.drop(['is_recid'], axis=1).values.astype(np.float32)
        y_test = testdf['is_recid'].values.astype(np.int64)

        maxnorm = np.max(np.linalg.norm(X_train, axis=1))
        
        if returndf:
            traindf.loc[:, traindf.columns != 'is_recid'] = traindf.loc[:, traindf.columns != 'is_recid']/maxnorm
            testdf.loc[:, testdf.columns != 'is_recid'] = testdf.loc[:, testdf.columns != 'is_recid']/maxnorm
            if generate_val:
                valdf.loc[:, valdf.columns != 'is_recid'] = valdf.loc[:, valdf.columns != 'is_recid']/maxnorm
                return traindf, valdf, 'is_recid'

            return traindf, testdf, 'is_recid'
        return X_train, y_train, X_test, y_test
   
    elif name == "Adult":
        if private_embeddings:
            embeddings_file = 'datasets/Adult/lbfgs3_mlp_extracted.pth'
            data = torch.load(embeddings_file)
            X_train = data['X_train'].squeeze().cpu()
            y_train = data['y_train'].cpu()
            X_test = data['X_test'].squeeze().cpu()
            y_test = data['y_test'].cpu()
            S_train = data["S_train"].cpu()
            S_test = data["S_test"].cpu()

            X_train = X_train/np.max(np.linalg.norm(X_train, axis=1))
            X_test = X_test/np.max(np.linalg.norm(X_train, axis=1))

            traindf = pd.DataFrame(X_train)
            traindf['race'] = S_train
            traindf['income'] = y_train

            testdf = pd.DataFrame(X_test)
            testdf['race'] = S_test
            testdf['income'] = y_test
            return traindf, testdf, 'income'
        file = 'datasets/Adult/adult_cleaned.csv'
        # file = "adult_cleaned.csv"
        df = pd.read_csv(file, index_col=0)

        if generate_val:
            traintestdf, valdf = train_test_split(df, test_size=0.2, random_state=123)
            traindf, testdf = train_test_split(traintestdf, test_size=0.25)
        else:
            traindf, testdf = train_test_split(df, test_size=0.2)

        ## Fixme
        X = df.drop(['income'], axis=1).values.astype(np.float32)
        y = df['income'].values.astype(np.int64)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

            # print(np.max(np.linalg.norm(X_train, axis=1)))
                    
        scaler = StandardScaler()
        scaler.fit(X_train)

        # print(np.linalg.norm(X_train, axis=1).shape)
        maxnorm = np.max(np.linalg.norm(scaler.transform(X_train), axis=1))

        if returndf:
            # ## normalize to control newton step
            traindf.loc[:, traindf.columns != 'income'] = scaler.transform(traindf.loc[:, traindf.columns != 'income'].values)/maxnorm
            testdf.loc[:, testdf.columns != 'income'] = scaler.transform(testdf.loc[:, testdf.columns != 'income'].values)/maxnorm
            if generate_val:
                valdf.loc[:, valdf.columns != 'income'] = scaler.transform(valdf.loc[:, valdf.columns != 'income'].values)/maxnorm
                return traindf, valdf, 'income'

            return traindf, testdf, 'income'       
        return X_train, Y_train, X_test, Y_test
    elif name == "HSLS":
        if private_embeddings:
            embeddings_file = 'datasets/HSLS/lbfgs_mlp_extracted.pth'
            data = torch.load(embeddings_file)
            X_train = data['X_train'].squeeze().cpu()
            y_train = data['y_train'].cpu()
            X_test = data['X_test'].squeeze().cpu()
            y_test = data['y_test'].cpu()
            S_train = data["S_train"].cpu()
            S_test = data["S_test"].cpu()

            X_train = X_train/np.max(np.linalg.norm(X_train, axis=1))
            X_test = X_test/np.max(np.linalg.norm(X_train, axis=1))

            traindf = pd.DataFrame(X_train)
            traindf['racebin'] = S_train
            traindf['gradebin'] = y_train

            testdf = pd.DataFrame(X_test)
            testdf['racebin'] = S_test
            testdf['gradebin'] = y_test
            return traindf, testdf, 'gradebin'
        file = 'datasets/HSLS/hsls_cleaned.csv'

        df = pd.read_csv(file, index_col=0)

        if generate_val:
            traintestdf, valdf = train_test_split(df, test_size=0.2, random_state=123)
            traindf, testdf = train_test_split(traintestdf, test_size=0.25)
        else:
            traindf, testdf = train_test_split(df, test_size=0.2)


        X = df.drop(['gradebin'], axis=1).values.astype(np.float32)
        y = df['gradebin'].values.astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        maxnorm = np.max(np.linalg.norm(X_train, axis=1))
        
        if returndf:
            traindf.loc[:, traindf.columns != 'gradebin'] = traindf.loc[:, traindf.columns != 'gradebin']/maxnorm
            testdf.loc[:, testdf.columns != 'gradebin'] = testdf.loc[:, testdf.columns != 'gradebin']/maxnorm
            if generate_val:
                valdf.loc[:, valdf.columns != 'gradebin'] = valdf.loc[:, valdf.columns != 'gradebin']/maxnorm
                return traindf, valdf, 'gradebin'
            return traindf, testdf, 'gradebin'
        
        return X_train, y_train, X_test, y_test
    elif name == "CelebA":
        if private_embeddings:
            embeddings_file = 'datasets/CelebA/extracted_tesla_good.pth'
            data = torch.load(embeddings_file)
            X_train = data['X_train'].squeeze().cpu()
            y_train = data['y_train'].cpu()
            X_test = data['X_test'].squeeze().cpu()
            y_test = data['y_test'].cpu()
            S_train = data["S_train"].cpu()
            S_test = data["S_test"].cpu()

            print(np.max(np.linalg.norm(X_train, axis=1)))

            X_train = X_train/np.max(np.linalg.norm(X_train, axis=1))
            X_test = X_test/np.max(np.linalg.norm(X_train, axis=1))

            traindf = pd.DataFrame(X_train)
            traindf['gender'] = S_train
            traindf['smiling'] = y_train

            testdf = pd.DataFrame(X_test)
            testdf['gender'] = S_test
            testdf['smiling'] = y_test
            return traindf, testdf, 'smiling'