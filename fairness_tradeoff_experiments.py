from fair_retraining import *

import pandas as pd
from tqdm import tqdm
import argparse
import scipy

from aif360.datasets import StandardDataset

from utils import *
from datasets import *
from datasets.Dataloader import load_data
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="Adult",
    help="Dataset name. Requires corresponding dataloader implementation."
)

parser.add_argument(
    "--protected-attribute",
    default=None,
    help="Attribute to test unlearning performance on"    
)

parser.add_argument(
    "--unlearned-attribute",
    default=None,
    help="Attribute to condition unlearning on"
)

parser.add_argument(
    "--value",
    default=1,
    type=int,
    help="Value of attribute to condition unlearning on"
)

parser.add_argument(
    "--fair",
    action='store_true',
    help="Train with fairnes regularizer or not"
)

parser.add_argument(
    "--std",
    default=1.,
    type=float,
    help="standard deviation of noise added during raining process to ensure unlearning"
)

parser.add_argument(
    "--reg",
    default=1.,
    type=float,
    help="L2 regularization"
)

parser.add_argument(
    "--fairreg",
    default=1.,
    type=float,
    help="Fair Regularization"
)

parser.add_argument(
    "--batch_size",
    default=None,
    type=int
)

args = parser.parse_args()
prefix = f"hyperparameter_sweep"

print(args)

print(f"output/fairness_tradeoff_reg_{args.reg:.1e}_fairreg_{args.fairreg:.1e}_{args.dataset}_{args.protected_attribute}_{args.unlearned_attribute}_{args.value}")

traindf, testdf, labelname = load_data(args.dataset, returndf=True)

traindf = traindf.astype(np.float64)
testdf = testdf.astype(np.float64)

privileged_value = traindf[args.protected_attribute].unique()[np.where(traindf[args.protected_attribute].unique()>0)][0]

dataset_train = StandardDataset(traindf, label_name=labelname, favorable_classes=[1],
                                protected_attribute_names=[args.protected_attribute], privileged_classes=[[privileged_value]])
dataset_train.labels = dataset_train.labels.ravel()
dataset_test = StandardDataset(testdf, label_name=labelname, favorable_classes=[1],
                                protected_attribute_names=[args.protected_attribute], privileged_classes=[[privileged_value]])
dataset_test.labels = dataset_test.labels.ravel()

X_train, Y_train, X_test, Y_test = dataset_train.features.astype(np.float64), dataset_train.labels.astype(np.float64), dataset_test.features.astype(np.float64), dataset_test.labels.astype(np.float64)
S_train, S_test = (dataset_train.protected_attributes>0).astype(np.float64), (dataset_test.protected_attributes>0).astype(np.float64)

dataset_test.protected_attributes = (dataset_test.protected_attributes > 0).astype(np.float32)
dataset_train.protected_attributes = (dataset_train.protected_attributes > 0).astype(np.float32)

if args.unlearned_attribute is not None:
    print("Unlearned attribute specified. Current implementation will only unlearn protected attribute based on the value argument.")

    unlearning_indices_train = np.where(dataset_train.protected_attributes == args.value)[0]
    complement_indices_train = np.where(dataset_train.protected_attributes != args.value)[0]

else:
    print("Unlearned attribute not specified. Unlearning uniformly at random.")
    unlearning_indices_train = list(range(1, len(X_train)))

repeats = 5

percentages = X_train.shape[0]/100
remove_sizes = percentages*np.array([1,5])
remove_sizes= np.round(remove_sizes).astype(int).tolist()
temp = remove_sizes.copy()
for remove_size in temp:
    if remove_size > len(unlearning_indices_train):
        remove_sizes.remove(remove_size)

l = len(remove_sizes)

methods = ["fair_unlearning"]

regs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] #[1e8,1e7,1e6,1e5,1e4]


output_dict = construct_output_dict(list(map(lambda x: "{:.0e}".format(x), regs)), l, repeats)

## Compute original classifier
b = np.random.normal(0, args.std, size=(X_train.shape[1]+1, 1))
theta_original = log_exact(X_train,Y_train,S_train,fair=args.fair,b=b,reg=args.reg,fairreg=args.fairreg, batch_size=args.batch_size, verbose=True)

print(theta_original.shape)

## Generate shards and train [shard] number of classifiers

## Precompute inversion matrices for residual update
if not args.fair:

    diagvals = sigmoid(X_inflated@theta_original)
    w = np.multiply(diagvals, (1-diagvals)) #ensure vector, not nx1 matrix for diag
    ## Adding 1e-6 to the diagonal to ensure W_full is nonsingular.
    W_full = np.diag(w)+1e-6*np.eye(len(w))

    Z_full = (X_inflated@theta_original + scipy.linalg.inv(W_full)@(Y_train-sigmoid(X_inflated@theta_original)))
    Hat_full = X_inflated@scipy.linalg.inv(X_inflated.T@W_full@X_inflated + np.eye(len(theta_original))*args.reg)@X_inflated.T@W_full
else:
    Z_full = None
    Hat_full = None
    X_inflated = np.concatenate((X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)), axis=1)

    start = time.time()
    precomputed_fairness_gradient = fair_precompute_summation(X_inflated, Y_train.reshape(-1), S_train.reshape(-1),np.arange(X_inflated.shape[0]), np.arange(X_inflated.shape[0]))
    print("Fairness Precomputation: {}".format(time.time()-start))

requests = []

for i in tqdm(range(len(remove_sizes)),position=0,leave=False, desc="Removal Sizes"):
    k = remove_sizes[i]
    runtimes = {
        "naive_retraining": [],
        "newton": [],
        "fair_unlearning": []
    }
    for j in tqdm(range(repeats),position=1,leave=False, desc="Repeats"):
        print("Removal Size: {} Repeat: {}".format(remove_sizes[i], j))
        requests.append(k)

        indices_to_remove = np.random.choice(unlearning_indices_train, k, replace=False)
        # indices_to_remove = np.random.choice(outliers, k, replace=False)
        indices_to_keep = [i for i in range(len(X_train)) if i not in indices_to_remove]
        X_retrain = X_train[indices_to_keep]
        Y_retrain = Y_train[indices_to_keep]
        S_retrain = S_train[indices_to_keep]

        dataset_retrain = StandardDataset(traindf.iloc[indices_to_keep], label_name=labelname, favorable_classes=[1],
                                protected_attribute_names=[args.protected_attribute], privileged_classes=[[privileged_value]])

        if len(np.unique(Y_retrain)) < 2: #if there arent 2 classes to separate anymore
            print("Unlearned so much there aren't 2 classes to separate anymore. Stopping early")
            break

        if len(np.unique(S_retrain)) < 2:
            print("Unlearned so much there aren't 2 subgroups to compute fairness over. Stopping early")
            break

        
        
        # theta_original = log_exact(X_train,Y_train,S_train,fair=args.fair,b=b,reg=args.reg,fairreg=lam, verbose=True)
        
        for penalty in regs:
            b = np.random.normal(0, args.std, size=(X_train.shape[1]+1, 1))
            theta_original = log_exact(X_train,Y_train,S_train,fair=args.fair,b=b,reg=args.reg,fairreg=penalty, batch_size=args.batch_size, verbose=True)

            args_dict = {
                "X_train": X_train,
                "X_retrain": X_retrain, 
                "Y_train": Y_train,
                "Y_retrain": Y_retrain,
                "S_train": S_train,
                "S_retrain": S_retrain,
                "indices_to_remove": indices_to_remove,
                "theta_original": theta_original,
                "fair": True,
                "fairreg": penalty,
                "b": b,
                "reg": args.reg,
                "Z_full": Z_full,
                "Hat_full": Hat_full,
                "batch_size": args.batch_size,
                "precomputed_fairness_gradient": precomputed_fairness_gradient
            }

            # theta_retrain = retrain(
            #     "naive_retraining",
            #     **args_dict
            # )
            unlearned_theta = retrain(
                "fair_unlearning",
                **args_dict
            )

            metrics = generate_metrics(theta_original, dataset_retrain, dataset_test, args.protected_attribute, theta_retrain=theta_original)
            print(metrics)
            update_dictionaries(output_dict, "{:.0e}".format(penalty), i, j, metrics)

df = pd.DataFrame()
df["Requests"] = requests

df_accs = pd.DataFrame()
df_accs["Requests"] = requests

write_metrics(output_dict, df, list(map(lambda x: "{:.0e}".format(x), regs)))
write_accs(output_dict, df_accs, list(map(lambda x: "{:.0e}".format(x), regs)))

if args.unlearned_attribute is not None:
    df.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_std_{args.std:.1e}_{args.dataset}_{args.protected_attribute}_metrics_{args.unlearned_attribute}_{args.value}.csv", index=None)
    df_accs.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_std_{args.std:.1e}_{args.dataset}_{args.protected_attribute}_accs_{args.unlearned_attribute}_{args.value}.csv", index=None)

else:
    df.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_std_{args.std:.1e}_{args.dataset}_{args.protected_attribute}_metrics.csv", index=None)
    df_accs.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_std_{args.std:.1e}_{args.dataset}_{args.protected_attribute}_accs.csv", index=None)