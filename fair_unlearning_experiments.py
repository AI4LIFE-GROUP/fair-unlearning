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

print(args)

print(f"output/fair_unlearning_reg_{args.reg:.1e}_fairreg_{args.fairreg:.1e}_{args.dataset}_{args.protected_attribute}_{args.unlearned_attribute}_{args.value}")

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

# remove group attribute from private embeddings:

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

# remove_sizes = list(range(0,len(unlearning_indices_train),args.step_size))
percentages = X_train.shape[0]/100
remove_sizes = percentages*np.array([1,5,10,15,20])
remove_sizes= np.round(remove_sizes).astype(int).tolist()
temp = remove_sizes.copy()
for remove_size in temp:
    if remove_size > len(unlearning_indices_train):
        remove_sizes.remove(remove_size)

l = len(remove_sizes)

if args.fair:
    methods = ["full_dataset", "naive_retraining", "newton", "fair_unlearning"]
else:
    methods = ["full_dataset", "naive_retraining", "residual", "sharding", "newton", "fair_unlearning"]


output_dict = construct_output_dict(methods, l, repeats)

## Compute original classifier
b = np.random.normal(0, args.std, size=(X_train.shape[1]+1, 1))
theta_original = log_exact(X_train,Y_train,S_train,fair=args.fair,b=b,reg=args.reg,fairreg=args.fairreg, batch_size=args.batch_size, verbose=True)

print(theta_original.shape)

## Generate shards and train [shard] number of classifiers
shards = 5
splits = np.split(
    np.arange(0, X_train.shape[0]),
    [
        (t * X_train.shape[0] // shards)
        for t in range(1, shards)
    ],
)

shard_thetas = []
for shard_ind in splits:
    shard_thetas.append(log_exact(X_train[shard_ind], Y_train[shard_ind],S=None,reg=args.reg))


## Precompute inversion matrices for residual update
precomputed_fairness_gradient = None
if not args.fair:
    X_inflated = np.concatenate((X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)), axis=1)

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

        args_dict = {
            "X_train": X_train,
            "X_retrain": X_retrain, 
            "Y_train": Y_train,
            "Y_retrain": Y_retrain,
            "S_train": S_train,
            "S_retrain": S_retrain,
            "indices_to_remove": indices_to_remove,
            "theta_original": theta_original,
            "fair": args.fair,
            "b": b,
            "reg": args.reg,
            "fairreg": args.fairreg,
            "Z_full": Z_full,
            "Hat_full": Hat_full,
            "batch_size": args.batch_size,
            "precomputed_fairness_gradient": precomputed_fairness_gradient
        }
        
        # theta_original = log_exact(X_train,Y_train,S_train,fair=args.fair,b=b,reg=args.reg,fairreg=lam, verbose=True)
        start = time.time()
        theta_retrain = retrain("naive_retraining",
            **args_dict
        )
        runtimes["naive_retraining"].append(time.time()-start)


        for method in methods:
            if method == "sharding":
                new_shards = retrain_shard(X_train, Y_train, indices_to_remove, splits, reg=args.reg)
                if new_shards is not None:
                    metrics = generate_sharding_metrics(new_shards, dataset_retrain, dataset_test, args.protected_attribute)
            elif method == "naive_retraining":
                metrics = generate_metrics(theta_retrain, dataset_retrain, dataset_test, args.protected_attribute, theta_retrain=theta_retrain)
            else:
                start = time.time()
                unlearned_theta = retrain(
                    method,
                    **args_dict
                )
                if method == "fair_unlearning" or method == "newton":
                    runtimes[method].append(time.time()-start)
                metrics = generate_metrics(unlearned_theta, dataset_retrain, dataset_test, args.protected_attribute, theta_retrain=theta_retrain)

                if method == "fair_unlearning":
                    theta_ours = unlearned_theta
            update_dictionaries(output_dict, method, i, j, metrics)

        metrics = generate_metrics(theta_original, dataset_retrain, dataset_test, args.protected_attribute, theta_retrain=theta_retrain)
        update_dictionaries(output_dict, 'full_dataset', i, j, metrics)
    
    ## Bounds for data dependent better bound
    print("Runtimes:")
    for method in runtimes.keys():
        print(method)
        print(np.mean(runtimes[method]))
        print(np.std(runtimes[method]))

    print("\n\nBounds:")
    X_inflated = np.concatenate((X_retrain, np.ones((X_retrain.shape[0], 1), dtype=np.float64)), axis=1)
    step = theta_original-theta_ours
    eps_prime = 0.25*np.linalg.norm(X_inflated, 2)*np.linalg.norm(step, 2)*np.linalg.norm(X_inflated@step, 2)

    print(eps_prime)

    eps = np.logspace(-4, 4, num=9)

    k = args.std*eps/eps_prime

    delta = 1.5*np.exp(-np.power(k, 2)/2)

    print(eps)
    print(delta)

    delta2 = np.logspace(-4, 4, num=9)
    eps2 = (eps_prime/args.std)*(np.sqrt(-2*np.log(delta2/1.5)))

    print(eps2)
    print(delta2)
    print("\n\n")

df = pd.DataFrame()
df["Requests"] = requests

df_accs = pd.DataFrame()
df_accs["Requests"] = requests

write_metrics(output_dict, df, methods)
write_accs(output_dict, df_accs, methods)

ftrain = "fair" if args.fair else "normal"
prefix = "fair_unlearning"
if args.unlearned_attribute is not None:
    df.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_fairreg_{args.fairreg:.1e}_{args.dataset}_{ftrain}_{args.protected_attribute}_metrics_{args.unlearned_attribute}_{args.value}.csv", index=None)
    df_accs.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_fairreg_{args.fairreg:.1e}_{args.dataset}_{ftrain}_{args.protected_attribute}_accs_{args.unlearned_attribute}_{args.value}.csv", index=None)

else:
    df.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_fairreg_{args.fairreg:.1e}_{args.dataset}_{ftrain}_{args.protected_attribute}_metrics.csv", index=None)
    df_accs.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_fairreg_{args.fairreg:.1e}_{args.dataset}_{ftrain}_{args.protected_attribute}_accs.csv", index=None)

## Bounds for Theorem 1
eps_prime = (0.25/(args.reg**2)/(X_train.shape[0]-1))*(
    2+np.linalg.norm(theta_original)*np.linalg.norm(8*(X_train.shape[0]-1)/sum(S_train)**2)**2
)

print(eps_prime)

eps = np.logspace(-4, 4, num=9)

k = args.std*eps/eps_prime

delta = 1.5*np.exp(-np.power(k, 2)/2)

print(eps)
print(delta)

delta2 = np.logspace(-4, 4, num=9)
eps2 = (eps_prime/args.std)*(np.sqrt(-2*np.log(delta2/1.5)))

print(eps2)
print(delta2)
print("done!")