from fair_retraining import *

import pandas as pd
import argparse
from tqdm import tqdm
from aif360.datasets import StandardDataset

from utils import *
from datasets import *
from datasets.Dataloader import load_data

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

args = parser.parse_args()

print("Hyperparameter tuning, dataset {}".format(args.dataset))

## Load Data
traindf, testdf, labelname = load_data(args.dataset, returndf=True)

traindf = traindf.astype(np.float32)
testdf = testdf.astype(np.float32)

dataset_train = StandardDataset(traindf, label_name=labelname, favorable_classes=[1],
                                protected_attribute_names=[args.protected_attribute], privileged_classes=[[traindf[args.protected_attribute].unique()[np.where(traindf[args.protected_attribute].unique()>0)][0]]])
dataset_train.labels = dataset_train.labels.ravel()
dataset_test = StandardDataset(testdf, label_name=labelname, favorable_classes=[1],
                                protected_attribute_names=[args.protected_attribute], privileged_classes=[[testdf[args.protected_attribute].unique()[np.where(testdf[args.protected_attribute].unique()>0)][0]]])
dataset_test.labels = dataset_test.labels.ravel()
X_train, Y_train, X_test, Y_test = dataset_train.features.astype(np.float32), dataset_train.labels.astype(np.float32), dataset_test.features.astype(np.float32), dataset_test.labels.astype(np.float32)
S_train, S_test = (dataset_train.protected_attributes>0).astype(np.float32), (dataset_test.protected_attributes>0).astype(np.float32)

dataset_test.protected_attributes = (dataset_test.protected_attributes > 0).astype(np.float32)
dataset_train.protected_attributes = (dataset_train.protected_attributes > 0).astype(np.float32)

uniq, counts = np.unique(S_train, return_counts=True)
print(uniq, counts/S_train.shape[0])
uniq, counts = np.unique(S_test, return_counts=True)
print(uniq, counts/S_test.shape[0])


if args.unlearned_attribute is not None:
    print("Unlearned attribute specified. Current implementation will only unlearn protected attribute based on the value argument.")

    unlearning_indices_train = np.where(dataset_train.protected_attributes == args.value)[0]
    complement_indices_train = np.where(dataset_train.protected_attributes != args.value)[0]

else:
    print("Unlearned attribute not specified. Unlearning uniformly at random.")
    unlearning_indices_train = list(range(1, len(X_train)))

percentages = X_train.shape[0]/100
remove_sizes = percentages*np.array([1,5,10,15,20])
remove_sizes= np.round(remove_sizes).astype(int).tolist()

methods=["fair_unlearning"]

stds = np.logspace(-4, 4, num=9)
delts = np.logspace(-4, -1, num=4)
repeats = 5

output_dict = construct_output_dict([f'$\delta = {delt}$' for delt in delts], len(stds), repeats)

k=100
indices_to_remove = np.random.choice(unlearning_indices_train, k, replace=False)
# indices_to_remove = np.random.choice(outliers, k, replace=False)
indices_to_keep = [i for i in range(len(X_train)) if i not in indices_to_remove]
X_retrain = X_train[indices_to_keep]
Y_retrain = Y_train[indices_to_keep]
S_retrain = S_train[indices_to_keep]
dataset_retrain = StandardDataset(traindf.iloc[indices_to_keep], label_name=labelname, favorable_classes=[1],
                        protected_attribute_names=[args.protected_attribute], privileged_classes=[[1]])



for delt in tqdm(delts,position=0,leave=False, desc="Deltas"):
    requests = []
    epsilons = []
    epsilons2 = []

    for i, std in enumerate(stds):
        epslist = []
        epslist2 = []
        for j in tqdm(range(repeats),position=1,leave=False, desc="Repeat"):
            b = np.random.normal(0, std, size=(X_train.shape[1]+1, 1))

            theta_original = log_exact(X_train,Y_train,S_train,fair=args.fair,b=b,epochs=50, reg=args.reg,fairreg=args.fairreg, verbose=True)
            print("Removal Size: {} Repeat: {}".format(k, j))
            requests.append(std)

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
                "fairreg": args.fairreg
            }

            theta_retrain = retrain("naive_retraining",
                **args_dict
            )
            fair_unlearning_theta = retrain("fair_unlearning", **args_dict)
            metrics = generate_metrics(fair_unlearning_theta, dataset_retrain, dataset_test, args.protected_attribute, theta_retrain=theta_retrain)
            update_dictionaries(output_dict, f'$\delta = {delt}$', i, j, metrics)

            X_inflated = np.concatenate((X_retrain, np.ones((X_retrain.shape[0], 1), dtype=np.float64)), axis=1)
            step = theta_original-fair_unlearning_theta
            eps_prime = 0.25*np.linalg.norm(X_inflated, 2)*np.linalg.norm(step, 2)*np.linalg.norm(X_inflated@step, 2)

            eps = np.sqrt(-2*np.log(delt/1.5))*eps_prime/np.power(std, 2)
            epslist.append(eps)

            eps_prime2 = (0.25/(args.reg**2)/(X_train.shape[0]-1))*(2+np.linalg.norm(theta_original)*np.linalg.norm(8*(X_train.shape[0]-1)/sum(S_train)**2)**2)

            eps2 = np.sqrt(-2*np.log(delt/1.5))*eps_prime2/np.power(std, 2)
            epslist2.append(eps2)


        epsilons += [np.mean(epslist) for _ in range(5)]
        epsilons2 += [np.mean(epslist2) for _ in range(5)]

        print(f"Epsilon for Delta = {delt}, Sigma = {std}: {np.mean(epslist)}+/-{np.std(epslist)}")


df = pd.DataFrame()
df["Requests"] = requests
df["Epsilons"] = epsilons
df["Epsilons2"] = epsilons2

df_accs = pd.DataFrame()
df_accs["Requests"] = requests
df_accs["Epsilons"] = epsilons
df_accs["Epsilons2"] = epsilons2


write_metrics(output_dict, df, [f'$\delta = {delt}$' for delt in delts])
write_accs(output_dict, df_accs, [f'$\delta = {delt}$' for delt in delts])

ftrain = "fair" if args.fair else "normal"

prefix = "privacy_tradeoffs"
if args.unlearned_attribute is not None:
    df.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_{args.dataset}_{args.protected_attribute}_metrics_{ftrain}_{args.unlearned_attribute}_{args.value}.csv", index=None)
    df_accs.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_{args.dataset}_{args.protected_attribute}_accs_{ftrain}_{args.unlearned_attribute}_{args.value}.csv", index=None)

else:
    df.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_{args.dataset}_{ftrain}_{args.protected_attribute}_metrics.csv", index=None)
    df_accs.to_csv(f"output/{prefix}_reg_{args.reg:.1e}_{args.dataset}_{ftrain}_{args.protected_attribute}_accs.csv", index=None)