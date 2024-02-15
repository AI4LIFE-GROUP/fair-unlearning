import numpy as np
from aif360.metrics import ClassificationMetric

def sigmoid(x):
    return 1/(1+np.exp(-x))

def accuracy(theta, X, Y):
    n = len(Y)
    Xfull = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    preds = np.round(sigmoid(Xfull@theta))
    correct = sum(preds==Y)
    return correct / n

def predict(theta, X):
    Xfull = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    preds = np.round(sigmoid(Xfull@theta))
    return preds

def shard_predict(thetas, X):
    # majority vote
    Xfull = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
    preds = np.zeros((Xfull.shape[0], len(thetas)))
    for i, shard_theta in enumerate(thetas):
        preds[:, i] = np.round(sigmoid(Xfull@shard_theta))
    return (np.sum(preds, axis=1) >= 3).astype(int)

def shard_accuracy(thetas,X,Y):
    preds = shard_predict(X, thetas)
    correct = sum(np.array(preds==Y))
    return correct / X.shape[0]

def construct_output_dict(output_types, l, repeats):
    output_dict = {
        "protected_train": dict(),
        "unprotected_train": dict(),
        "full_train": dict(),
        "protected_test": dict(),
        "unprotected_test": dict(),
        "full_test": dict(),
        "balanced_acc": dict(),
        "equality_of_odds": dict(),
        "demographic_parity": dict(),
        "equality_of_opportunity": dict(),
        "disparate_impact": dict(),
        "l2_norm": dict()
    }

    for key in output_dict.keys():
        for output_type in output_types:
            output_dict[key][output_type] = np.zeros((l, repeats))
    return output_dict

def update_dictionaries(output_dict, method, i, j, metrics, theta_retrain=None):
    for key in output_dict.keys():
        output_dict[key][method][i,j] = metrics[key]

def write_metrics(output_dict, df, modeltypes):
    for key in modeltypes:
        avg_balanced_acc = output_dict["balanced_acc"][key].reshape(-1,1)
        avg_demographic_parity = output_dict["demographic_parity"][key].reshape(-1,1)
        avg_equality_of_odds = output_dict["equality_of_odds"][key].reshape(-1,1)
        avg_equality_of_opportunity = output_dict["equality_of_opportunity"][key].reshape(-1,1)
        avg_disparity = output_dict["disparate_impact"][key].reshape(-1,1)
        avg_l2norm = output_dict["l2_norm"][key].reshape(-1,1)

        df[key+"_balanced_acc"] = avg_balanced_acc
        df[key+"_demographic_parity"] = avg_demographic_parity
        df[key+"_equality_of_odds"] = avg_equality_of_odds
        df[key+"_equality_of_opportunity"] = avg_equality_of_opportunity
        df[key+"_disparity"] = avg_disparity
        df[key+"_l2_norm"] = avg_l2norm


def write_accs(output_dict, df_accs, modeltypes):
    for key in modeltypes:
        avg_protected_test = output_dict["protected_test"][key].reshape(-1,1)
        avg_unprotected_test = output_dict["unprotected_test"][key].reshape(-1,1)
        avg_full_test = output_dict["full_test"][key].reshape(-1,1)
        avg_protected_train = output_dict["protected_train"][key].reshape(-1,1)
        avg_unprotected_train = output_dict["unprotected_train"][key].reshape(-1,1)
        avg_full_train = output_dict["full_train"][key].reshape(-1,1)
        df_accs[key+"_protected_test"] = avg_protected_test
        df_accs[key+"_unprotected_test"] = avg_unprotected_test
        df_accs[key+"_full_test"] = avg_full_test
        df_accs[key+"_unprotected_train"] = avg_unprotected_train
        df_accs[key+"_protected_train"] = avg_protected_train
        df_accs[key+"_full_train"] = avg_full_train

def compute_metrics_from_preds(train_preds, dataset_train, test_preds, dataset_test, protected_attribute):
    dataset_pred = dataset_test.copy()

    dataset_pred.labels = test_preds

    protected_attribute_value = np.unique(dataset_test.protected_attributes)[np.where(np.unique(dataset_test.protected_attributes)==1)][0]
    unprotected_attribute_value = np.unique(dataset_test.protected_attributes)[np.where(np.unique(dataset_test.protected_attributes)==0)][0]

    classification_metric = ClassificationMetric(dataset_test, dataset_pred, 
                                                unprivileged_groups=[{protected_attribute: unprotected_attribute_value}],
                                                privileged_groups=[{protected_attribute: protected_attribute_value}])

    TPR = classification_metric.true_positive_rate()
    TNR = classification_metric.true_negative_rate()
    bal_acc = 0.5*(TPR+TNR)


    dataset_pred = dataset_train.copy()

    dataset_pred.labels=train_preds

    classification_metric_train = ClassificationMetric(
        dataset_train, 
        dataset_pred, 
        unprivileged_groups=[{protected_attribute: unprotected_attribute_value}],
        privileged_groups=[{protected_attribute: protected_attribute_value}]
    )

    return {
        "protected_test": classification_metric.accuracy(privileged=True),
        "unprotected_test": classification_metric.accuracy(privileged=False),
        "full_test": classification_metric.accuracy(privileged=None),
        "protected_train": classification_metric_train.accuracy(privileged=True),
        "unprotected_train": classification_metric_train.accuracy(privileged=False),
        "full_train": classification_metric_train.accuracy(privileged=None),
        "balanced_acc": bal_acc,
        "disparate_impact": classification_metric.disparate_impact(),
        "equality_of_odds": classification_metric.average_abs_odds_difference(),
        "equality_of_opportunity": classification_metric.equal_opportunity_difference(),
        "demographic_parity": classification_metric.statistical_parity_difference(),
    }

def generate_metrics(theta, dataset_train, dataset_test, protected_attribute, theta_retrain):
    X_retrain, X_test = dataset_train.features,dataset_test.features
    preds = predict(theta, X_test)
    train_preds = predict(theta, X_retrain)
    metrics = compute_metrics_from_preds(train_preds, dataset_train, preds, dataset_test,protected_attribute)

    metrics["l2_norm"] = np.linalg.norm(theta-theta_retrain)

    return metrics

def generate_sharding_metrics(thetas, dataset_train, dataset_test, protected_attribute):
    X_retrain, X_test = dataset_train.features, dataset_test.features
    preds = shard_predict(thetas, X_test)
    train_preds = shard_predict(thetas, X_retrain)

    metrics = compute_metrics_from_preds(train_preds, dataset_train, preds, dataset_test,protected_attribute)
    metrics["l2_norm"] = 0

    return metrics