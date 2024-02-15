"""
Retraining methods largely adapted from Izzo et al.'s work and accompanying codebase:
Izzo et al, https://arxiv.org/abs/2002.10077
"""
import numpy as np
import torch
import torch.nn as nn
from utils import *
from model import *
import time

## TODO
def retrain(method,
            X_train=None,
            X_retrain=None, 
            Y_train=None,
            Y_retrain=None,
            S_train=None,
            S_retrain=None,
            indices_to_remove=None,
            theta_original=None,
            fair=None,
            b=None,
            reg=None,
            fairreg=None,
            Z_full=None,
            Hat_full=None,
            batch_size=None,
            precomputed_fairness_gradient=None):
    if method == "naive_retraining":
        return log_exact(X_retrain, Y_retrain, S_retrain, fair=fair, b=b, reg=reg, fairreg=fairreg, batch_size=batch_size, verbose=True)
    elif method == "newton":
        return log_newton(X_train, Y_train, theta_original, indices_to_remove, reg=reg)
    elif method == "fair_unlearning":
        return fair_unlearning(X_train, Y_train, S_train, theta_original, indices_to_remove, precomputed_fairness_gradient=precomputed_fairness_gradient, fairreg=fairreg, reg=reg)
    elif method == "residual":
        return lin_res(X_train, Z_full, theta_original, indices_to_remove, Hat_full, reg=reg)
    elif method == "full_dataset":
        return theta_original
    return None

def log_exact(X,Y,S=None,fair=False,b=None,reg=1e-4,fairreg=1.,epochs=20, batch_size=None, verbose=False):
    """log_exact Trains a logistic regression classifier

    Parameters
    ----------
    X : a matrix in (n x d)
        data
    Y : a matrix in (n x 1)
        labels
    S : a matrix in (n x 1)
        sensitive attribute labels, optional
    fair : whether to train with fairness regularization or not, optional
        if fair=True, then sensitive attributes S must be provided.
    b : objective perturbation, noise vector in (1 x d), optional
    reg : float, optional
        l2 regulariation parameter, by default 1e-4
    fairreg : float, optional
        fairness regularization parameter, by default 1.0
    epochs : int, optional
        number of steps taken to train LR classifier, by default 20
    batch_size : int, optional
        batch size if using SGD or large datasets, not necessary for datasets in this repo, by default None
    verbose : bool, optional
        print various information during run, by default False

    Returns
    -------
    theta:
        a weight vector in (1 x d)
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if b is not None:
        b = torch.tensor(b).to(torch.float32).to(device)
    if batch_size is None:
        batch_size = X.shape[0]

    ## Wrap data in pytorch loader
    train_data = []
    if fair:
        S0 = np.where(S == 0)[0]
        S1 = np.where(S == 1)[0]
        n0 = S0.shape[0]
        n1 = S1.shape[0]

        if batch_size is None:
            total_A = fair_precompute_summation(np.concatenate((X, np.ones((X.shape[0], 1), dtype=np.float32)), axis=-1), Y, S, np.arange(X.shape[0]), np.arange(X.shape[0]), verbose=verbose)
            total_diff_mat = torch.tensor(total_A).to(torch.float32).to(device)/(n0*n1)

        for i in range(len(X)):
            train_data.append([X[i], Y[i], S[i]])
     
    else:
        for i in range(len(X)):
            train_data.append([X[i], Y[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    ## Set up model and optimizer
    model = LogisticRegression(X.shape[1]+1).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), tolerance_grad=1e-10, tolerance_change=1e-20)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        total_fair_loss = 0
        total = 0
        correct = 0
        for payload in trainloader:
            if fair:
                data, label, sensitive = payload
                sensitive = sensitive.to(torch.int64).to(device)
            else:
                data, label = payload

            optimizer.zero_grad()

            data = torch.cat((data, torch.ones((data.size(0), 1))), dim=-1).to(torch.float32).to(device)

            label = label.to(torch.float32).to(device)

            output = model(data)

            if b is None:
                loss = loss_fn(output.reshape(-1), label)
            else:
                loss = loss_fn(output.reshape(-1), label) + b.T@model.theta/data.size(0)

            ## weight decay
            loss += torch.square(torch.linalg.norm(model.theta))*reg/2

            total_loss += loss.item()*data.size(0)

            if fair:
                if batch_size is not None:
                    X_batch = data.detach().cpu().numpy()
                    y_batch = label.detach().cpu().numpy()
                    s_batch = sensitive.detach().cpu().numpy()
                    n0_batch = np.where(s_batch == 0)[0].shape[0]
                    n1_batch = np.where(s_batch == 1)[0].shape[0]

                    total_A = fair_precompute_summation(X_batch, y_batch, s_batch, np.arange(X_batch.shape[0]), np.arange(X_batch.shape[0]), verbose=verbose)
                    total_diff_mat = torch.tensor(total_A).to(torch.float32).to(device)/(n0_batch*n1_batch)
                loss_fair = fairreg*fair_loss(total_diff_mat, model.theta)
                loss += loss_fair
                    
            def closure():
                loss = loss_fn(output.reshape(-1), label)
                loss += torch.square(torch.linalg.norm(model.theta))*reg/2
                if fair:
                    loss += fairreg*fair_loss(total_diff_mat, model.theta)
                if b is not None:
                    loss += (b.T@model.theta/data.size(0)).item()
                return loss
            loss.backward()
            optimizer.step(closure)

            if verbose and (epoch+1)%10 == 0:
                with torch.no_grad():
                    # Calculating the loss and accuracy for the train dataset
                    total += label.size(0)
                    correct += np.sum(torch.squeeze(output).round().detach().cpu().numpy() == label.detach().cpu().numpy())
                    if fair:
                        total_fair_loss += loss_fair.item()*data.size(0)

        if verbose and (epoch+1)%10 == 0:
            accuracy = 100 * correct/total
            print("Epoch {} \t Loss {} \t Accuracy {}".format(epoch, total_loss/X.shape[0], accuracy))
            print("Norm Theta {}".format(torch.linalg.norm(model.theta)))
            print("Grad : {}".format(torch.linalg.norm(model.theta.grad)))

            if fair:
                print("Fair Grad {}".format(torch.linalg.norm(2*fairreg*X.shape[0]*model.theta.T@torch.outer(total_diff_mat,total_diff_mat))))
            with torch.no_grad():
                inputs = torch.cat((torch.tensor(X), torch.ones((X.shape[0], 1))), dim=-1).to(torch.float32).to(device)
                preds = model(inputs)
                print("MSE Error {}".format(nn.functional.mse_loss(preds, torch.tensor(Y).unsqueeze(-1).to(device))))
                print("Grad {}".format(torch.linalg.norm(inputs.T@(torch.tensor(Y).unsqueeze(-1).to(torch.float32).to(device)-preds))), flush=True)
    return model.theta.detach().cpu().numpy().reshape(-1)

def fair_loss(diff_mat, theta):
    """fair_loss

    Parameters
    ----------
    diff_mat : a tensor in (1 x d)
        a vector output by fair_precompute_summation() that sums the differences in each feature for inputs of different subgroups with the same label.
        in other words, from the paper, \sum_{i,j} 1[y_i = y_j](x_i - x_j).
        at this point, already normalized by 1/(n_a n_b)
    theta : a tensor in (1 x d)
        a model weight vector

    Computes 1/(n_a n_b)^2 (\sum_{i, j} 1[y_i = y_j]<x_i - x_j, theta>)^2 by doing:
        theta^T 1/(n_a n_b) (\sum_{i,j} 1[y_i = y_j](x_i - x_j))^T 1/(n_a n_b) (\sum_{i,j}1[y_i = y_j](x_i - x_j)) theta.

    Returns
    -------
    The fair loss penalty as described in the paper, a scalar value to backprop over.
    """
    return (theta.T@torch.outer(diff_mat, diff_mat)@theta)[0,0]

def fair_precompute_summation(X, Y, S, ind_group_0, ind_group_1, verbose=False):
    """fair_precompute_summation
    
    Allows the output of matrices N, C_{D'} as outlined in the paper. 
    For N, use the entire set of indices (np.arange(X[indices_of_group].shape[0])), and for C_{D'} use the indices of just the remaining data.

    Parameters
    ----------
    X : a matrix in (n x d)
        data
    Y : a matrix in (n x 1)
        labels
    S : a matrix in (n x 1)
        sensitive attribute labels
    ind_group_0 : a list or array of indices
        indices from group a to consider. Usually the indices of the remaining dataset or full dataset for group a.
    ind_group_1 : a list or array of indices
        indices from group b to consider. Usually the indices of the remaining dataset or full dataset for group b.
    verbose : bool, optional
        whether to print various statistics or not, by default False

    Returns
    -------
    summation : a matrix in (1 x d)
        the summation values for each dimension in the data. Used to compute or unlearn over the fair loss.
    """
    assert len(S[ind_group_0]) == len(X[ind_group_0])
    assert len(S[ind_group_1]) == len(X[ind_group_1])

    X_set_0, X_set_1 = X[ind_group_0], X[ind_group_1]
    Y_set_0, Y_set_1 = Y[ind_group_0], Y[ind_group_1]
    S0 = np.where(S[ind_group_0] == 0)[0]
    S1 = np.where(S[ind_group_1] == 1)[0]

    y_outer = (np.outer((2*Y_set_0[S0]-1).reshape(-1),(2*Y_set_1[S1]-1).reshape(-1))>0).astype(np.float32)
    x_diff = np.zeros((X.shape[1], X_set_0[S0].shape[0], X_set_1[S1].shape[0]), dtype=np.float32).astype(np.float32)
    
    for i in range(X.shape[1]):
        x_diff[i, :, :] = X_set_0[S0, i:i+1] - X_set_1[S1, i:i+1].T

    summation = np.einsum('ijk,jk->i', x_diff, y_outer)

    return summation

def fair_unlearning(X, Y, S, theta, ind, precomputed_fairness_gradient=None, reg=1e-4, fairreg=1):
    """fair_unlearning

    Our Fair Unlearning method

    Parameters
    ----------
    X : a matrix in (n x d)
        data
    Y : a matrix in (n x 1)
        labels
    S : a matrix in (n x 1)
        sensitive attribute labels
    theta : a vector in (1 x d)
        fully-trained model parameters
    ind : a list of indices in [1, n]
        the set of indices to unlearn
    precomputed_fairness_gradient : a matrix in (1 x d), optional
        precomputed fairness gradient over the full dataset to accelerate runtime, by default None
    reg : float, optional
        l2 regulariation parameter, by default 1e-4
    fairreg : float, optional
        fairness regularization parameter, by default 1.0

    Returns
    -------
    updated : a vector in (1 x d)
        the unlearned model weights.
    """
    n = len(Y)
    k = len(ind)
    d = len(theta)
    grad = np.zeros(d)
    ind_comp = [i for i in range(n) if i not in ind]
    Sfull_0 = np.where(S == 0)[0]
    Sfull_1 = np.where(S == 1)[0]
    Sremaining_0 = np.where(S[ind_comp] == 0)[0]
    Sremaining_1 = np.where(S[ind_comp] == 1)[0]
    n0 = Sfull_0.shape[0]
    n1 = Sfull_1.shape[0]
    n0_remaining = Sremaining_0.shape[0]
    n1_remaining = Sremaining_1.shape[0]

    X_inflated = np.concatenate((X, np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
    
    A = fair_precompute_summation(X_inflated, Y.reshape(-1), S.reshape(-1),ind_comp,ind_comp)

    if precomputed_fairness_gradient is None:
        precomputed_fairness_gradient = (fair_precompute_summation(X_inflated, Y.reshape(-1), S.reshape(-1),np.arange(X_inflated.shape[0]), np.arange(X_inflated.shape[0])))

    # assert (np.equal(decomposed, total).all()), "Decomposed fairness loss between unlearned and remaining data does not equal full fairness loss \n Decomposed: {} \n Total: {} \n Diff: {}".format(decomposed, total, decomposed-total)

    fairdelta = np.zeros(d)

    grad = X_inflated[ind, :].T@(sigmoid(X_inflated[ind, :]@theta.T)-Y[ind]) + k*reg*theta

    full_normalization = n/(np.power(n0*n1,2)) ## FIXME Mean
    remainder_normalization = (n-k)/(np.power(n0_remaining*n1_remaining,2)) ## FIXME Mean

    fairdelta += fairreg*2*full_normalization*theta@np.outer(precomputed_fairness_gradient, precomputed_fairness_gradient)
    fairdelta -= fairreg*2*remainder_normalization*theta@np.outer(A,A)

    preds = sigmoid(np.dot(X_inflated[ind_comp, :], theta))
    diagvals = preds*(1-preds)
    W = np.diag(diagvals)

    fairness_hess = 2*fairreg*remainder_normalization*np.outer(A,A)
    remainder_hess = np.matmul(np.matmul(X_inflated[ind_comp, :].T, W), X_inflated[ind_comp, :]) + n*np.eye(X_inflated.shape[1])*reg

    step = np.linalg.solve(remainder_hess+fairness_hess, grad+fairdelta)
    updated = theta + step
    return updated

def log_newton(X, Y, theta, ind, invhess=None, reg=1e-4):
    """log_newton

    Guo et al.'s Unlearning method (https://arxiv.org/abs/1911.03030)
    Adapted from code by Izzo et al. (https://proceedings.mlr.press/v130/izzo21a)

    Parameters
    ----------
    X : a matrix in (n x d)
        data
    Y : a matrix in (n x 1)
        labels
    theta : a vector in (1 x d)
        fully-trained model parameters
    ind : a list of indices in [1, n]
        the set of indices to unlearn
    invhess : a matrix in (d x d), optional
        a precomputed inverse hessian to accelerate runtime, by default None
    reg : float, optional
        l2 regulariation parameter, by default 1e-4

    Returns
    -------
    updated : a vector in (1 x d)
        the unlearned model weights.
    """
    n = len(Y)
    k = len(ind)
    d = len(theta)
    grad = np.zeros(d)

    X_inflated = np.concatenate((X, np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)

    grad = X_inflated[ind, :].T@(sigmoid(X_inflated[ind, :]@theta.T)- Y[ind]) + k*reg*theta

    if invhess is not None:
        invLKOhess = SMW(invhess, -X[ind, :], X[ind, :])
        updated = theta + np.matmul(invLKOhess, grad)
    else:
        ind_comp = [i for i in range(n) if i not in ind]

        preds = np.array(sigmoid(np.dot(X_inflated[ind_comp, :], theta[0:X_inflated[ind_comp, :].shape[1]]) + theta[-1]))
        diagvals = preds*(1-preds)#+1e-3

        W = np.diag(diagvals)
        H = np.matmul(np.matmul(X_inflated[ind_comp, :].T, W), X_inflated[ind_comp, :])#+1e-3*np.eye(X_inflated.shape[1])
        
        LKOhess = H + n*np.eye(X_inflated[ind_comp, :].shape[1])*reg

        step = np.linalg.solve(LKOhess, grad)
        updated = theta + step
    return updated

def retrain_shard(X,Y,ind,splits,method=log_exact,reg=1e-3):
    """retrain_shard

    Parameters
    ----------
    X : a matrix in (n x d)
        data
    Y : a matrix in (n x 1)
        labels
    ind : a list of indices in [1, n]
        the set of indices to unlearn
    splits : a list of lists of indices in [1,n]
        a list of the allocation of samples per split. The union of all splits should be the set of integers [1,n].
    method : function, optional
        training function used on each split, log_exact is the only one supported for the experiments
    reg : float, optional
        l2 regulariation parameter, by default 1e-4

    Returns
    -------
    thetas
        a list of retrained thetas, one for each shard.
    """
    thetas = []
    for split in splits:
        ind_to_remove = np.intersect1d(ind, split)
        ind_to_keep = [i for i in split if i not in ind_to_remove]
        if len(np.unique(Y[ind_to_keep])) < 2:
            return None
        if method == log_exact:
            thetas.append(method(X[ind_to_keep], Y[ind_to_keep], reg=reg))
        else:
            print("method specified incorrectly for sharding retraining.")
    return thetas

'''
The below methods come directly from Izzo et al. (https://proceedings.mlr.press/v130/izzo21a)

'''

def gram_schmidt(X):
    """
    Uses numpy's qr factorization method to perform Gram-Schmidt.

    Args:
        X: (k x d matrix) X[i] = i-th vector

    Returns:
        U: (k x d matrix) U[i] = i-th orthonormal vector
        C: (k x k matrix) Coefficient matrix, C[i] = coeffs for X[i], X = CU
    """
    (k, d) = X.shape

    if k <= d:
        q, r = np.linalg.qr(np.transpose(X)) # dxk, kxk, out: kxd, kxk
    else:
        q, r = np.linalg.qr(np.transpose(X), mode='complete') # dxd, dxk out: dxd, kxd
    U = np.transpose(q)
    C = np.transpose(r)
    return U, C


def LKO_pred(X, Y, ind, H=None, reg=1e-4):
    """
    Computes the LKO model's prediction values on the left-out points.

    Args:
        X: (n x d matrix) Covariate matrix
        Y: (n x 1 vector) Response vector
        ind: (k x 1 list) List of indices to be removed
        H: (n x n matrix, optional) Hat matrix X (X^T X)^{-1} X^T

    Returns:
        LKO: (k x 1 vector) Retrained model's predictions on X[i], i in ind
    """
    n = len(Y)
    k = len(ind)
    d = len(X[0, :])
    if H is None:
        H = np.matmul(X, np.linalg.solve(np.matmul(X.T, X) + reg * np.eye(d), X.T))

    LOO = np.zeros(k)
    for i in range(k):
        idx = ind[i]
        # This is the LOO residual y_i - \hat{y}^{LOO}_i
        LOO[i] = (Y[idx] - np.matmul(H[idx, :], Y)) / (1 - H[idx, idx])

    # S = I - T from the paper
    S = np.eye(k)
    for i in range(k):
        for j in range(k):
            if j != i:
                idx_i = ind[i]
                idx_j = ind[j]
                S[i, j] = -H[idx_i, idx_j] / (1 - H[idx_i, idx_i])

    LKO = np.linalg.solve(S, LOO)

    return Y[ind] - LKO

def lin_res(X, Y, theta, ind, H=None, reg=1e-4):
    """
    Approximate retraining via the projective residual update. 

    Note that for logistic regression, we conduct Algorithm 3 from (https://proceedings.mlr.press/v130/izzo21a) by passing in
    Z = X@theta + S^{-1}(Y-h_theta) for Y. 

    Args:
        X: (n x d matrix) Covariate matrix
        Y: (n x 1 vector) Response vector
        theta: (d x 1 vector) Current value of parameters to be updated
        ind: (k x 1 list) List of indices to be removed
        H: (n x n matrix, optional) Hat matrix X (X^T X)^{-1} X^T

    Returns:
        updated: (d x 1 vector) Updated parameters
    """
    d = len(theta)
    k = len(ind)

    X_inflated = np.concatenate((X, np.ones((X.shape[0], 1), dtype=np.float32)), axis=1)
    # Step 1: Compute LKO predictions
    LKO = LKO_pred(X_inflated, Y, ind, H, reg)

    # Step 2: Eigendecompose B
    # 2.I
    U, C = gram_schmidt(X_inflated[ind, :])

    # 2.II
    Cmatrix = np.matmul(C.T, C) # dxd matrix
    eigenval, a = np.linalg.eigh(Cmatrix)
    V = np.matmul(a.T, U)

    # Step 3: Perform the update
    # 3.I
    grad = np.zeros(d)
    for i in range(k):
        grad += (np.dot(X_inflated[i, :], theta) - LKO[i]) * X_inflated[ind[i], :]

    # 3.II (pseudoinverse)
    mat = np.zeros((d,d))
    for data in X_inflated[ind, :]:
        mat += np.outer(data, data)
    # S_i = np.linalg.pinv(mat)
    # 3.II
    step = np.zeros(d) #inv matrix @ gradient
    for i in range(min(d,k)):
        factor = 1 / eigenval[i] if eigenval[i] > 1e-10 else 0
        step += factor * np.dot(V[i, :], grad) * V[i, :]


    # step, _, _, _ = np.linalg.lstsq(mat, grad, rcond=None)#np.matmul(S_i, grad)
    # 3.III
    update = theta - step
    return update


def SMW(Ainv, U, V):
    """
    Computes (A + U^T V)^{-1} given A^{-1}, U, and V.
    Uses the Sherman-Morrison-Woodbury (SMW) formula.
    (A + U^T V)^{-1} = A^{-1} - A^{-1} U^T (I + V A^{-1} U^T)^{-1} V A^{-1}

    Args:
        Ainv: (d x d matrix)
        U: (k x d matrix)
        V: (k x d matrix)

    Returns:
        inv: (d x d matrix) (A + U^T V)^{-1}
    """
    k = len(U)
    # Compute (I + V A^{-1} U^T)^{-1} V A^{-1}
    S = np.linalg.solve(np.eye(k) + np.matmul(V, np.matmul(Ainv, U.T)), np.matmul(V, Ainv))
    # Compute A^{-1} - A^{-1} U^T S
    return Ainv - np.matmul(Ainv, np.matmul(U.T, S))

if __name__ == "__main__":
    '''
    Just some unit tests for the fair_precompute_summation algorithm
    '''
    X = np.array([
        [0, 1, 0],
        [.5, .5, 0],
        [0, 0, 0],
        [0, 0, 1],
        [-0.5, 0, -0.5]
    ])

    S = np.array([
        0,0,1,1,0
    ])

    Y = np.array([
        0,1,0,1,0
    ])

    # Expected A value:
    A = fair_precompute_summation(X, Y, S, np.arange(X.shape[0]), np.arange(X.shape[0]), verbose=True)
    print(A)
    print((A == np.array([0,1.5,-1.5])/6).all())

    n0 = 3
    n1 = 2
    print(2*(n0**1)*(n1**1))
    print("A Norm: {}".format(np.linalg.norm(A)))