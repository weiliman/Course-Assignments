"""
Don't run this file, it won't give you any outputs.
In this file I implemented the K-means algorithm by defining the assignment step and refitting step.
"""
import numpy as np


def km_assignment_step(data, Mu):
    """ Compute K-Means assignment step

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the cluster means locations

    Returns:
        R_new: a NxK matrix of responsibilities
    """

    N, D = data.shape
    K = Mu.shape[1]
    r = np.zeros((N, K))
    for k in range(K):
        r[:, k] = np.sum((data - Mu[:, k]) ** 2, axis=1)
    arg_min = np.argmin(r, axis=1)
    R_new = np.zeros((N, K))
    R_new[np.arange(N), arg_min] = 1
    return R_new


def km_refitting_step(data, R):
    """ Compute K-Means refitting step.

    Args:
        data: a NxD matrix for the data points
        R: a NxK matrix of responsibilities
        Mu: a DxK matrix for the cluster means locations

    Returns:
        Mu_new: a DxK matrix for the new cluster means locations
    """
    Mu_new = np.matmul(data.T, R) / np.sum(R, axis=0)
    return Mu_new


def cost(data, R, Mu):
    """ Compute the cost for K-means algorithm. This shows how model performance changes over iterations.

    """
    N, D = data.shape
    K = Mu.shape[1]
    J = 0
    for k in range(K):
        J += np.dot(np.linalg.norm(data - np.array([Mu[:, k], ] * N), axis=1)**2, R[:, k])
    return J


def train_km(data, labels, max_iter):
    N, D = data.shape
    K = 2
    class_init = np.random.binomial(1., .5, size=N)
    R = np.vstack([class_init, 1 - class_init]).T

    Mu = np.zeros([D, K])
    Mu[:, 1] = 1.
    R.T.dot(data), np.sum(R, axis=0)
    costs = []

    for it in range(max_iter):
        R = km_assignment_step(data, Mu)
        Mu = km_refitting_step(data, R)
        cost_i = cost(data, R, Mu)
        costs.append(cost_i)

    class_1 = np.where(R[:, 0])
    class_2 = np.where(R[:, 1])
    error_rate = (np.sum(labels[class_1]) + np.sum(1 - labels[class_2])) / N
    return R, costs, error_rate
