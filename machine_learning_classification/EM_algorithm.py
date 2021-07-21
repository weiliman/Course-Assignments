"""
Don't run this file, it won't give you any outputs.
In this file I implemented the EM algorithm by defining Gaussian Mixture Expectation Step and
Gaussian Mixture Expectation Step.
"""

import numpy as np


def normal_density(x, mu, Sigma):
    return np.exp(-.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))) \
        / np.sqrt(np.linalg.det(2 * np.pi * Sigma))


def log_likelihood(data, Mu, Sigma, Pi):
    """ Compute log likelihood on the data given the Gaussian Mixture Parameters. This shows how model performance
    changes over iterations.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients

    Returns:
        L: a scalar denoting the log likelihood of the data given the Gaussian Mixture
    """
    N, D = data.shape
    K = Mu.shape[1]
    L, T = 0., 0.
    for n in range(N):
        T = 0
        for k in range(K):
            T += Pi[k] * normal_density(data[n], Mu[:, k], Sigma[k])
        L += np.log(T)
    return L


def gm_e_step(data, Mu, Sigma, Pi):
    """ Gaussian Mixture Expectation Step.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients

    Returns:
        Gamma: a NxK matrix of responsibilities
    """
    N, D = data.shape
    K = Mu.shape[1]
    Gamma = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            Gamma[n, k] = Pi[k] * normal_density(data[n], Mu[:, k], Sigma[k])
        Gamma[n, :] /= np.sum(Gamma[n, :])
    return Gamma


def gm_m_step(data, Gamma):
    """ Gaussian Mixture Maximization Step.

    Args:
        data: a NxD matrix for the data points
        Gamma: a NxK matrix of responsibilities

    Returns:
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients
    """
    N, D = data.shape
    K = Gamma.shape[1]
    Nk = np.sum(Gamma, axis=0)
    Mu = np.zeros((D, K))
    Sigma = [np.zeros((D, D)) for i in range(K)]

    gamma_data = np.matmul(data.T, Gamma)
    for k in range(K):
        Mu[k] = gamma_data[k] / Nk[k]
        cov = np.zeros((D, D))
        for n in range(N):
            cov += Gamma[n, k] * np.outer(data[n] - Mu[:, k], data[n] - Mu[:, k])
        Sigma[k] = cov / Nk[k]
    Pi = Nk / N
    return Mu, Sigma, Pi


def train_em(data, labels, max_iter):
    N, D = data.shape
    K = 2
    Mu = np.zeros([D, K])
    Mu[:, 1] = 1.
    Sigma = [np.eye(2), np.eye(2)]
    Pi = np.ones(K) / K
    Gamma = np.zeros([N, K])  # Gamma is the matrix of responsibilities

    log_likelihoods = []
    for it in range(max_iter):
        Gamma = gm_e_step(data, Mu, Sigma, Pi)
        Mu, Sigma, Pi = gm_m_step(data, Gamma)
        log_likelihoods.append(log_likelihood(data, Mu, Sigma, Pi))

    class_1 = np.where(Gamma[:, 0] >= .5)
    class_2 = np.where(Gamma[:, 1] >= .5)
    error_rate = (np.sum(labels[class_1]) + np.sum(1 - labels[class_2])) / N
    return Gamma, log_likelihoods, error_rate
