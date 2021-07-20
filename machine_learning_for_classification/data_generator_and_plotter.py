"""
Don't run this file, it won't give you any outputs.
This file contains a data generator function and several plotting functions for data visualization.
"""
import numpy as np


def generate_data(num_samples, cov, mean_1, mean_2):
    """ Generate two groups of multi-normal samples with different means and shared covariance.

    Args:
        num_samples: an int representing number of samples to be generated
        cov: DxD covariance matrix
        mean_1: vector of length D, the mean for group 1
        mean_2: vector of length D, the mean for group 2

    Returns:
        data: num_samples x D matrix
        labels: vector of length num_samples, each element is 0 if corresponding data belongs to group 1 and
         1 if data belongs to group 2.
    """

    x_class1 = np.random.multivariate_normal(mean_1, cov, num_samples // 2)
    x_class2 = np.random.multivariate_normal(mean_2, cov, num_samples // 2)
    xy_class1 = np.column_stack((x_class1, np.zeros(num_samples // 2)))
    xy_class2 = np.column_stack((x_class2, np.ones(num_samples // 2)))
    data_full = np.row_stack([xy_class1, xy_class2])
    np.random.shuffle(data_full)
    data = data_full[:, :2]
    labels = data_full[:, 2]
    return data, labels


def draw_data_plot(data, labels, title, plt):
    xy_c1 = data[labels == 0, :]
    xy_c2 = data[labels == 1, :]
    plt.plot(xy_c1[:, 0], xy_c1[:, 1], 'x', color='red')
    plt.plot(xy_c2[:, 0], xy_c2[:, 1], 'o', color='blue')
    plt.set_title(title)
    plt.legend(['k = 1', 'k = 2'], loc='upper left')


def draw_info_plot(max_iter, info, title, plt, error):
    plt.plot(np.arange(max_iter), info[:max_iter])
    plt.set_title(title)
    plt.set(xlabel="number of iterations")
    plt.text(max_iter/3, (max(info) - min(info))/2 + min(info), "error rate: {}%".format(round(error * 100, 2)),
             fontsize="x-large")


def draw_text_plot(plt, s):
    plt.text(0.1, 0.4, s, fontsize="large")