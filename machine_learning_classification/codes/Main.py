"""
The main program generates 5 datasets and apply both K-Means and EM algorithm to each dataset for classification
prediction. This program also reports error rate and draws graphs for data visualization and algorithm performance
measurement.
"""
import matplotlib.pyplot as plt
from data_generator_and_plotter import *
from K_means_algorithm import *
from EM_algorithm import *


def train_model(num_samples, cov, mean_1, mean_2, max_iter, i):
    data, labels = generate_data(num_samples, cov, mean_1, mean_2)
    R, costs, error_rate_km = train_km(data, labels, max_iter)
    Gamma, log_likelihoods, error_rate_em = train_em(data, labels, max_iter)
    fig, ax = plt.subplots(2, 3)
    fig.suptitle("Graphs for dataset {}".format(i))
    fig.set_size_inches((20, 8))
    # upper three data plots
    draw_data_plot(data, labels, "Simulated Data Points", ax[0, 0])
    draw_data_plot(data, R[:, 0] == 0, 'Predicted Classifications by K-Means', ax[0, 1])
    draw_data_plot(data, Gamma[:, 1] >= .5, 'Predicted Classfications by EM', ax[0, 2])

    s = "number of samples: {}\n class 1 mean:{} \n class 2 mean:{}\n common covariance matrix:\n {}\n " \
        "maximum iterations:{}".format(num_samples, mean_1, mean_2, cov, max_iter)
    draw_text_plot(ax[1, 0], s)

    draw_info_plot(max_iter, costs, "K-Means Cost Over {} Iterations".format(max_iter), ax[1, 1], error_rate_km)
    draw_info_plot(max_iter, log_likelihoods, "EM Log Likelihood Over {} Iterations".format(max_iter), ax[1, 2],
                   error_rate_em)


# generate 5 datasets and apply both algorithms to them
# using same means and covariances for all 5 datasets
num_samples = 400
cov = np.array([[1., .7], [.7, 1.]]) * 10
mean_1 = [.1, .1]
mean_2 = [6., .1]
max_iter = 25
for i in range(5):
    train_model(num_samples, cov, mean_1, mean_2, max_iter, i + 1)
