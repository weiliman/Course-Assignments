"""
Changes:
1. display_plot will save plots instead of just showing.
2. added two functions plot_images and save_images
   to create a plot of the 5 examples.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name):
    """ Loads the data.
    """
    npzfile = np.load(file_name)

    inputs_train = npzfile["inputs_train"].T / 255.0
    inputs_valid = npzfile["inputs_valid"].T / 255.0
    inputs_test = npzfile["inputs_test"].T / 255.0
    target_train = npzfile["target_train"].tolist()
    target_valid = npzfile["target_valid"].tolist()
    target_test = npzfile["target_test"].tolist()

    num_class = max(target_train + target_valid + target_test) + 1
    target_train_1hot = np.zeros([num_class, len(target_train)])
    target_valid_1hot = np.zeros([num_class, len(target_valid)])
    target_test_1hot = np.zeros([num_class, len(target_test)])

    for ii, xx in enumerate(target_train):
        target_train_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_valid):
        target_valid_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_test):
        target_test_1hot[xx, ii] = 1.0

    inputs_train = inputs_train.T
    inputs_valid = inputs_valid.T
    inputs_test = inputs_test.T
    target_train_1hot = target_train_1hot.T
    target_valid_1hot = target_valid_1hot.T
    target_test_1hot = target_test_1hot.T
    return inputs_train, inputs_valid, inputs_test, target_train_1hot, target_valid_1hot, target_test_1hot


def save(file_name, data):
    """ Saves the model to a numpy file.
    """
    print("Writing to " + file_name)
    np.savez_compressed(file_name, data)


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name,allow_pickle=True))['arr_0'].item()


def display_plot(train, valid, y_label, number=0):
    """ Displays training curve.
    :param train: Training statistics
    :param valid: Validation statistics
    :param y_label: Y-axis label of the plot
    :param number: The number of the plot
    :return: None
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], "b", label="Train")
    plt.plot(valid[:, 0], valid[:, 1], "g", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.draw()
    # save picture
    plt.savefig("{}.png".format(y_label))


def plot_5_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(48, 48),
                cmap="gray", vmin=None, vmax=None):
    """
    arrange the 5 images in a row and add some white lines between the 5 images.
    """
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = 1  # padding white lines
    concat_images = np.ones(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding))
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def plot_10_images(images, ax, ims_per_row=5, padding=10, digit_dimensions=(48, 48),
                cmap="gray", vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    concat_images = np.ones(((digit_dimensions[0] + padding) * N_rows + 30,
                             (digit_dimensions[1] + padding) * ims_per_row + padding))
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (20 + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, number, **kwargs):
    """
    save the 5 images into a png file.
    """
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    if number == 5:
        plot_5_images(images, ax, **kwargs)
    else:
        plot_10_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)
