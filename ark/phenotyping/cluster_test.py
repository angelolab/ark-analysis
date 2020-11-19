import numpy as np
import pandas as pd

import ark.phenotyping.cluster as cluster


def test_winner():
    test_sample = np.array([0.01, 0.02])
    test_weights = np.array([[[0.001, 0.005],
                              [0.002, 0.004]],
                             [[0.003, 0.009],
                              [0.008, 0.027]]])

    winning_coords = cluster.winner(test_sample, test_weights)

    assert winning_coords == (1, 1)


def test_update():
    test_sample = np.array([0.01, 0.02])
    test_weights = np.array([[[0.001, 0.005],
                              [0.002, 0.004]],
                             [[0.003, 0.009],
                              [0.008, 0.027]]])
    test_winning_coords = (1, 1)
    test_sigma = 1.0
    test_learning_rate = 0.5
    test_x_mesh, test_y_mesh = np.meshgrid(np.arange(2), np.arange(2))

    test_t = 1
    test_num_iters = 1

    # it's impractical to check exact values, so we'll round to make life easier
    weights = cluster.update(test_sample, test_weights, test_winning_coords,
                             test_sigma, test_learning_rate, test_x_mesh, test_y_mesh,
                             test_t, test_num_iters)
    weights = np.round(weights, decimals=8)

    result = np.array([[[0.00108549, 0.00514249],
                        [0.00231832, 0.00463663]],
                       [[0.00327853, 0.00943768],
                        [0.00833333, 0.02583333]]])

    assert np.all(weights == result)


def test_train_som():
    test_pixel_mat = pd.DataFrame(np.random.rand(10, 4))
    test_x = 2
    test_y = 2
    test_num_passes = 10

    # only to see if it runs to completion with default sigma, learning_rate, and randomization
    cluster.train_som(test_pixel_mat, test_x, test_y, test_num_passes)


def test_cluster_som():
    test_pixel_mat = pd.DataFrame(np.reshape(np.arange(0.01, 0.28, 0.01), (-1, 3)))
    test_weights = np.array([[[0.001, 0.005, 0.010],
                              [0.002, 0.004, 0.007],
                              [0.003, 0.006, 0.009]],
                             [[0.003, 0.009, 0.027],
                              [0.008, 0.027, 0.064],
                              [0.011, 0.022, 0.033]],
                             [[0.025, 0.050, 0.075],
                              [0.033, 0.066, 0.099],
                              [0.016, 0.032, 0.064]]])

    cluster_labels = cluster.cluster_som(test_pixel_mat, test_weights)

    assert np.all(cluster_labels.values == np.array([0, 1, 2, 2, 2, 2, 2, 2, 2]))
