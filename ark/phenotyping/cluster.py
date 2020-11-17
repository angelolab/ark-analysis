import os

import numpy as np
import pandas as pd


def winner(sample, weights):
    """Find the coordinates to the winning neuron in the SOM

    Args:
        sample (numpy.ndarray):
            A row in the pixel matrix identifying information associated with one pixel
        weights (numpy.ndarray):
            A square matrix identifying the weights between two neurons

    Returns:
        tuple:
            A coordinate array which indicates the winning neuron's position
    """

    # get euclidean distance between the sample and the weights
    activation_map = np.linalg.norm(np.subtract(sample, weights), axis=-1)

    # find the winning neuron's coordinates, needed for easy access
    winning_coords = np.unravel_index(activation_map.argmin(), activation_map.shape)

    return winning_coords


def update(sample, weights, winning_coords, sigma, learning_rate,
           x_mesh, y_mesh, t, num_iters):
    """Updates the weights, learning rate, and sigma parameters

    Args:
        sample (numpy.ndarray):
            A row in the pixel matrix identifying information associated with one pixel
        weights (numpy.ndarray):
            A square matrix identifying the weights between two neurons
        winning_coords (tuple):
            A coordinate array which indicates the winning neuron's position
        sigma (float):
            Determines the spread of the Gaussian neighborhood function
        learning_rate (float):
            Determines how sensitive the weight updates will be to new data
        x_mesh (numpy.ndarray):
            The x coordinate matrix of the weights vectors
        y_mesh (numpy.ndarray):
            The y coordinate matrix of the weights vectors
        t (int):
            The current iteration index
        num_iters (int):
            The maximum number of iterations

    Returns:
        tuple (numpy.ndarray, float, float):

        - The updated weights matrix
        - The updated learning rate
        - The updated sigma
    """

    # update learning rate with asymptotic decay
    learning_rate = learning_rate / (1 + t / (num_iters / 2))

    # update sigma with asymptotic decay
    sigma = sigma / (1 + t / (num_iters / 2))

    # return a Gaussian centered around the winning coordinates
    d = 2 * np.pi * (sigma**2)
    ax = np.exp(-np.power(x_mesh - x_mesh.T[winning_coords], 2) / d)
    ay = np.exp(-np.power(y_mesh - y_mesh.T[winning_coords], 2) / d)
    g = (ax * ay).T * learning_rate

    # update the weights based on the Gaussian neighborhood
    weights += np.einsum('ij, ijk->ijk', g, sample - weights)

    return weights, learning_rate, sigma


def train_flowsom(pixel_mat, num_iters=100, sigma=1.0, learning_rate=0.5):
    """Trains the SOM by iterating through the each data point and updating the params

    Args:
        pixel_mat (pandas.DataFrame):
            A matrix with pixel-level channel information for non-zero pixels in img_xr
        num_iters (int):
            The maximum number of iterations for training
        sigma (float):
            Determines the spread of the Gaussian neighborhood function
        learning_rate (float):
            Determines how sensitive the weight updates will be to new data

    Returns:
        tuple (numpy.ndarray, float, float):

        - The weights matrix after training
        - The learning rate after training
        - The sigma after training
    """

    # compute number of neurons to use, based on suggestion in MiniSom init docstring
    num_neurons = 5 * pixel_mat.shape[0]**(1 / 2)

    # define the random generator
    rand_gen = np.random.RandomState(random_seed)

    # initialize the weights and normalize
    weights = rand_gen.rand(num_neurons, num_neurons, pixel_mat.shape[1]) * 2 - 1
    weights /= np.linalg.norm(weights, axis=-1, keepdims=True)

    # define the activation map
    activation_map = np.zeros((num_neurons, num_neurons))

    # define meshgrid coords for the weights matrix
    x_mesh, y_mesh = np.meshgrid(np.arange(num_neurons), np.arange(num_neurons))

    # define the iterations iterable
    iterations = np.arange(num_iters) % pixel_mat.shape[0]

    for t, iteration in enumerate(iterations):
        # find the winning neuron's coordinates
        winning_coords = winner(pixel_mat.loc[iteration, :].values, weights)

        # update the weights, learning rate, and sigma
        weights, learning_rate, sigma = update(pixel_mat.loc[iteration, :].values, weights,
                                               winning_coords, sigma, learning_rate,
                                               x_mesh, y_mesh, t, num_iters)

    return weights, learning_rate, sigma


def cluster_flowsom(pixel_mat, weights, cluster_col='pixel_cluster'):
    """Assigns the winning neuron to each pixel in the matrix based on the trained weights

    Args:
        pixel_mat (pandas.DataFrame):
            A matrix with pixel-level channel information for non-zero pixels in img_xr
        weights (numpy.ndarray):
            The weights matrix after training
        cluster_col (str):
            The name of the pixel cluster column to create

    Returns:
        pandas.DataFrame:
            The pixel matrix with cluster_col indicating the cluster the pixel belongs to
    """

    # iterate through each row and assign the cluster value accordingly
    pixel_mat[cluster_col] = pixel_mat.apply(
        lambda row: winner(np.array(row.values()), weights), axis=1)

    return pixel_mat
