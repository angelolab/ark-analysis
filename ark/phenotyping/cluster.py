import os

import numpy as np
import pandas as pd


def decay_function(param, t, num_iters):
    """Decays the parameter using an asymptotic decay function

    Args:
        param (float):
            The value to decay (either sigma or learning_rate)
        t (int):
            The current iteration index
        num_iters (int):
            The maximum number of iterations

    Returns:
        float:
            The decayed parameter
    """

    return param / (1 + t / (num_iters / 2))


def winner(sample, weights):
    """Find the coordinates to the winning neuron in the SOM for one sample

    Args:
        sample (numpy.ndarray):
            A row in the pixel matrix identifying information associated with one pixel
        weights (numpy.ndarray):
            A weight matrix of dimensions [num_x, num_y, num_chans]

    Returns:
        tuple:
            The coordinates of the winning neuron
    """

    # get euclidean distance between the sample and the weights
    activation_map = np.linalg.norm(np.subtract(sample, weights), axis=-1)

    # find the winning neuron's coordinates, needed for easy access
    winning_coords = np.unravel_index(activation_map.argmin(), activation_map.shape)

    return winning_coords


def batch_winner(samples, weights):
    """Predict multiple samples at once

    Args:
        samples (numpy.ndarray):
            Contains the rows of the pixel matrix to predict
        weights (numpy.ndarray):
            A weight matrix of dimensions [num_x, num_y, num_chans]

    Returns:
        list:
            The list with indices corresponding to the prediction of the respective sample
    """

    # only 1 sample provided: expand to 2-D
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=0)

    # collapse the weights for proper subtraction
    # dimensions will be (m x n) x c
    # (m x n): number of m and n SOM nodes
    # c: number of channels
    weights_collapse = np.reshape(weights, (weights.shape[0] * weights.shape[1], weights.shape[2]))

    # subtract weights from each sample individually, need to reshape back to original
    # for proper Euclidean distance calculation
    samples_subtract = samples - weights_collapse[:, None]
    samples_subtract = np.reshape(samples_subtract, (weights.shape[0], weights.shape[1],
                                                     samples_subtract.shape[1],
                                                     samples_subtract.shape[2]))

    # take the Euclidean distance along the last axis, then reshuffle
    # to a final matrix of r x (m x n)
    # r: number of samples
    # (m x n): number of m and n SOM nodes
    activation_map = np.linalg.norm(samples_subtract, axis=-1)
    activation_map = np.swapaxes(np.swapaxes(activation_map, 1, 2), 0, 1)

    # collapse the last two dimensions of activation map so we can unravel the indices properly
    activation_map_collapse = np.reshape(activation_map,
                                         (samples.shape[0], weights.shape[0] * weights.shape[1]))

    # get the winning coordinates for each position
    x_min_coords, y_min_coords = np.unravel_index(np.argmin(activation_map_collapse, axis=1),
                                                  (weights.shape[0], weights.shape[1]))

    # zip the coordinates together to get the final result
    winning_coords_list = list(zip(x_min_coords, y_min_coords))

    return winning_coords_list


def update(sample, weights, winning_coords, sigma, learning_rate, x_mesh, y_mesh):
    """Updates the weights, learning rate, and sigma parameters

    Args:
        sample (numpy.ndarray):
            A row in the pixel matrix identifying information associated with one pixel
        weights (numpy.ndarray):
            A weight matrix of dimensions [num_x, num_y, num_chans]
        winning_coords (tuple):
            A coordinate array which indicates the winning neuron's position
        sigma (float):
            Determines the spread of the Gaussian neighborhood function, decayed
        learning_rate (float):
            Determines how sensitive the weight updates will be to new data, decayed
        x_mesh (numpy.ndarray):
            The x coordinate matrix of the weights vectors
        y_mesh (numpy.ndarray):
            The y coordinate matrix of the weights vectors

    Returns:
        numpy.ndarray:
            The updated weights matrix
    """

    # return a Gaussian centered around the winning coordinates
    d = 2 * np.pi * sigma * sigma
    ax = np.exp(-np.power(x_mesh - x_mesh.T[winning_coords], 2) / d)
    ay = np.exp(-np.power(y_mesh - y_mesh.T[winning_coords], 2) / d)
    g = (ax * ay).T * learning_rate

    # update the weights based on the Gaussian neighborhood
    new_weights = weights + np.einsum('ij, ijk->ijk', g, sample - weights)

    return new_weights


def train_som(pixel_mat, x_neurons, y_neurons, num_passes,
              sigma=1.0, learning_rate=0.5, random_seed=None):
    """Trains the SOM by iterating through the each data point and updating the params

    Args:
        pixel_mat (pandas.DataFrame):
            A matrix with pixel-level channel information for non-zero pixels in img_xr
        x_neurons (int):
            The number of x neurons to use
        y_neurons (int):
            The number of y neurons to use
        num_passes (int):
            The maximum number of passes to make through the dataset for training
        sigma (float):
            Determines the spread of the Gaussian neighborhood function
        learning_rate (float):
            Determines how sensitive the weight updates will be to new data
        random_seed (int):
            The seed to set for random weight initialization

    Returns:
        numpy.ndarray:
            The weights matrix after training
    """

    # define the random generator
    rand_gen = np.random.RandomState(random_seed)

    # initialize the weights and normalize
    weights = rand_gen.rand(x_neurons, y_neurons, pixel_mat.shape[1]) * 2 - 1
    weights /= np.linalg.norm(weights, axis=-1, keepdims=True)

    # define meshgrid coords for the weights matrix, convert to float (yikes, memory...)
    x_mesh, y_mesh = np.meshgrid(np.arange(x_neurons), np.arange(y_neurons))
    x_mesh = x_mesh.astype(float)
    y_mesh = y_mesh.astype(float)

    # define the number of iterations and the row in pixel_mat corresponding to each iteration
    num_iters = num_passes * pixel_mat.shape[0]
    iter_row = np.arange(num_iters) % pixel_mat.shape[0]

    for t, row in enumerate(iter_row):
        # find the winning neuron's coordinates
        winning_coords = winner(pixel_mat.loc[row, :].values, weights)

        # decay both sigma and learning_rate using the decay function
        decay_sigma = decay_function(sigma, t, num_iters)
        decay_learning_rate = decay_function(learning_rate, t, num_iters)

        # update the weights
        weights = update(pixel_mat.loc[row, :].values, weights,
                         winning_coords, decay_sigma, decay_learning_rate,
                         x_mesh, y_mesh)

    return weights


def cluster_som(pixel_mat, weights, batch_size=10000):
    """Assigns the cluster label to each entry in the pixel matrix based on the trained weights

    Args:
        pixel_mat (pandas.DataFrame):
            A matrix with pixel-level channel information for non-zero pixels in img_xr
        weights (numpy.ndarray):
            The weights matrix after training
        batch_size (int):
            The number of pixels we want to cluster at once

    Returns:
        pandas.Series:
            The cluster labels to assign to the corresponding rows in pixel_mat
    """

    # just in case...
    if batch_size < 1:
        raise ValueError("Batch size provided must be positive")

    # generate the winning coordinates for each sample in batches
    cluster_coords = []
    num_batches = pixel_mat.shape[0] // batch_size

    # generate all the equal-sized batches we can find
    for batch in range(num_batches):
        vals_to_cluster = pixel_mat.loc[batch * batch_size:(batch + 1) * batch_size - 1]
        cluster_batch_coords = batch_winner(vals_to_cluster.values, weights)
        cluster_coords.extend(cluster_batch_coords)

    # didn't divide up evenly, need to process final row(s)
    if pixel_mat.shape[0] % batch_size != 0:
        vals_to_cluster = pixel_mat.loc[num_batches * batch_size:]
        cluster_batch_coords = batch_winner(vals_to_cluster.values, weights)
        cluster_coords.extend(cluster_batch_coords)

    # convert to series, cast to str so the replace function works
    cluster_coords = pd.Series(cluster_coords).astype(str)

    # reassign the coordinates to integers to make the label col more understandable
    unique_cluster_coords = cluster_coords.unique()
    coord_to_label = list(range(len(unique_cluster_coords)))
    cluster_labels = cluster_coords.replace(to_replace=unique_cluster_coords,
                                            value=coord_to_label)

    return cluster_labels
