import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
from scipy.spatial.distance import cdist


# Constants for random distance matrix generation
AB_DIST_MEAN = 100
AB_DIST_VAR = 1

AC_DIST_MEAN = 10
AC_DIST_VAR = 1


# Constants for random centroid matrix generation
A_CENTROID_FACTOR = 0.5
B_CENTROID_FACTOR = 0.6
C_CENTROID_FACTOR = 0.1

A_CENTROID_COV = [[1, 0], [0, 1]]
B_CENTROID_COV = [[1, 0], [0, 1]]
C_CENTROID_COV = [[1, 0], [0, 1]]


def generate_labels(num_A=100, num_B=100, num_C=100):
    """
    This function will generate a set of associated labels for each cell centroid.

    A helper function to get_random_dist_matrix.

    Return value will be a list of values associated with AB and AC, as well as
    a dictionary of mappings of the type cell_num: cell_label.

    Cell label will be either 'AB' or 'AC'

    Args:
        num_A: the number of A cells we wish to generate. Default 100.
        num_B: the number of B cells we wish to generate. Default 100.
        num_C: the number of C cells we wish to generate. Default 100.
    """

    # Generate a range of numbers the same length as the total of A, B, and C labels desired
    # And select a random set of indices to identify as A cells
    num_range = np.arange(num_A + num_B + num_C)
    a_indices = np.random.choice(num_range.size, num_A, replace=False)

    # Get a indices
    a_values = num_range[a_indices]

    # From remaining B and C cells, select a random set of indices for B cells
    non_a_indices = num_range[~a_indices]
    b_indices = np.random.choice(non_a_indices, num_B, replace=False)

    # Get b indices, set the remainder as c indices
    b_indices = non_a_indices[b_indices]
    c_indices = non_a_indices[c_indices]

    a_dict = dict([(a, 'A') for a in a_values])
    b_dict = dict([(b, 'B') for b in b_values])
    c_dict = dict([(c, 'C') for c in c_values])

    label_dict = {**a_dict, **b_dict, **c_dict}

    return a_indices, b_indices, c_indices, label_dict


def direct_init_dist_matrix(num_A=100, num_B=100, num_C=100, distr_AB=None, distr_AC=None):
    """
    This function will return a random dist matrix such that the distance between cells
    of types A and B are overall larger than the distance between cells of types A and C

    Each row and column representing a cell.
    We generate the points using Gaussian distributions
    Ideally, the parameters for A to B distances will be set such that they produce a lower range of values
    than A to C distances.

    Will return a random distance matrix as well as the dictionary of associated cell: label IDs
    The above is generated from the generate_labels function

    Args:
        num_A: the number of A cells we wish to generate. Default 100
        num_B: the number of B cells we wish to generate. Default 100
        num_C: the number of C cells we wish to generate. Default 100
        distr_AB: if specified, will be a dict listing the mean and variance of the Gaussian distribution
            we wish to generate numbers from. If None, use the default values.
        distr_AC: similar to dist_AB. Default will have a higher mean value.
    """

    # generate a list of A, B, and C cells
    labels_a, labels_b, labels_c, dict_labels = generate_labels(num_A, num_B, num_C)

    # initialize the distance matrix
    sample_dist_mat = np.zeros((num_A + num_B + num_C, num_A + num_B + num_C))

    # set the mean and variance of the Gaussian distributions of both AB and AC distances
    if distr_AB = None:
        mean_ab = AB_DIST_MEAN
        var_ab = AB_DIST_VAR
    else:
        mean_ab = distr_AB['mean']
        var_ab = distr_AB['var']

    if distr_AC = None:
        mean_ac = AC_DIST_MEAN
        var_ac = AC_DIST_VAR
    else:
        mean_ac = distr_AC['mean']
        var_ac = distr_AC['var']

    # generate a random numpy matrix of Gaussian values from specified distribution
    # and assign to corresponding labels_a, labels_b values
    random_ab = np.random.normal(mean_ab, var_ab, (num_A, num_B))
    # random_ab = (random_ab + random_ab.T) / 2
    sample_dist_mat[labels_a, labels_b] = random_ab
    sample_dist_mat[labels_b, labels_a] = random_ab.T

    # assert that the created submatrix is symmetric
    assert np.alclose(sample_dist_mat[labels_a, labels_b], sample_dist_mat[labels_a, labels_b].T, rtol=1e-05, atol=1e-08)

    # follow the same steps for labels_a and labels_c
    random_ac = np.random.normal(mean_ac, var_ac, (num_A, num_C))
    random_ac = (random_ac + random_ac.T) / 2
    sample_dist_mat[labels_a, labels_c] = random_ac

    assert np.alclose(sample_dist_mat[labels_a, labels_c], sample_dist_mat[labels_a, labels_c].T, rtol=1e-05, atol=1e-08)

    # we don't care about a-a, b-b, c-c, or b-c distances, so we just return the matrix along with the labels
    return sample_dist_mat, dict_labels


def get_random_centroid_centers(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100, distr_A=None, distr_B=None, distr_C=None):
    """
    This function generates random centroid centers such that those of type A will have centers
    closer on average to those of type B than those of type C

    We will use a multivariate Gaussian distribution for A, B, and C type cells to generate their respective centers.

    Returns the set of points associated with the centroids of cells of types A, B, and C.

    Args:
        size_img: a tuple indicating the size of the image. Default 1024 x 1024
        num_A: the number of A centroids to generate. Default 100.
        num_B: the number of B centroids to generate. Default 100.
        num_C: the number of C centroids to generate. Default 100.

        distr_A: a dict indicating the parameters of the multivariate normal distribution to generate A cell centroid.
            If None, use predefined parameters.
        distr_B: similar to distr_A
        distr_C: similar to distr_C
    """

    height = size_img[0]
    width = size_img[1]

    if distr_A = None:
        a_mean = (height * A_CENTROID_FACTOR, width * A_CENTROID_FACTOR)
        a_cov = A_CENTROID_COV
    else:
        mean_a = distr_A['mean']
        var_a = distr_A['cov']

    if distr_B = None:
        b_mean = (height * B_CENTROID_FACTOR, width * B_CENTROID_FACTOR)
        b_cov = B_CENTROID_COV
    else:
        mean_b = distr_B['mean']
        var_b = distr_B['cov']

    if distr_C = None:
        b_mean = (height * C_CENTROID_FACTOR, width * C_CENTROID_FACTOR)
        b_cov = C_CENTROID_COV
    else:
        mean_ac = distr_C['mean']
        var_ac = distr_C['var']

    a_points = np.random.multivariate_normal(a_mean, a_cov, num_A)
    b_points = np.random.multivariate_normal(b_mean, b_cov, num_B)
    c_points = np.random.multivariate_normal(c_mean, c_cov, num_C)

    return a_points, b_points, c_points

def point_init_dist_matrix(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100, distr_A=None, distr_B=None, distr_C=None):
    """
    This function generates random points using the get_random_centroid_centers function and from that
    generates a distance matrix.

    Each row and column of the matrix represents a specific cell, and the elements represent the distance
    from the respective cell centroids.

    Args:
        size_img: a tuple indicating the size of the image. Default 1024 x 1024.
        num_A: the number of A centroids to generate. Used by the get_random_centroid_centers function. Default 100.
        num_B: similar to num_A
        num_C: similar to num_C
        distr_A: a dict indicating the mean and covariance matrix of the multivariate distribution we pull A centroids from.
            Used by get_random_centroid_centers. Default None, and will use predefined parameters.
        distr_B: dimilar to distr_A
        distr_C: similar to distr_C
    """

    # generate points for cell types A, B, and C
    a_points, b_points, c_points = get_random_centroid_centers(size_img, num_A, num_B, num_C, distr_A, distr_B, distr_C)

    a_thresh = a_points.size
    b_thresh = b_points.size
    c_thresh = c_points.size

    a_a_dist = cdist(a_points, a_points)
    a_b_dist = cdist(a_points, b_points)
    a_c_dist = cdist(a_points, c_points)

    b_b_dist = cdist(b_points, b_points)
    b_c_dist = cdist(b_points, c_points)

    c_c_dist = cdist(c_points, c_points)

    first_row = np.concatenate((a_a_dist, a_b_dist, a_c_dist), axis=1)
    second_row = np.concatenate((a_b_dist.T, b_b_dist, b_c_dist), axis=1)
    third_row = np.concatenate((a_c_dist.T, b_c_dist.T, c_c_dist), axis=1)

    complete_mat = np.concatenate(first_row, second_row, third_row, axis=0)

    return complete_mat