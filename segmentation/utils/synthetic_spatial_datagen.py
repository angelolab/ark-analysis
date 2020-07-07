import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
import xarray as xr
import copy

from segmentation.utils import spatial_analysis_utils as sau
from scipy.spatial.distance import cdist
from skimage.measure import label


def direct_init_dist_matrix(num_A=100, num_B=100, num_C=100,
                            distr_AB={'mean': 100, 'var': 1}, distr_AC={'mean': 20, 'var': 1},
                            seed=None):
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
            we wish to generate numbers from. Default mean=100 and var=1
        distr_AC: similar to dist_AB. Default mean=20 and var=1
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default 42.

    Returns:
        dist_mat: the randomized distance matrix we generate directly from predefined distributions
            where the average distances between cell types of a and b > average distances between
            cell types of b and c
    """

    # set the mean and variance of the Gaussian distributions of both AB and AC distances
    mean_ab = distr_AB['mean']
    var_ab = distr_AB['var']

    mean_ac = distr_AC['mean']
    var_ac = distr_AC['var']

    # set random seed if set
    if seed:
        np.random.seed(seed)

    # we initialize the random distances across different types of points
    # note that we don't really care about aa, bb, bc, or cc, so we
    # initialize those to garbage. We do need them for a proper
    # distance matrix format, however.
    random_aa = np.random.normal(0, 1, (num_A, num_A))
    random_ab = np.random.normal(mean_ab, var_ab, (num_A, num_B))
    random_ac = np.random.normal(mean_ac, var_ac, (num_A, num_C))
    random_bb = np.random.normal(0, 1, (num_B, num_B))
    random_bc = np.random.normal(0, 1, (num_B, num_C))
    random_cc = np.random.normal(0, 1, (num_C, num_C))

    # create each partition one-by-one first
    # we need to correct each aa, bb, and cc matrix to ensure symmetry
    a_partition = np.concatenate(((random_aa + random_aa.T) / 2, random_ab, random_ac), axis=1)
    b_partition = np.concatenate((random_ab.T, (random_bb + random_bb.T) / 2, random_bc), axis=1)
    c_partition = np.concatenate((random_ac.T, random_bc.T, (random_cc + random_cc.T) / 2), axis=1)

    # then concatenate them together
    dist_mat = np.concatenate((a_partition, b_partition, c_partition), axis=0)

    return dist_mat


def point_init_dist_matrix(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100,
                           distr_A={'centroid_factor': (0.5, 0.5), 'cov': [[200, 0], [0, 200]]},
                           distr_B={'centroid_factor': (0.9, 0.9), 'cov': [[200, 0], [0, 200]]},
                           distr_C={'centroid_factor': (0.4, 0.4), 'cov': [[200, 0], [0, 200]]},
                           seed=None):
    """
    This function generates random centroid centers in the form of a label map
    such that those of type A will have centers closer on average to those of type B
    than those of type C

    We will use a multivariate Gaussian distribution for A, B, and C type cells to generate their respective centers.

    Returns the set of points associated with the centroids of cells of types A, B, and C.

    Args:
        size_img: a tuple indicating the size of the image. Default 1024 x 1024
        num_A: the number of A centroids to generate. Default 100.
        num_B: the number of B centroids to generate. Default 100.
        num_C: the number of C centroids to generate. Default 100.

        distr_A: a dict indicating the parameters of the multivariate normal distribution to generate A cell centroids.
            Params:
                centroid_factor: a tuple to determine which number to multiply the height and width by
                    to indicate the center (mean) of the distribution
                cov: in the format [[varXX, varXY], [varYX, varYY]]
        distr_B: similar to distr_A
        distr_C: similar to distr_C
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        sample_img_xr: the data in xarray format containing the randomized label matrix
            based on the randomized centroid centers we generated. The label mat portion
            of sample_img_xr is generated from a randomly initialized set of cell centroids
            where those of type a are on average closer to those of type b than they
            are to those of type c.
    """

    # extract the height and width
    height = size_img[0]
    width = size_img[1]

    a_mean = (height * distr_A['centroid_factor'][0], width * distr_A['centroid_factor'][1])
    a_cov = distr_A['cov']

    b_mean = (height * distr_B['centroid_factor'][0], width * distr_B['centroid_factor'][1])
    b_cov = distr_B['cov']

    c_mean = (height * distr_C['centroid_factor'][0], width * distr_C['centroid_factor'][1])
    c_cov = distr_C['cov']

    # if specified, set the random seed
    if seed:
        np.random.seed(seed)

    # use the multivariate_normal distribution to generate the points
    # because we're passing these into skimage.measure.label, it is important
    # that we convert these to integers beforehand
    # since label only takes a binary matrix
    # we pass the result through the unique function to eliminate any possibility of duplicate points
    # appearing within any of these arrays
    a_points = np.unique(np.random.multivariate_normal(a_mean, a_cov, num_A).astype(np.int16), axis=0)
    b_points = np.unique(np.random.multivariate_normal(b_mean, b_cov, num_B).astype(np.int16), axis=0)
    c_points = np.unique(np.random.multivariate_normal(c_mean, c_cov, num_C).astype(np.int16), axis=0)

    # this ensures that we only keep the points that are not duplicate across different cell types
    points, counts = np.unique(np.concatenate((a_points, b_points, c_points), axis=0), axis=0, return_counts=True)
    non_dup_points = points[counts == 1]

    # an astronomically unlikely error, would occur if for some reason all of the points were labelled as duplicate
    if len(non_dup_points) == 0:
        raise ValueError("Bad run: no unique points generated. Try again, will work next time.")

    rows, cols = zip(*non_dup_points)

    # generate the binary matrix to pass into label_map
    binary_mat = np.zeros(size_img)
    binary_mat[rows, cols] = True

    # generate the label matrix for the image now
    label_mat = label(binary_mat)

    # and create the output to be able to run through calc_dist_matrix
    # for now, I'm assuming that the array returned will have just one fov
    # and we don't know anything about the segmentation labels
    sample_img = np.zeros((1, size_img[0], size_img[1], 1)).astype(np.int16)
    sample_img[0, :, :, 0] = copy.deepcopy(label_mat)
    sample_img_xr = xr.DataArray(sample_img,
                                 coords=[[1], range(size_img[0]), range(size_img[1]), ['segmentation_label']],
                                 dims=['fovs', 'rows', 'cols', 'channels'])

    # and return the xarray to pass into calc_dist_matrix
    return sample_img_xr
