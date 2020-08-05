import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
import xarray as xr
import matplotlib.pyplot as plt

from random import seed
from random import random
from scipy.spatial.distance import cdist
from scipy.stats import norm
from skimage.measure import label
from copy import deepcopy
from skimage.draw import circle
from skimage.draw import circle_perimeter


def generate_test_dist_matrix(num_A=100, num_B=100, num_C=100,
                              distr_AB=(10, 1), distr_random=(200, 1),
                              seed=None):
    """
    This function will return a random dist matrix specifying the distance between cells of
    types A and B and between cells of all other groups (type C).

    Each row and column representing a cell.
    We generate the points using Gaussian distributions
    Ideally, the parameters for A to B distances will be set such that they produce a lower range of values
    than A to C distances.

    Note that these distance matrices created are non-Euclidean.

    Args:
        num_A (int): the number of A cells we wish to generate. Default 100
        num_B (int): the number of B cells we wish to generate. Default 100
        num_C (int): the number of C cells we wish to generate. Default 100
        distr_AB (tuple): if specified, will be a tuple listing the mean and variance of the Gaussian distribution
            we wish to generate numbers from. Default mean=10 and var=1
        distr_random (tuple): similar to dist_AB, except it's what we set the distribution of
            all other distances to be. Default mean=200 and var=1
        seed (int): whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        dist_mat (numpy): the randomized distance matrix we generate directly from predefined distributions
            where the average distances between cell types of a and b > average distances between
            cell types of b and c
    """

    # set the mean and variance of the Gaussian distributions of both AB and AC distances
    mean_ab = distr_AB[0]
    var_ab = distr_AB[1]

    mean_random = distr_random[0]
    var_random = distr_random[1]

    # set random seed if set
    if seed:
        np.random.seed(seed)

    # we initialize the random distances across different types of points
    # note that we don't really care about aa, bb, bc, or cc, so we
    # initialize those to garbage. We do need them for a proper
    # distance matrix format, however.
    random_aa = np.abs(np.random.normal(mean_random, var_random, (num_A, num_A)))
    random_ab = np.abs(np.random.normal(mean_ab, var_ab, (num_A, num_B)))
    random_ac = np.abs(np.random.normal(mean_random, var_random, (num_A, num_C)))
    random_bb = np.abs(np.random.normal(mean_random, var_random, (num_B, num_B)))
    random_bc = np.abs(np.random.normal(mean_random, var_random, (num_B, num_C)))
    random_cc = np.abs(np.random.normal(mean_random, var_random, (num_C, num_C)))

    # create each partition one-by-one first
    # we need to correct each aa, bb, and cc matrix to ensure symmetry
    a_partition = np.concatenate(((random_aa + random_aa.T) / 2, random_ab, random_ac), axis=1)
    b_partition = np.concatenate((random_ab.T, (random_bb + random_bb.T) / 2, random_bc), axis=1)
    c_partition = np.concatenate((random_ac.T, random_bc.T, (random_cc + random_cc.T) / 2), axis=1)

    # then concatenate them together
    dist_mat = np.concatenate((a_partition, b_partition, c_partition), axis=0)

    # finally, fill the diagonals with 0 to ensure a proper distance matrix
    np.fill_diagonal(dist_mat, 0)

    return dist_mat


def generate_random_centroids(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100,
                              mean_A_factor=None, cov_A=None, mean_B_factor=None, cov_B=None,
                              mean_C_factor=None, cov_C=None, seed=None):
    """
    Generate a set of random centroids given distribution parameters.
    Used as a helper function by generate_test_label_map.

    Args:
        size_img (tuple): a tuple indicating the size of the image. Default 1024 x 1024
        num_A (int): the number of A centroids to generate. Default 100.
        num_B (int): the number of B centroids to generate. Default 100.
        num_C (int): the number of C centroids to generate. Default 100.

        mean_A_factor (tuple): a tuple to determine which number to multiply the height and width by
            to indicate the center (mean) of the distribution to generate A points.
            Will be randomly set to a predefined value if None.
        cov_A (numpy): the covariance used to generate A poins in the format [[varXX, varXY], [varYX, varYY]].
            Will be randomly set to a predefined value if None.
        mean_B_factor (tuple): similar to mean_A_factor
        cov_B (numpy): similar to cov_A
        mean_C_factor (tuple): similar to mean_A_factor
        cov_C (numpy): similar to cov_A
        seed (int): whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        total_points (list): a list of non-duplicated cell centroids.
    """

    # extract the height and width
    height = size_img[0]
    width = size_img[1]

    a_mean = (height * mean_A_factor, width * mean_A_factor) if mean_A_factor else (0.5, 0.5)
    a_cov = cov_A if cov_A else [[200, 0], [0, 200]]

    b_mean = (height * mean_B_factor, width * mean_B_factor) if mean_B_factor else (0.6, 0.6)
    b_cov = cov_B if cov_B else [[200, 0], [0, 200]]

    c_mean = (height * mean_C_factor, width * mean_C_factor) if mean_C_factor else (0.1, 0.1)
    c_cov = cov_C if cov_C else [[200, 0], [0, 200]]

    # if specified, set the random seed
    if seed:
        np.random.seed(seed)

    # use the multivariate_normal distribution to generate the points
    # because we're passing these into skimage.measure.label, it is important
    # that we convert these to integers beforehand
    # since label only takes a binary matrix
    a_points = np.random.multivariate_normal(a_mean, a_cov, num_A).astype(np.int16)
    b_points = np.random.multivariate_normal(b_mean, b_cov, num_B).astype(np.int16)
    c_points = np.random.multivariate_normal(c_mean, c_cov, num_C).astype(np.int16)

    # combine the points together into one list
    total_points = np.concatenate((a_points, b_points, c_points), axis=0)

    # remove points with negative values since they're out of range
    total_points = total_points[np.logical_and(total_points[:, 0] >= 0, total_points[:, 1] >= 0), :]

    # remove points with values greater than the size_img dimensions since they're out of range
    total_points = total_points[np.logical_and(total_points[:, 0] < size_img[0], total_points[:, 1] < size_img[1]), :]

    # this ensures that we only keep the points that are not duplicate across different cell types
    non_dup_points, non_dup_counts = np.unique(total_points, axis=0, return_counts=True)
    total_points = non_dup_points[non_dup_counts == 1]

    return total_points


def generate_test_label_map(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100,
                            mean_A_factor=None, cov_A=None, mean_B_factor=None, cov_B=None,
                            mean_C_factor=None, cov_C=None, seed=None):
    """
    This function generates random centroid centers in the form of a label map
    such that those of type A will have centers closer on average to those of type B
    than those of type C

    We will use a multivariate Gaussian distribution for A, B, and C type cells to generate their respective centers.

    Args:
        size_img (tuple): a tuple indicating the size of the image. Default 1024 x 1024
        num_A (int): the number of A centroids to generate. Default 100.
        num_B (int): the number of B centroids to generate. Default 100.
        num_C (int): the number of C centroids to generate. Default 100.
        mean_A_factor (tuple): a tuple to determine which number to multiply the height and width by
            to indicate the center (mean) of the distribution to generate A points.
            Will be randomly set to a predefined value if None.
        cov_A (numpy): the covariance used to generate A poins in the format [[varXX, varXY], [varYX, varYY]].
            Will be randomly set to a predefined value if None.
        mean_B_factor (tuple): similar to mean_A_factor
        cov_B (numpy): similar to cov_A
        mean_C_factor (tuple): similar to mean_A_factor
        cov_C (numpy): similar to cov_A
        seed (int): whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        sample_img_xr (xarray): the data in xarray format containing the randomized label matrix
            based on the randomized centroid centers we generated. The label mat portion
            of sample_img_xr is generated from a randomly initialized set of cell centroids
            where those of type a are on average closer to those of type b than they
            are to those of type c.
    """

    # generate the list of centroids and zip them into x and y coords
    all_centroids = \
        generate_random_centroids(size_img=size_img, num_A=num_A, num_B=num_B, num_C=num_C,
                                  mean_A_factor=mean_A_factor, cov_A=cov_A,
                                  mean_B_factor=mean_B_factor, cov_B=cov_B,
                                  mean_C_factor=mean_C_factor, cov_C=cov_C,
                                  seed=seed)

    point_x_coords, point_y_coords = zip(*all_centroids)

    # all_centroids is ordered specifically to reflect a-labeled centroids first, then b, lastly c
    # unfortunately, when we pass the final label map into calc_dist_matrix of spatial_analysis_utils
    # we lose this desired ordering because of a call to regionprops, which automatically orders
    # the centroids by ascending x-coordinate (in the case of ties, ascending y-coordinate)
    # this messes up the ordering of the distance matrix which screws up, for example, the tests
    # if a user wants to generate a distance matrix from randomly generated centroid points
    # fortunately, lexsort allows us to compute the indices needed to reorder the sorted centroid
    # list back to where they were originally, thus they can also be used to reorder the
    # distance matrix back to the desired partitioning of first a-labeled rows/columns, then b, finally c
    centroid_indices = np.lexsort(all_centroids[:, ::-1].T)

    # generate the label matrix for the image
    # doing it this way because using the label function in skimage does so based on
    # connected components and that messes up the regionprops call in calc_dist_matrix
    # we don't want to assume that we won't get points of distance 1 away from each other
    # so we can just use the labels generated from centroid_indices to assign this
    label_mat = np.zeros(size_img)
    label_mat[point_x_coords, point_y_coords] = centroid_indices + 1

    # now generate the sample xarray
    sample_img = np.zeros((1, size_img[0], size_img[1], 1)).astype(np.int16)
    sample_img[0, :, :, 0] = deepcopy(label_mat)
    sample_img_xr = xr.DataArray(sample_img,
                                 coords=[[1], range(size_img[0]), range(size_img[1]), ['segmentation_label']],
                                 dims=['fovs', 'rows', 'cols', 'channels'])

    # and return the xarray to pass into calc_dist_matrix, plus the centroid_indices to readjust it
    return sample_img_xr, centroid_indices


def generate_two_cell_ring_coords(size_img, center_1, center_2, outer_radius, inner_radius):
    """
    This function generates the coordinates for a ring by generating circle perimeters in the desired range of radii

    Arguments:
        size_img (tuple): the dimensions of the image we wish to generate
        center_1 (tuple): the center of the first cell
        center_2 (tuple): the center of the second cell
        outer_radius (int): the radius of the outer disk of the desired ring
        inner_radius (int): the radius of the inner disk of the desired ring

    Returns:
        ring_region_1 (tuple): a tuple indicating the coordinates for the ring we wish to generate for cell 1
        ring_region_2 (tuple): similar to ring_region_1 for cell 2
    """

    # create numpy arrays to hold the resulting coordinates
    ring_region_1_x = np.array([]).astype(np.int16)
    ring_region_1_y = np.array([]).astype(np.int16)
    ring_region_2_x = np.array([]).astype(np.int16)
    ring_region_2_y = np.array([]).astype(np.int16)

    # for each radius in the range of the inner and outer radii
    for rad in range(inner_radius, outer_radius + 1):
        # generate circle perimeters for region 1 and 2
        region_1_perim_x, region_1_perim_y = circle_perimeter(center_1[0], center_1[1], rad, shape=size_img)
        region_2_perim_x, region_2_perim_y = circle_perimeter(center_2[0], center_2[1], rad, shape=size_img)

        # now add the coordinates to the respective numpy array
        ring_region_1_x = np.concatenate((ring_region_1_x, region_1_perim_x))
        ring_region_1_y = np.concatenate((ring_region_1_y, region_1_perim_y))
        ring_region_2_x = np.concatenate((ring_region_2_x, region_2_perim_x))
        ring_region_2_y = np.concatenate((ring_region_2_y, region_2_perim_y))

    # create a tuple for the x and y coordinates for each cell ring
    ring_region_1 = (ring_region_1_x, ring_region_1_y)
    ring_region_2 = (ring_region_2_x, ring_region_2_y)

    return ring_region_1, ring_region_2


def generate_two_cell_ring_jitter_coords(size_img, ring_region_1, ring_region_2,
                                         jitter_factor, jitter_multiplier, seed):
    """
    This function generates the coordinates we need to set jittered signal for rings
    around the nucleus or membrane respectively

    Arguments:
        size_img (tuple): the dimensions of the image we wish to generate
        ring_region_1 (tuple): a tuple representing the x and y coodinates of the first cell's ring region in question
        ring_region_2 (tuple): similar to ring_region_1 but for cell 2
        jitter_factor (int): controls the amount of random noise we add to the cell
        jitter_multiplier (int): an additional jitter scatter to add more jitter the further away from
            the membrane or nucleus
        seed (int): whether to set the random seed or not, useful for testing

    Returns:
        ring_region_1_jitter (tuple): a tuple with the x and y coordinates to assign jitter for the specified ring for cell 1
        ring_region_2_jitter (tuple): similar to ring_region_1_jitter for ring_region_2
    """

    if seed:
        np.random.seed(seed)

    # generate some jitter for each x and y coordinate
    # the way we do this is to generate random offset based on the jitter_factor multiplier
    # and sampling from a uniform distribution with an additional jitter_factor and jitter_multiplier multiplier
    ring_region_1_x_jitter = (ring_region_1[0] + np.around(np.random.uniform(-jitter_factor * jitter_multiplier, jitter_factor * jitter_multiplier, len(ring_region_1[0])))).astype(np.int16)
    ring_region_1_y_jitter = (ring_region_1[1] + np.around(np.random.uniform(-jitter_factor * jitter_multiplier, jitter_factor * jitter_multiplier, len(ring_region_1[1])))).astype(np.int16)
    ring_region_2_x_jitter = (ring_region_2[0] + np.around(np.random.uniform(-jitter_factor * jitter_multiplier, jitter_factor * jitter_multiplier, len(ring_region_2[0])))).astype(np.int16)
    ring_region_2_y_jitter = (ring_region_2[1] + np.around(np.random.uniform(-jitter_factor * jitter_multiplier, jitter_factor * jitter_multiplier, len(ring_region_2[1])))).astype(np.int16)

    # now ensure the jitter coordinates don't fall out of bounds
    # this we need to do because the offsets could fall out of bounds
    ring_region_1_jitter_indices = (ring_region_1_x_jitter >= 0) & (ring_region_1_x_jitter < size_img[0]) & \
                                   (ring_region_1_y_jitter >= 0) & (ring_region_1_y_jitter < size_img[1])

    ring_region_2_jitter_indices = (ring_region_2_x_jitter >= 0) & (ring_region_2_x_jitter < size_img[0]) & \
                                   (ring_region_2_y_jitter >= 0) & (ring_region_2_y_jitter < size_img[1])

    ring_region_1_x_jitter_coords = ring_region_1_x_jitter[ring_region_1_jitter_indices]
    ring_region_1_y_jitter_coords = ring_region_1_y_jitter[ring_region_1_jitter_indices]
    ring_region_2_x_jitter_coords = ring_region_2_x_jitter[ring_region_2_jitter_indices]
    ring_region_2_y_jitter_coords = ring_region_2_y_jitter[ring_region_2_jitter_indices]

    # add an additional random point removal for clearer results
    # might be a bit overkill but it definitely did help
    # over trying to fine tune jitter factor and what not
    # number of points to keep is inversely proportional to
    # how far away the ring is from the nucleus or membrane respectively
    ring_region_1_selection = np.random.choice(len(ring_region_1_x_jitter_coords), int(len(ring_region_1_x_jitter_coords) / jitter_multiplier))
    ring_region_2_selection = np.random.choice(len(ring_region_2_x_jitter_coords), int(len(ring_region_2_x_jitter_coords) / jitter_multiplier))

    ring_region_1_x_jitter_coords = ring_region_1_x_jitter_coords[ring_region_1_selection]
    ring_region_1_y_jitter_coords = ring_region_1_y_jitter_coords[ring_region_1_selection]
    ring_region_2_x_jitter_coords = ring_region_2_x_jitter_coords[ring_region_2_selection]
    ring_region_2_y_jitter_coords = ring_region_2_y_jitter_coords[ring_region_2_selection]

    # create a tuple for the x and y coordinates for each cell ring
    ring_region_1_jitter = (ring_region_1_x_jitter_coords, ring_region_1_y_jitter_coords)
    ring_region_2_jitter = (ring_region_2_x_jitter_coords, ring_region_2_y_jitter_coords)

    return ring_region_1_jitter, ring_region_2_jitter


def generate_two_cell_nuclear_test_signal_data(size_img, center_1, center_2, nuc_radius,
                                               cell_radius, jitter_factor, num_radii,
                                               plot, seed):
    """
    This function generates sample nuclear-level channel signal data for two bordering cells.

    Arguments:
        size_img (tuple): the dimensions of the image we wish to generate
        center_1 (tuple): the center of the first cell
        center_2 (tuple): the center of the second cell
        nuc_radius (int): the radius of the nucleus
        cell_radius (int): the radius of the entire cell
        jitter_factor (int): controls the amount of random noise we add to the cell
        num_radii (int): will define the number of ring partitions outside of the nucleus or membrane
            the further away the partition is from the nucleus or membrane the more noisy the signal is
        plot (bool): whether to show what was plotted in the function
        seed (int): whether to set the random seed or not, useful for testing

    Returns:
        nuc_channel_data (numpy): a numpy array of dims size_img with the random nuclear-level channel signal data
    """

    # generate the array to hold the nuclear-level channel-based data
    nuc_channel_data = np.zeros(size_img)

    # generate the coordinates of each nuclear disk
    nuc_region_1_x, nuc_region_1_y = circle(center_1[0], center_1[1], nuc_radius, shape=size_img)
    nuc_region_2_x, nuc_region_2_y = circle(center_2[0], center_2[1], nuc_radius, shape=size_img)

    # set each nuclear region to 1
    nuc_channel_data[nuc_region_1_x, nuc_region_1_y] = 1
    nuc_channel_data[nuc_region_2_x, nuc_region_2_y] = 1

    # generate the radii of the surrounding rings of the nucleus
    radii = [int(nuc_radius + (cell_radius - nuc_radius) / num_radii * rad) for rad in range(num_radii + 1)]

    # set a jitter multiplier to increase the base jitter_factor as we get further away from the nucleus
    jitter_multiplier = 1

    for r in range(1, len(radii)):
        # generate each ring region based on the radii list
        ring_region_1, ring_region_2 = \
            generate_two_cell_ring_coords(size_img=size_img, center_1=center_1, center_2=center_2,
                                          outer_radius=radii[r], inner_radius=radii[r - 1])

        # now add some jitter to each ring region
        ring_region_1_jitter, ring_region_2_jitter = \
            generate_two_cell_ring_jitter_coords(size_img=size_img, ring_region_1=ring_region_1,
                                                 ring_region_2=ring_region_2, jitter_factor=jitter_factor,
                                                 jitter_multiplier=jitter_multiplier, seed=seed)

        # set each jitter region to 1
        nuc_channel_data[ring_region_1_jitter[0], ring_region_1_jitter[1]] = 1
        nuc_channel_data[ring_region_2_jitter[0], ring_region_2_jitter[1]] = 1

        # increase the jitter multiplier the further away from the nucleus we get
        jitter_multiplier += 1

    # plot the resulting channel data created
    if plot:
        plt.imshow(nuc_channel_data)
        plt.show()

    return nuc_channel_data


def generate_two_cell_membrane_test_signal_data(size_img, center_1, center_2, memb_radius,
                                                cell_radius, jitter_factor, num_radii,
                                                plot, seed):
    """
    This function generates sample nuclear-level channel signal data for two bordering cells.

    Arguments:
        size_img (tuple): the dimensions of the image we wish to generate
        center_1 (tuple): the center of the first cell
        center_2 (tuple): the center of the second cell
        memb_radius (int): the radius of the membrane
        cell_radius (int): the radius of the entire cell
        jitter_factor (int): controls the amount of random noise we add to the cell
        num_radii (int): will define the number of ring partitions outside of the nucleus or membrane
            the further away the partition is from the nucleus or membrane the more noisy the signal is
        plot (bool): whether to show what was plotted in the function
        seed (int): whether to set the random seed or not, useful for testing

    Returns:
        memb_channel_data (numpy): a numpy array of dims size_img with the random membrane-level channel signal data
    """

    # generate the array to hold the membrane-level channel-based data
    memb_channel_data = np.zeros(size_img)

    # generate the coordinates of each membrane ring
    memb_ring_1, memb_ring_2 = \
        generate_two_cell_ring_coords(size_img=size_img, center_1=center_1, center_2=center_2,
                                      outer_radius=cell_radius, inner_radius=cell_radius - memb_radius)

    # set each membrane region to 1
    memb_channel_data[memb_ring_1[0], memb_ring_1[1]] = 1
    memb_channel_data[memb_ring_2[0], memb_ring_2[1]] = 1

    # generate some jitter around each membrane ring to handle uncertainty around the border
    memb_ring_1_jitter, memb_ring_2_jitter = \
        generate_two_cell_ring_jitter_coords(size_img=size_img, ring_region_1=memb_ring_1,
                                             ring_region_2=memb_ring_2, jitter_factor=jitter_factor,
                                             jitter_multiplier=1, seed=seed)

    # now set each jitter coordinate around the membrane to 1
    memb_channel_data[memb_ring_1_jitter[0], memb_ring_1_jitter[1]] = 1
    memb_channel_data[memb_ring_2_jitter[0], memb_ring_2_jitter[1]] = 1

    # generate the radii of the surrounding rings of the nucleus
    radii = [int((cell_radius - memb_radius) - (cell_radius - memb_radius) / num_radii * rad) for rad in range(num_radii + 1)]

    # set a jitter multiplier to increase the base jitter_factor as we get further away from the nucleus
    jitter_multiplier = 1

    for r in range(1, len(radii)):
        # generate each ring region based on the radii list
        ring_region_1, ring_region_2 = \
            generate_two_cell_ring_coords(size_img=size_img, center_1=center_1, center_2=center_2,
                                          outer_radius=radii[r - 1], inner_radius=radii[r])

        # now add some jitter to each ring region
        ring_region_1_jitter, ring_region_2_jitter = \
            generate_two_cell_ring_jitter_coords(size_img=size_img, ring_region_1=ring_region_1,
                                                 ring_region_2=ring_region_2, jitter_factor=jitter_factor,
                                                 jitter_multiplier=jitter_multiplier, seed=seed)

        # set each jitter region to 1
        memb_channel_data[ring_region_1_jitter[0], ring_region_1_jitter[1]] = 1
        memb_channel_data[ring_region_2_jitter[0], ring_region_2_jitter[1]] = 1

        # increase the jitter multiplier the further away from the nucleus we get
        jitter_multiplier += 1

    # plot th resulting membrane data created
    if plot:
        plt.imshow(memb_channel_data)
        plt.show()

    return memb_channel_data


def generate_two_cell_test_signal_data(size_img=(1024, 1024), cell_radius=200, nuc_radius=35, memb_radius=20,
                                       jitter_factor=5, num_radii=3, seed=None):
    """
    This function generates test channel data assuming we're just generating two cells.

    Args:
        size_img (tuple): the dimensions of the image we wish to generate
        cell_radius (int): the radius of the entire cell
        nuc_radius (int): the radius of the nucleus of each cell
        memb_radius (int): the radius of the membrane of each cell
        jitter_factor (int): controls the amount of random noise we add to the cell
        num_radii (int): will define the number of ring partitions outside of the nucleus or membrane
            the further away the partition is from the nucleus or membrane the more noisy the signal is
        seed (int): whether to fix the random seed or not, useful for testing

    Returns:
        sample_channel_data (numpy): a m x n x p array where m is the number of channels and (n x p)
            is the same as size_img. We'll have 2 channels: nuclear and membrane.
    """

    # define the three-dimensional sample channel array
    sample_channel_data = np.zeros((2, size_img[0], size_img[1]))

    # place the centers in the middle of the image
    # we don't have to worry about out-of-range coordinates because
    # the functions to draw the cells have built in protection
    center_1 = (int(size_img[0] / 2), int(size_img[0] / 2))
    center_2 = (int(size_img[0] / 2), int(size_img[0] / 2 + cell_radius * 2))

    # generate the nuclear-level channel data
    sample_channel_data[0, :, :] = generate_two_cell_nuclear_test_signal_data(size_img=size_img, center_1=center_1,
                                                                              center_2=center_2, nuc_radius=nuc_radius,
                                                                              cell_radius=cell_radius, jitter_factor=jitter_factor,
                                                                              num_radii=num_radii, plot=True, seed=seed)

    # generate the membrane-level channel data
    sample_channel_data[1, :, :] = generate_two_cell_membrane_test_signal_data(size_img=size_img, center_1=center_1,
                                                                               center_2=center_2, memb_radius=memb_radius,
                                                                               cell_radius=cell_radius, jitter_factor=jitter_factor,
                                                                               num_radii=num_radii, plot=True, seed=seed)

    return sample_channel_data
