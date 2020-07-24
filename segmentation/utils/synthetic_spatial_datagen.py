import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
import xarray as xr

from random import seed
from random import random
from scipy.spatial.distance import cdist
from skimage.measure import label
from copy import deepcopy
from skimage.draw import disk

from segmentation.utils import signal_analysis


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
        num_A: the number of A cells we wish to generate. Default 100
        num_B: the number of B cells we wish to generate. Default 100
        num_C: the number of C cells we wish to generate. Default 100
        distr_AB: if specified, will be a tuple listing the mean and variance of the Gaussian distribution
            we wish to generate numbers from. Default mean=10 and var=1
        distr_random: similar to dist_AB, except it's what we set the distribution of
            all other distances to be. Default mean=200 and var=1
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        dist_mat: the randomized distance matrix we generate directly from predefined distributions
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
        size_img: a tuple indicating the size of the image. Default 1024 x 1024
        num_A: the number of A centroids to generate. Default 100.
        num_B: the number of B centroids to generate. Default 100.
        num_C: the number of C centroids to generate. Default 100.

        mean_A_factor: a tuple to determine which number to multiply the height and width by
            to indicate the center (mean) of the distribution to generate A points.
            Will be randomly set to a predefined value if None.
        cov_A: the covariance used to generate A poins in the format [[varXX, varXY], [varYX, varYY]].
            Will be randomly set to a predefined value if None.
        mean_B_factor: similar to mean_A_factor
        cov_B: similar to cov_A
        mean_C_factor: similar to mean_A_factor
        cov_C: similar to cov_A
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        total_points: a list of non-duplicated cell centroids.
    """

    if float_type:
        numpy_type = np.float64
    else:
        numpy_type = np.int16

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
        size_img: a tuple indicating the size of the image. Default 1024 x 1024
        num_A: the number of A centroids to generate. Default 100.
        num_B: the number of B centroids to generate. Default 100.
        num_C: the number of C centroids to generate. Default 100.
        mean_A_factor: a tuple to determine which number to multiply the height and width by
            to indicate the center (mean) of the distribution to generate A points.
            Will be randomly set to a predefined value if None.
        cov_A: the covariance used to generate A poins in the format [[varXX, varXY], [varYX, varYY]].
            Will be randomly set to a predefined value if None.
        mean_B_factor: similar to mean_A_factor
        cov_B: similar to cov_A
        mean_C_factor: similar to mean_A_factor
        cov_C: similar to cov_A
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.

    Returns:
        sample_img_xr: the data in xarray format containing the randomized label matrix
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
    sample_img[0, :, :, 0] = copy.deepcopy(label_mat)
    sample_img_xr = xr.DataArray(sample_img,
                                 coords=[[1], range(size_img[0]), range(size_img[1]), ['segmentation_label']],
                                 dims=['fovs', 'rows', 'cols', 'channels'])

    # and return the xarray to pass into calc_dist_matrix, plus the centroid_indices to readjust it
    return sample_img_xr, centroid_indices


def generate_two_cell_segmentation_mask(size_img=(1024, 1024), radius=10, expressions=None):
    """
    This function is a very basic implementation of generate_test_segmentation_mask
    as defined for 2 cells.

    Args:
        size_img: a tuple specifying the height and width of the image. Default (1024, 1024)
        radius: the radius of the disks we desire to draw. Default 10.
        expressions: whether each cell should be expresssed as nuclear or membrane. 
            Should be a NumPy array determining whether we use nuclear expression or not
            (1 if nuclear, 0 if membrane). Default None which means we'll generate it ourselves.
            Note that the length of expressions should be the same as num_cells.

    Returns:
        sample_mask: a test segmentation mask the dimensions of size_img, each cell labeled
            with a specfic marker label. In addition, we'll be labeling the areas designated
            nuclear or membrane, depending on which analysis we're doing. This is the third
            dimension returned.
    """

    if radius > size_img[0] and radius > size_img[1]:
        raise ValueError("Radius specified is larger than one of the image dimensions")

    if expressions and expressions.size != 2:
        raise ValueError("Expressions list is not of length two")

    # the mask we'll be returning, will contain both the cells and the respective nuclear/membrane markers
    sample_mask = np.zeros((size_img[0], size_img[1], 2), dtype=np.int8)

    # generate the two cells at the top left of the image
    center_1 = (radius, radius)
    center_2 = (radius + radius * 2, radius + radius * 2)

    # draw the coordnates covered for the two cell
    x_coords_cell_1, y_coords_cell_1 = disk(center)
    x_coords_cell_2, y_coords_cell_2 = disk(center)

    # set the markers of the two cells
    sample_mask[x_coords_cell_1, y_coords_cell_1, 0] = 1
    sample_mask[x_coords_cell_2, y_coords_cell_2, 0] = 2

    # group centers in a list
    centers = [center_1, center_2]

    # iterate over centers and expressions list
    for i in range(len(expressions)):
        # membrane-level cell analysis
        if expressions[i] == 0:
            # generate an inner disk of a smaller radius size, call everything outside of this disk
            # but still within the cell in question the membrane
            x_coords_non_memb, y_coords_non_memb = disk(centers[i], int(radius / 2))

            # in the future, we'll probably store these x_coords and y_coords in an array
            # to access rather than have to regenerate again
            x_coords_orig, y_coords_orig = disk(centers[i], int(radius / 2))
            overlay_mask = np.zeros(size_img, dtype=np.int8)

            # set the respective values of the membrane portion of the cell to 1
            overlay_mask[x_coords_orig, y_coords_orig] = 1
            overlay_mask[x_coords_non_memb, y_coords_non_memb] = 0

            # add this mask created to the third dimension of sample_mask to update accordingly
            sample_mask[:, :, 1] += overlay_mask
        # nuclear-level cell analysis
        else:
            # generate an inner disk of a smaller radius size, call this the nucleus
            x_coord_nuc, y_coords_nuc = disk(centers[i], int(radius / 5))
            sample_mask[x_coords_nuc, y_coords_nuc, 1] = 1

    return sample_mask


def generate_test_segmentation_mask(size_img=(1024, 1024), num_cells=2, radius=10,
                                    expressions=None):
    """
    This function generates a random segmentation mask with the assumption that cells are disks.
    For the time being, we just iterate through the image and place random disks next to each other.
    The key is that the cells have to be bordering each other.

    Args:
        size_img: a tuple specifying the height and width of the image. Default (1024, 1024)
        num_cells: the number of cells to generate. Note that the radius parameter
            may override this if there's not enough space. Default 2.
        radius: the radius of the disks we desire to draw. Default 10.
        expressions: whether each cell should be expresssed as nuclear or membrane. 
            Should be a NumPy array determining whether we use nuclear expression or not
            (1 if nuclear, 0 if membrane). Default None which means we'll generate it ourselves.
            Note that the length of expressions should be the same as num_cells.

    Returns:
        sample_mask: a test segmentation mask the dimensions of size_img, each cell labeled
            with a specfic marker label.
    """

    # the mask we'll be returning
    sample_mask = np.zeros(size_img, dtype=np.int8)

    # obviously, you have to have num_cells be at least 2 so we have 2 markers to compare
    if num_cells < 2
        raise ValueError("The parameter num_cells has to be at least 2")

    # if the expressions parameter is specified, we need to assert that the length is the same as num_cells...
    if expressions and expressions.size != num_cells:
        raise ValueError("The expressions list should be the same length as num_cells specified")

    # and the radius set cannot be larger than half of the image
    if radius > size_img[0] / 2 or radius > size_img[1] / 2:
        raise ValueError("Radius value is too large for image")

    # if expressions parameter is not specified, we generate our own list of expressions
    if not expressions:
        expressions = np.random.randint(size=num_cells)

    # we'll start drawing cells in the upper-left of the image
    center_row = radius
    center_col = radius

    # keep a cells_covered counter to learn when 
    cells_covered = deepcopy(num_cells)

    # we'll start by assigning marker_num = 1
    marker_num = 1

    # keep iterating until we're done drawing all the cells or we can no longer fit any more cells
    while cells_covered > 0 and center_row < size_img[0]:
        # generate the x and y coords of the disk, and set the respective values to the marker_num
        x_coords, y_coords = disk((center_x, center_y), radius)
        sample_mask[x_coords, y_coords] = marker_num

        # for now, we just alternate marker_nums per cell
        marker_num = 1 if marker_num == 2 else 2

        # decrease cells_covered by 1
        cells_covered -= 1

        # move the next disk to the right by amount radius * 2 aka diameter
        center_col += radius * 2

        # however, if that puts center_col out of range, restart on the next row on the far left
        if center_col >= size_img[1]:
            center_row += radius * 2
            center_col = radius

    return sample_mask


def generate_test_channel_data(seg_mask, nuclear_labels):
    """
    This function generates test channel data based on a segmentation mask provided
    and whether the data is nuclear or membrane in nature.

    Args:
        seg_mask: a segmentation mask with labeled cells

    Returns:
        channel_data: an array with the channel-based data we're looking at
    """
    pass
