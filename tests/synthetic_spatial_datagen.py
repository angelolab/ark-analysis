from copy import deepcopy

import numpy as np
import xarray as xr
from skimage.draw import disk


def generate_test_dist_matrix(num_A=100, num_B=100, num_C=100,
                              distr_AB=(10, 1), distr_random=(200, 1),
                              seed=None):
    """
    This function will return a random dist matrix specifying the distance between cells of types
    A and B and between cells of all other groups (type C).

    Each row and column representing a cell. We generate the points using Gaussian distributions
    Ideally, the parameters for A to B distances will be set such that they produce a lower range
    of values than A to C distances.

    Note that these distance matrices created are non-Euclidean.

    Args:
        num_A (int):
            the number of A cells we wish to generate. Default 100
        num_B (int):
            the number of B cells we wish to generate. Default 100
        num_C (int):
            the number of C cells we wish to generate. Default 100
        distr_AB (tuple):
            if specified, will be a tuple listing the mean and variance of the Gaussian
            distribution we wish to generate numbers from. Default mean=10 and var=1
        distr_random (tuple):
            similar to dist_AB, except it's what we set the distribution of all other distances to
            be. Default mean=200 and var=1
        seed (int):
            whether to fix the random seed or not. Useful for testing. Should be a specified
            integer value. Default None.

    Returns:
        xarray.DataArray:
            The randomized distance matrix we generate directly from predefined distributions
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
    # completely randomize aa, ac, bb, bc, and cc distances, use params for ab distances
    random_aa = np.abs(np.random.normal(mean_random, var_random, (num_A, num_A)))
    random_ab = np.abs(np.random.normal(mean_ab, var_ab, (num_A, num_B)))
    random_ac = np.abs(np.random.normal(mean_random, var_random, (num_A, num_C)))
    random_bb = np.abs(np.random.normal(mean_random, var_random, (num_B, num_B)))
    random_bc = np.abs(np.random.normal(mean_random, var_random, (num_B, num_C)))
    random_cc = np.abs(np.random.normal(mean_random, var_random, (num_C, num_C)))

    # create each partition one-by-one first and ensure symmetry
    a_partition = np.concatenate(((random_aa + random_aa.T) / 2, random_ab, random_ac), axis=1)
    b_partition = np.concatenate((random_ab.T, (random_bb + random_bb.T) / 2, random_bc), axis=1)
    c_partition = np.concatenate((random_ac.T, random_bc.T, (random_cc + random_cc.T) / 2), axis=1)

    # concatenate partitions together
    dist_mat = np.concatenate((a_partition, b_partition, c_partition), axis=0)

    # ensure a proper dist mat
    np.fill_diagonal(dist_mat, 0)

    # randomly permute dist_mat to make more realistic
    coords_in_order = np.arange(dist_mat.shape[0])
    coords_permuted = deepcopy(coords_in_order)
    np.random.shuffle(coords_permuted)
    dist_mat = dist_mat[np.ix_(coords_permuted, coords_permuted)]

    # 1-index coords because that's where cell labels start at
    coords_dist_mat = [coords_permuted + 1, coords_permuted + 1]
    dist_mat = xr.DataArray(dist_mat, coords=coords_dist_mat)

    return dist_mat


def generate_random_centroids(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100,
                              mean_A_factor=None, cov_A=None, mean_B_factor=None, cov_B=None,
                              mean_C_factor=None, cov_C=None, seed=None):
    """
    Generate a set of random centroids given distribution parameters. Used as a helper function by
    generate_test_label_map.

    Args:
        size_img (tuple):
            a tuple indicating the size of the image. Default (1024, 1024)
        num_A (int):
            the number of A centroids to generate. Default 100.
        num_B (int):
            the number of B centroids to generate. Default 100.
        num_C (int):
            the number of C centroids to generate. Default 100.
        mean_A_factor (tuple):
            a tuple to determine which number to multiply the height and width by to indicate the
            center (mean) of the distribution to generate A points. Will be randomly set to a
            predefined value if None.
        cov_A (numpy.ndarray):
            the covariance used to generate A points as [[varXX, varXY], [varYX, varYY]]. Will be
            randomly set to a predefined value if None.
        mean_B_factor (tuple):
            similar to mean_A_factor
        cov_B (numpy.ndarray):
            similar to cov_A
        mean_C_factor (tuple):
            similar to mean_A_factor
        cov_C (numpy.ndarray):
            similar to cov_A
        seed (int):
            whether to fix the random seed or not. Useful for testing. Should be a specified
            integer value. Default None.

    Returns:
        list:
            List of non-duplicated cell centroids.
    """

    # extract different variable factors
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

    # use the multivariate_normal distribution, convert to int for label mat generation
    a_points = np.random.multivariate_normal(a_mean, a_cov, num_A).astype(np.int16)
    b_points = np.random.multivariate_normal(b_mean, b_cov, num_B).astype(np.int16)
    c_points = np.random.multivariate_normal(c_mean, c_cov, num_C).astype(np.int16)

    # combine points
    total_points = np.concatenate((a_points, b_points, c_points), axis=0)

    # remote out-of-range points
    total_points = total_points[
        np.logical_and(total_points[:, 0] >= 0, total_points[:, 1] >= 0), :]
    total_points = total_points[
        np.logical_and(total_points[:, 0] < size_img[0], total_points[:, 1] < size_img[1]), :]

    # only keep the non-duplicate points
    non_dup_points, non_dup_counts = np.unique(total_points, axis=0, return_counts=True)
    total_points = non_dup_points[non_dup_counts == 1]

    # randomly permute order to make more realistic
    total_points = total_points[np.random.permutation(total_points.shape[0]), :]

    return total_points


def generate_test_label_map(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100,
                            mean_A_factor=None, cov_A=None, mean_B_factor=None, cov_B=None,
                            mean_C_factor=None, cov_C=None, seed=None):
    """
    This function generates random centroid centers in the form of a label map such that those of
    type A will have centers closer on average to those of type B than those of type C

    We will use a multivariate Gaussian distribution for A, B, and C type cells to generate their
    respective centers.

    Args:
        size_img (tuple):
            a tuple indicating the size of the image. Default (1024, 1024)
        num_A (int):
            the number of A centroids to generate. Default 100.
        num_B (int):
            the number of B centroids to generate. Default 100.
        num_C (int):
            the number of C centroids to generate. Default 100.
        mean_A_factor (tuple):
            a tuple to determine which number to multiply the height and width by to indicate the
            center (mean) of the distribution to generate A points. Will be randomly set to a
            predefined value if None.
        cov_A (numpy.ndarray):
            the covariance used to generate A points as [[varXX, varXY], [varYX, varYY]]. Will be
            randomly set to a predefined value if None.
        mean_B_factor (tuple):
            similar to mean_A_factor
        cov_B (numpy.ndarray):
            similar to cov_A
        mean_C_factor (tuple):
            similar to mean_A_factor
        cov_C (numpy.ndarray):
            similar to cov_A
        seed (int):
            whether to fix the random seed or not. Useful for testing. Should be a specified
            integer value. Default None.

    Returns:
        xarray.DataArray:
            Data in xarray format containing the randomized label matrix based on the randomized
            centroid centers we generated. The label mat portion of sample_img_xr is generated
            from a randomly initialized set of cell centroids where those of type a are on average
            closer to those of type b than they are to those of type c.
    """

    # generate the list of centroids
    all_centroids = \
        generate_random_centroids(size_img=size_img, num_A=num_A, num_B=num_B, num_C=num_C,
                                  mean_A_factor=mean_A_factor, cov_A=cov_A,
                                  mean_B_factor=mean_B_factor, cov_B=cov_B,
                                  mean_C_factor=mean_C_factor, cov_C=cov_C,
                                  seed=seed)

    point_x_coords, point_y_coords = zip(*all_centroids)

    # assign each centroid a unique label
    centroid_indices = np.arange(len(all_centroids))
    label_mat = np.zeros(size_img)
    label_mat[point_x_coords, point_y_coords] = centroid_indices + 1

    # generate label mat
    sample_img = np.zeros((1, size_img[0], size_img[1], 1)).astype(np.int16)
    sample_img[0, :, :, 0] = deepcopy(label_mat)
    sample_img_xr = xr.DataArray(
        sample_img,
        coords=[[1], range(size_img[0]), range(size_img[1]), ['segmentation_label']],
        dims=['fovs', 'rows', 'cols', 'channels']
    )

    return sample_img_xr


def generate_two_cell_seg_mask(size_img=(1024, 1024), cell_radius=10):
    """
    This function generates a test segmentation mask with each separate cell labeled separately.

    Args:
        size_img (tuple):
            the dimensions of the image we wish to generate
        cell_radius (int):
            the radius of each cell

    Returns:
        numpy.ndarray:
            An array of dimensions size_img with two separate labeled cells that border each other
    """

    # define the segmentation mask
    sample_segmentation_mask = np.zeros(size_img)

    # define the centers of the cells
    center_1 = (size_img[0] // 2, size_img[0] // 2)
    center_2 = (size_img[0] // 2, size_img[0] // 2 + cell_radius * 2 - 1)

    # generate the coordinates of each nuclear disk
    cell_region_1_x, cell_region_1_y = disk(center=center_1, radius=cell_radius, shape=size_img)

    cell_region_2_x, cell_region_2_y = disk(center=center_2, radius=cell_radius, shape=size_img)

    # assign the respective cells value according to their label
    sample_segmentation_mask[cell_region_1_x, cell_region_1_y] = 1
    sample_segmentation_mask[cell_region_2_x, cell_region_2_y] = 2

    # store the centers and return for reference purposes
    cell_centers = {1: center_1, 2: center_2}

    return sample_segmentation_mask, cell_centers


def generate_two_cell_nuc_signal(segmentation_mask, cell_centers,
                                 size_img=(1024, 1024), nuc_cell_ids=[1],
                                 nuc_radius=3, nuc_signal_strength=10,
                                 nuc_uncertainty_length=0):
    """
    This function generates nuclear signal for the provided cells

    Args:
        segmentation_mask (numpy.ndarray):
            an array which contains the labeled cell regions
        cell_centers (dict):
            a dictionary which contains the centers associated with each cell region
        size_img (tuple):
            the dimensions of the image we wish to generate
        nuc_cell_ids (list):
            a list of cells we wish to generate nuclear signal for, if None assume just cell 1
        nuc_radius (int):
            the radius of the nucleus of each cell
        nuc_signal_strength (int):
            the value we want to assign for nuclear signal
        nuc_uncertainty_length (int):
            will extend nuc_radius by the specified length

    Returns:
        numpy.ndarray:
            An array of equal dimensions to segmentation_mask which have nuclear signal generated
            for the provided cell ids
    """

    # define the nuclear signal array
    sample_nuclear_signal = np.zeros(segmentation_mask.shape)

    for cell in nuc_cell_ids:
        center = cell_centers[cell]

        # generate nuclear region
        nuc_region_x, nuc_region_y = disk(center=center,
                                          radius=nuc_radius + nuc_uncertainty_length,
                                          shape=size_img)

        # set nuclear signal
        sample_nuclear_signal[nuc_region_x, nuc_region_y] = nuc_signal_strength

        # TODO: include jitter (probably after context-spatial randomization is done)

    return sample_nuclear_signal


def generate_two_cell_memb_signal(segmentation_mask, cell_centers,
                                  size_img=(1024, 1024), cell_radius=10,
                                  memb_cell_ids=[2], memb_thickness=5,
                                  memb_signal_strength=10, memb_uncertainty_length=0):
    """
    This function generates membrane signal for the provided cells

    Args:
        segmentation_mask (numpy.ndarray):
            an array which contains the labeled cell regions
        cell_centers (dict):
            a dictionary which contains the centers associated with each cell region
        size_img (tuple):
            the dimensions of the image we wish to generate
        cell_radius (int):
            the radius of the entire cell, needed to do proper circle subtraction for a
            ring-shaped membrane
        memb_cell_ids (list):
            a list of cells we wish to generate nuclear signal for, if None assume just cell 2
        memb_thickness (int):
            the diameter of the membrane ring of each cell
        memb_signal_strength (int):
            the value we want to assign to membrane signal
        memb_uncertainty_length (int):
            will extend memb_radius by the specified length

    Returns:
        numpy.ndarray:
            An array of equal dimensions to segmentation_mask which have membrane signal generated
            for the provided cell ids
    """

    # define the nuclear signal array
    sample_membrane_signal = np.zeros(segmentation_mask.shape)

    for cell in memb_cell_ids:
        center = cell_centers[cell]

        # generate coordinates of the cell region
        cell_region_x, cell_region_y = disk(center=center,
                                            radius=cell_radius + memb_uncertainty_length,
                                            shape=size_img)

        # generate coordinates of the non-membrane region
        non_memb_region_x, non_memb_region_y = disk(center=center,
                                                    radius=cell_radius - memb_thickness,
                                                    shape=size_img)

        # perform circle subtraction to generate membrane region
        sample_membrane_signal[cell_region_x, cell_region_y] = memb_signal_strength
        sample_membrane_signal[non_memb_region_x, non_memb_region_y] = 0

        # TODO: include jitter (probably after context-spatial randomization is done)

    return sample_membrane_signal


def generate_two_cell_chan_data(size_img=(1024, 1024), cell_radius=10,
                                nuc_radius=3, memb_thickness=5, nuc_cell_ids=[1],
                                memb_cell_ids=[2], nuc_signal_strength=10,
                                memb_signal_strength=10,
                                nuc_uncertainty_length=0,
                                memb_uncertainty_length=0):
    """
    This function generates the complete package of channel-level synthetic data we're looking for

    Args:
        size_img (tuple):
            the dimensions of the image we wish to generate
        cell_radius (int):
            the radius of each cell
        nuc_radius (int):
            the radius of each nucleus
        memb_thickness (int):
            the thickness of each membrane
        nuc_cell_ids (list):
            a list of which cells we wish to generate nuclear signal for, if None assume just
            cell 1
        memb_cell_ids (list):
            a list of which cells we wish to generate membrane signal for, if None assume just
            cell 2
        nuc_signal_strength (int):
            defines the constant value we want to assign to nuclear signal
        memb_signal_strength (int):
            defines the constant value we want to assign to membrane signal
        nuc_uncertainty_length (int):
            will extend nuc_radius by specified length
        memb_uncertainty_length (int):
            will extend memb_radius by specified length

    Returns:
        tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray):

        - an array with the labeled cell regions
        - an array defining the nuclear signal for the desired cells
        - an array defining the membrane signal for the desired cells
    """

    # generate the segmentation mask
    sample_segmentation_mask, sample_cell_centers = generate_two_cell_seg_mask(
        size_img=size_img,
        cell_radius=cell_radius
    )

    # generate the nuclear and membrane-level signal
    sample_nuclear_signal = generate_two_cell_nuc_signal(
        segmentation_mask=sample_segmentation_mask,
        cell_centers=sample_cell_centers,
        size_img=size_img,
        nuc_cell_ids=nuc_cell_ids,
        nuc_radius=nuc_radius,
        nuc_signal_strength=nuc_signal_strength,
        nuc_uncertainty_length=nuc_uncertainty_length
    )

    sample_membrane_signal = generate_two_cell_memb_signal(
        segmentation_mask=sample_segmentation_mask,
        cell_centers=sample_cell_centers,
        size_img=size_img,
        cell_radius=cell_radius,
        memb_cell_ids=memb_cell_ids,
        memb_thickness=memb_thickness,
        memb_signal_strength=memb_signal_strength,
        memb_uncertainty_length=memb_uncertainty_length)

    # generate the channel data matrix
    sample_channel_data = np.zeros((size_img[0], size_img[1], 2))
    sample_channel_data[:, :, 0] = sample_nuclear_signal
    sample_channel_data[:, :, 1] = sample_membrane_signal

    return sample_segmentation_mask, sample_channel_data
