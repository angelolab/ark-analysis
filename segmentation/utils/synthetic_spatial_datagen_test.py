import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
from scipy.spatial.distance import cdist
from segmentation.utils import synthetic_spatial_datagen

import importlib
importlib.reload(synthetic_spatial_datagen)


def test_generate_test_dist_matrix():
    # this function tests the functionality of a random initialization of a
    # distance matrix directly.
    # the tests are fairly basic for now but will be expanded on.
    # for now, we just want to ensure two things:
        # matrix symmetry: a property of distance matrices
        # mean AB > mean AC: the mean distance between cells of types A and B
        # is greater than that of types A and C

    # we'll be using the default parameters provided in the functions
    # except for the random seed, which we specify
    sample_dist_mat = synthetic_spatial_datagen.generate_test_dist_matrix()

    # assert matrix symmetry
    assert np.allclose(sample_dist_mat, sample_dist_mat.T, rtol=1e-05, atol=1e-08)

    # assert the average of the distance between A and B is smaller
    # than the average of the distance between A and C.
    # this may not eliminate the possibility that the null is proved true
    # but it's definitely a great check that can ensure greater success
    assert sample_dist_mat[:100, 100:200].mean() < sample_dist_mat[:100, 200:].mean()


def test_generate_random_centroids():
    # this function tests the functionality of a random initialization of centroids
    # the tests are fairly basic for now but will be expanded on
    # for the time being, all we do are the following:
        # test that there are no duplicates in centroid_list
        # test that all the centroids generated are in range

    # generate some sample stats to pass into generate_random_centroids to get our values
    size_img = (1024, 1024)

    num_A = 100
    num_B = 100
    num_C = 100

    mean_A_factor = 0.5
    mean_B_factor = 0.7
    mean_C_factor = 0.3

    cov_A = [[100, 0], [0, 100]]
    cov_B = [[100, 0], [0, 100]]
    cov_c = [[100, 0], [0, 100]]

    centroid_list = synthetic_spatial_datagen.generate_random_centroids(size_img=size_img, num_A=num_A, num_B=num_B, num_C=num_C,
                                                                        mean_A_factor=mean_A_factor, cov_A=cov_A,
                                                                        mean_B_factor=mean_B_factor, cov_B=cov_B,
                                                                        mean_C_factor=mean_C_factor, cov_C=cov_c)

    # try to extract non-duplicate centroids in the list
    _, centroid_counts = np.unique(centroid_list, axis=0, return_counts=True)
    non_dup_centroids = centroid_list[centroid_counts > 1]

    # assert that there are no duplicates in the list
    assert len(non_dup_centroids) == 0

    # separate x and y coords
    x_coords = centroid_list[:, 0]
    y_coords = centroid_list[:, 1]

    # assert the x and y coordinates are in range
    assert len(x_coords[(x_coords < 0) & (x_coords >= size_img[0])]) == 0
    assert len(y_coords[(y_coords < 0) & (y_coords >= size_img[0])]) == 0


def test_generate_test_label_map():
    # this function tests the general nature of the random label map generated
    # the crux of which are the centroids generated from test_generate_random_centroids
    # the tests are fairly basic for now but will be expanded on
    # for now, all we're doing is checking if we're labeling each centroid with a unique label
    # we also check and see that each label appears in the label_map

    # generate test data
    sample_img_xr, centroid_indices = synthetic_spatial_datagen.generate_test_label_map()

    # all we're looking at is the label map
    # we flatten and remove all non-centroids for testing purposes
    label_map = np.stack(sample_img_xr[0, :, :, 0])
    label_map_flat = label_map.flatten()
    label_map_flat = label_map_flat[label_map_flat > 0]

    # need to assert that we're labeling all centroids with a unique id
    _, label_map_id_counts = np.unique(label_map_flat, return_counts=True)

    assert len(label_map_flat[label_map_id_counts > 1]) == 0

    # also need to assert that each of our labels is being assigned to a centroid
    # need to add 1 to centroid_indices because those values are 1-less due to
    # needing to index arrays, we couldn't 0-index the label values in label_map
    # because values of 0 in a label map are ignored by regionprops
    assert (np.sort(label_map_flat) == np.sort(centroid_indices) + 1).all()


def test_generate_two_cell_segmentation_mask():
    # this function tests if we have generated two cells with two different marker labels
    # that border each other, as well as if their nuclear/membrane hotspots were correctly labeled

    size_img = (1024, 1024)
    radius = 10
    expressions = [1, 0]

    # generate test data
    sample_mask = synthetic_spatial_datagen.generate_two_cell_segmentation_mask(
        size_img=size_img, radius=radius, expressions=expressions)

    # separate the cell and the hot spots aka th nuclar/membrane portion we wish to analyze further
    sample_mask_cell_label = sample_mask[:, :, 0]
    sample_mask_hot_spots = sample_mask[:, :, 1]

    # assert that we have two labels for cells in the array: 1 and 2
    unique_cell_labels = np.sort(np.unique(sample_mask.flatten()[sample_mask.flatten() > 0]))
    assert (unique_cell_labels == np.array([1, 2])).all()

    # assert that the cells border each other
    border_1 = (radius * 2, radius * 2)
    border_2 = (radius * 2 + 1, radius * 2 + 1)

    assert sample_mask_cell_label[border_1[0], border_1[1]] == 1
    assert sample_mask_cell_label[border_2[0], border_2[1]] == 2

    # assert that we have correct nuclear and membrane representation
    center_1 = (radius, radius)
    center_2 = (radius * 3 + 2, radius * 3 + 2)

    # start with nuclear representation
    # test that the center is marked and that the border is set properly
    assert sample_cell_hot_spots[center_1[0], center_1[1]] == 1
    assert sample_cell_hot_spots[center_1[0] - (int(radius / 5) + 1), center_1[1]] == 1
    assert sample_cell_hot_spots[center_1[0] - int(radius / 5), center_1[1]] == 0

    # move on to membrane representation
    # test that the border is marked and that the inner edge of the cell is set properly
    assert sample_cell_hot_spots[border_2[0], border_2[1]] == 1
    assert sample_cell_hot_spots[border_2[0] - 1, border_2[1]] == 0
    assert sample_cell_hot_spots[center_2[0] - int(radius / 2) - 1, center_2[1]] == 1
    assert sample_cell_hot_spots[center_2[0] - int(radius / 2), center_2[1]] == 0


def test_generate_test_segmentation_mask():
    # this function tests that we're generating proper segmentation masks
    # for phase 2 of testing
    pass
