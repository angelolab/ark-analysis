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

    mean_A_factor = (0.5, 0.5)
    mean_B_factor = (0.7, 0.7)
    mean_C_factor = (0.3, 0.3)

    cov_A = [[100, 0], [0, 100]]
    cov_B = [[100, 0], [0, 100]]
    cov_c = [[100, 0], [0, 100]]

    centroid_list = generate_random_centroids(size_img=size_img, num_A=num_A, num_B=num_B, num_C=num_C,
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

    assert len(x_coords[(x_coords < 0) & (x_coords >= size_img[0])]) == 0
    assert len(y_coords[(y_coords < 0) & (y_coords >= size_img[0])]) == 0

def test_generate_test_label_map():
    # this function tests the general nature of the random label map generated
    # the crux of which are the centroids generated from test_generate_random_centroids
    # the tests are fairly basic for now but will be expanded on
    # for now, all we're doing is checking if we're labeling each centroid with a unique label
    # we also check and see that each label appears in the label_map

    # generate test data
    sample_img_xr, centroid_indices = generate_test_label_map()

    # all we're looking at is the label map
    # we flatten and remove all non-centroids for testing purposes
    label_map = sample_img_xr[:, 'rows', 'cols', :]
    label_map_flat = label_map.flatten()
    label_map_flat = label_map_flat[label_map_flat > 0]

    # need to assert that we're labeling all centroids with a unique id
    _, label_map_id_counts = np.unique(label_map, return_counts=True)
    assert len(label_map_flat[label_map_id_counts > 1]) == 0

    # also need to assert that each of our labels is being assigned to a centroid
    # need to add 1 to centroid_indices because those values are 1-less due to
    # needing to index arrays, we couldn't 0-index the label values in label_map
    # because values of 0 in a label map are ignored by regionprops
    assert np.sort(label_map_flat) == np.sort(centroid_indices) + 1
