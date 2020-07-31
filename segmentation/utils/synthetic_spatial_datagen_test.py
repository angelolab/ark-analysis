import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os

from scipy.spatial.distance import cdist
from skimage.draw import circle

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


def test_generate_two_cell_test_regions():
    # this function tests if we're genering our test regions correctly
    # for now, the tests are very basic but they assume that
    # we return a dictionary with the same keys for both
    # and that the circles we draw have the same number of coordinates

    size_img = (1024, 1024)
    radius = 10

    sample_cell_regions, sample_cell_centers = synthetic_spatial_datagen.generate_two_cell_test_regions(
        size_img=size_img, radius=radius)
    labels_regions = list(sample_cell_regions.keys())
    labels_centers = list(sample_cell_centers.keys())

    # assert both the two markers constraint and the same labels restraint
    assert set([1, 2]) == set(labels_regions) == set(labels_centers)

    # assert that we're generating the same number of coordinates for both marker 1 and marker 2
    # this we can assume because the radius is the same for both
    assert len(sample_cell_regions[1][0]) == len(sample_cell_regions[1][1]) == \
        len(sample_cell_regions[2][0]) == len(sample_cell_regions[2][1])


def test_generate_two_cell_test_signal_data():
    # this function tests if we're genering our signal data correctly
    # for now, all we do is check if we're generating only positive values

    size_img = (1024, 1024)
    radius = 10
    expression = 'nuclear'
    pattern = 'nuclear'

    # generate the two cells at the top left of the image
    center_1 = (radius, radius)
    center_2 = (radius, radius * 3 + 1)

    # draw the coordnates covered for the two cell
    x_coords_cell_1, y_coords_cell_1 = circle(center_1[0], center_1[1], radius + 1)
    x_coords_cell_2, y_coords_cell_2 = circle(center_2[0], center_2[1], radius + 1)

    # now store the regions generated by skimage.draw.circle into a dict with the key
    # identifying the marker
    cell_regions = {1: (x_coords_cell_1, y_coords_cell_1),
                    2: (x_coords_cell_2, y_coords_cell_2)}

    # now store the centers of each marker in a cell
    # we're not storing the radius because that parameter is passed into generate_test_channel_data
    # which calls this function, meaning it is already known
    cell_centers = {1: center_1, 2: center_2}

    sample_signal_data = synthetic_spatial_datagen.generate_two_cell_test_signal_data(
        size_img=size_img, radius=radius, expression=expression, pattern=pattern,
        cell_region=cell_regions[1], cell_center=cell_centers[1])

    # assert that we only have positive values or 0 values in the array for nuclear expression
    assert (sample_signal_data >= 0).all()

    # ditto for membrane expression/pattern
    expression = 'membrane'
    pattern = 'membrane'

    sample_signal_data = synthetic_spatial_datagen.generate_two_cell_test_signal_data(
        size_img=size_img, radius=radius, expression=expression, pattern=pattern,
        cell_region=cell_regions[1], cell_center=cell_centers[1])

    assert (sample_signal_data >= 0).all()


def test_generate_two_cell_test_channel_data():
    # this function tests if we're generating our multi-dimensional channel-level data correctly
    # here's where we include checks for borders and whether or not our nuclear/membrane
    # values look like a mountain/valley

    size_img = (1024, 1024)
    radius = 10

    sample_channel_data, _, _ = synthetic_spatial_datagen.generate_two_cell_test_channel_data(
        size_img=size_img, radius=radius)

    nuclear_channel_data = sample_channel_data[0, :, :]
    membrane_channel_data = sample_channel_data[1, :, :]

    # assert that the cells border each other
    # this we will do by checking whether the values at the border are both greater than 0
    # note that because we know how the centers are being generated we can just use radius
    # to help us calculate the borders and by extnsion the centers
    border_1 = (radius, radius * 2)
    border_2 = (radius, radius * 2 + 1)

    assert nuclear_channel_data[border_1[0], border_1[1]] > 0
    assert nuclear_channel_data[border_2[0], border_2[1]] > 0
    assert membrane_channel_data[border_1[0], border_1[1]] > 0
    assert membrane_channel_data[border_2[0], border_2[1]] > 0

    # now assert that the nuclear data was generated like a mountain
    # and the membrane data was generated like a valley
    # this we do by comparing the values taken from the center and border
    center_1 = (radius, radius)
    center_2 = (radius, radius * 3 + 1)

    assert nuclear_channel_data[center_1[0], center_1[1]] > nuclear_channel_data[border_1[0], border_1[1]]
    assert membrane_channel_data[center_2[0], center_2[1]] < membrane_channel_data[border_1[0], border_1[1]]
