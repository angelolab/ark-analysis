import importlib
import tempfile
import os

import numpy as np
import skimage.io as io

from segmentation.utils import plot_utils, data_utils
from skimage.draw import circle


def _generate_segmentation_labels(img_dims, num_cells=20):
    if len(img_dims) != 2:
        raise ValueError("must be image data of shape [rows, cols]")
    labels = np.zeros(img_dims, dtype="int16")
    radius = 20

    for i in range(num_cells):
        r, c = np.random.randint(radius, img_dims[0] - radius, 2)
        rr, cc = circle(r, c, radius)
        labels[rr, cc] = i

    return labels


def _generate_image_data(img_dims):
    if len(img_dims) != 3:
        raise ValueError("must be image data of [rows, cols, channels]")

    mostly_blank = data_utils.create_blank_channel(img_dims[:2])
    mostly_blank_stack = np.repeat(mostly_blank, img_dims[-1])
    mostly_blank_stack = mostly_blank_stack.reshape(img_dims)

    return mostly_blank_stack


def test_plot_overlay():
    example_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    with tempfile.TemporaryDirectory() as temp_dir:

        # save with both tif and labels
        plot_utils.plot_overlay(predicted_contour=example_labels, plotting_tif=example_images,
                                alternate_contour=None,
                                path=os.path.join(temp_dir, "example_plot1.tiff"))
        # save with just labels
        plot_utils.plot_overlay(predicted_contour=example_labels, plotting_tif=None,
                                alternate_contour=None,
                                path=os.path.join(temp_dir, "example_plot2.tiff"))

        # save with two sets of labels
        plot_utils.plot_overlay(predicted_contour=example_labels, plotting_tif=example_images,
                                alternate_contour=example_labels,
                                path=os.path.join(temp_dir, "example_plot3.tiff"))


def test_randomize_labels():
    labels = _generate_segmentation_labels((1024, 1024))
    randomized = plot_utils.randomize_labels(labels)

    assert np.array_equal(np.unique(labels), np.unique(randomized))

    # check that all pixels to belong to single cell in newly transformed image
    unique_vals = np.unique(labels)
    for val in unique_vals:
        coords = labels == val
        assert len(np.unique(randomized[coords])) == 1


def test_outline_objects():
    labels = _generate_segmentation_labels((1024, 1024), num_cells=300)
    unique_vals = np.unique(labels)[1:]

    # generate a random subset of unique vals to be placed in each list
    vals = np.random.choice(unique_vals, size=60, replace=False)
    object_list = [vals[:20], vals[20:40], vals[40:]]

    outlined = plot_utils.outline_objects(labels, object_list)

    # check that cells in same object list were assigned the same label
    mask1 = np.isin(labels, object_list[0])
    assert len(np.unique(outlined[mask1])) == 1

    mask2 = np.isin(labels, object_list[1])
    assert len(np.unique(outlined[mask2])) == 1

    mask3 = np.isin(labels, object_list[2])
    assert len(np.unique(outlined[mask3])) == 1



