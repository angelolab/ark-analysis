import tempfile
import os

import numpy as np
import pytest

from ark.utils import plot_utils
from skimage.draw import circle

from ark.utils.plot_utils import plot_clustering_result


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

    return np.random.randint(low=0, high=100, size=img_dims)


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
