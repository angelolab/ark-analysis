import os
import pytest
import tempfile

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


def test_tif_overlay_preprocess():
    example_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    # 2-D tests
    # dimensions are not the same for 2-D example_images
    with pytest.raises(ValueError):
        plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels[:100, :100],
                                          plotting_tif=example_images[..., 0])

    plotting_tif = plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels,
                                                     plotting_tif=example_images[..., 0])

    # assert the channels all contain the same data
    assert np.all(plotting_tif[:, :, 0] == 0)
    assert np.all(plotting_tif[:, :, 1] == 0)
    assert np.all(plotting_tif[:, :, 2] == example_images[..., 0])

    # 3-D tests
    # test for third dimension == 1
    plotting_tif = plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels,
                                                     plotting_tif=example_images[..., 0:1])

    assert np.all(plotting_tif[..., 0] == 0)
    assert np.all(plotting_tif[..., 1] == 0)
    assert np.all(plotting_tif[..., 2] == example_images[..., 0])

    # test for third dimension == 2
    plotting_tif = plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels,
                                                     plotting_tif=example_images[..., 0:2])

    assert np.all(plotting_tif[..., 0] == 0)
    assert np.all(plotting_tif[..., 1] == example_images[..., 1])
    assert np.all(plotting_tif[..., 2] == example_images[..., 0])

    # test for third dimension == 3
    plotting_tif = plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels,
                                                     plotting_tif=example_images)

    assert np.all(plotting_tif[..., 0] == example_images[..., 2])
    assert np.all(plotting_tif[..., 1] == example_images[..., 1])
    assert np.all(plotting_tif[..., 2] == example_images[..., 0])

    # test for third dimension == anything else
    with pytest.raises(ValueError):
        # add another layer to the last dimension
        blank_channel = np.zeros(example_images.shape[:2] + (1,), dtype=example_images.dtype)
        bad_example_images = np.concatenate((example_images, blank_channel), axis=2)

        plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels,
                                          plotting_tif=bad_example_images)

    # n-D test (n > 3)
    with pytest.raises(ValueError):
        # add a fourth dimension
        plot_utils.tif_overlay_preprocess(segmentation_labels=example_labels,
                                          plotting_tif=np.expand_dims(example_images, axis=0))


def test_create_overlay():
    example_labels = _generate_segmentation_labels((1024, 1024))
    alternate_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    # base test: just contour and tif provided
    contour_mask = plot_utils.create_overlay(segmentation_labels=example_labels,
                                             plotting_tif=example_images,
                                             alternate_segmentation=None)

    assert contour_mask.shape == (1024, 1024, 3)

    # test with an alternate contour
    contour_mask = plot_utils.create_overlay(segmentation_labels=example_labels,
                                             plotting_tif=example_images,
                                             alternate_segmentation=alternate_labels)

    assert contour_mask.shape == (1024, 1024, 3)

    # invalid alternate contour provided
    with pytest.raises(ValueError):
        plot_utils.create_overlay(segmentation_labels=example_labels,
                                  plotting_tif=example_images,
                                  alternate_segmentation=alternate_labels[:100, :100])
