import os
import pytest
import tempfile


import numpy as np

from ark.utils import plot_utils
from skimage.draw import circle


def test_preprocess_tif():
    example_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    # 2-D tests
    # dimensions are not the same for 2-D example_images
    with pytest.raises(ValueError):
        plot_utils.preprocess_tif(predicted_contour=example_labels[:100, :100],
                                  plotting_tif=example_images[..., 0])

    plotting_tif = plot_utils.preprocess_tif(predicted_countour=example_labels,
                                             plotting_tif=example_images[..., 0])

    # assert the channels all contain the same data
    assert np.all(plotting_tif[:, :, 0] == plotting_tif[:, :, 1])
    assert np.all(plotting_tif[:, :, 1] == plotting_tif[:, :, 2])

    # 3-D tests
    # test for third dimension == 1
    plotting_tif = plot_utils.preprocess_tif(predicted_contour=example_labels,
                                             plotting_tif=example_images[..., 0:1])

    assert np.all(plotting_tif[..., 0] == example_images[..., 0])
    assert np.all(plotting_tif[..., 1] == 0)
    assert np.all(plotting_tif[..., 2] == 0)

    # test for third dimension == 2
    plotting_tif = plot_utils.preprocess_tif(predicted_contour=example_labels,
                                             plotting_tif=example_images[..., 0:2])

    assert np.all(plotting_tif[..., 0] == example_images[..., 0])
    assert np.all(plotting_tif[..., 1] == example_images[..., 1])
    assert np.all(plotting_tif[..., 2] == 0)

    # test for third dimension == 3
    plotting_tif = plot_utils.preprocess_tif(predicted_contour=example_labels,
                                             plotting_tif=example_images)

    assert np.all(plotting_tif[..., 0] == example_images[..., 0])
    assert np.all(plotting_tif[..., 1] == example_images[..., 1])
    assert np.all(plotting_tif[..., 2] == example_images[..., 2])

    # test for third dimension == anything else
    with pytest.raises(ValueError):
        # add another layer to the last dimension
        blank_channel = np.zeros(example_images.shape[:2] + (1,), dtype=example_images.dtype)
        bad_example_images = np.concatenate((example_images, blank_channel), axis=2)

        plot_utils.preprocess_tif(predicted_contour=example_labels,
                                  plotting_tif=example_images)

    # n-D test
    with pytest.raises(ValueError):
        # add a fourth dimension
        plot_utils.preprocess_tif(predicted_countour=example_labels,
                                  plotting_tif=np.expand_dims(example_images, axis=0))


def test_create_overlay():
    example_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    # base test: just contour and tif provided
    contour_mask = plot_utils.plot_overlay(predicted_contour=example_labels,
                                           plotting_tif=example_images,
                                           alternate_contour=None)

    assert contour_mask.shape == (1024, 1024, 3)

    # test with only labels
    contour_mask = plot_utils.plot_overlay(predicted_contour=example_labels,
                                           plotting_tif=None,
                                           alternate_contour=None)

    assert contour_mask.shape == (1024, 1024, 3)

    # test with an alternate contour
    contour_mask = plot_utils.plot_overlay(predicted_contour=example_labels,
                                           plotting_tif=example_images,
                                           alternate_contour=example_labels)

    assert contour_mask.shape == (1024, 1024, 3)

    # invalid alternate contour provided
    with pytest.raises(ValueError):
        plot_utils.plot_overlay(predicted_contour=example_labels,
                                plotting_tif=example_images,
                                alternate_contour=example_labels[:100, :100])


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
