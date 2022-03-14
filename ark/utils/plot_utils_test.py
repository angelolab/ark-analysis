import os
import pytest
import tempfile

import numpy as np
import pandas as pd
import skimage.io as io
import xarray as xr
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


def test_plot_pixel_cell_cluster_overlay():
    sample_img_data = np.random.randint(0, 20, size=(3, 1024, 1024))
    sample_img_xr = xr.DataArray(
        sample_img_data,
        coords=[['fov1', 'fov2', 'fov3'], np.arange(1024), np.arange(1024)],
        dims=['fovs', 'x', 'y']
    )

    # invalid FOVs provided
    with pytest.raises(ValueError):
        plot_utils.plot_pixel_cell_cluster_overlay(
            sample_img_xr, ['fov3', 'fov4'], 'sample_mapping_path.csv', {}
        )

    # invalid mapping path provided
    with pytest.raises(FileNotFoundError):
        plot_utils.plot_pixel_cell_cluster_overlay(
            sample_img_xr, ['fov1', 'fov2'], 'bad_sample_mapping_path.csv', {}
        )

    with tempfile.TemporaryDirectory() as td:
        # define the path to the cluster map
        mapping_path = os.path.join(td, 'sample_mapping_path.csv')

        # invalid columns provided in mapping
        df = pd.DataFrame.from_dict({
            'bad_cluster_col': np.arange(20),
            'metacluster': np.repeat(np.arange(5), 4),
            'mc_name': ['meta' + str(i) for i in np.repeat(np.arange(5), 4)]
        })
        df.to_csv(mapping_path, index=False)

        with pytest.raises(ValueError):
            plot_utils.plot_pixel_cell_cluster_overlay(
                sample_img_xr, ['fov1', 'fov2'], mapping_path, {}
            )

        # rename bad_cluster_col to cluster so it passes that test
        df = df.rename({'bad_cluster_col': 'cluster'}, axis=1)
        df.to_csv(mapping_path, index=False)

        # invalid sample_mapping dict provided, metaclusters do not match
        # those found in mapping_path
        bad_sample_mapping = {i + 2: (0.0, 0.0, 0.0) for i in np.arange(5)}

        with pytest.raises(ValueError):
            plot_utils.plot_pixel_cell_cluster_overlay(
                sample_img_xr, ['fov1', 'fov2'], mapping_path, bad_sample_mapping
            )

        # define a valid mapping
        sample_mapping = {
            i: tuple(np.random.rand(3)) for i in np.arange(5)
        }

        # test 1: save_dir not specified
        plot_utils.plot_pixel_cell_cluster_overlay(
            sample_img_xr, ['fov1', 'fov2'], mapping_path, sample_mapping
        )

        # assert no files created in temp_dir
        for fov in sample_img_xr.fovs.values:
            assert not os.path.exists(os.path.join(td, fov + '.png'))

        # test 2: save_dir specified
        plot_utils.plot_pixel_cell_cluster_overlay(
            sample_img_xr, ['fov1', 'fov2'], mapping_path, sample_mapping,
            save_dir=td
        )

        # assert files only created for fov1`and fov2, not fov3
        assert os.path.exists(os.path.join(td, 'fov1.png'))
        assert os.path.exists(os.path.join(td, 'fov2.png'))
        assert not os.path.exists(os.path.join(td, 'fov3.png'))


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
    fov = 'fov8'

    example_labels = _generate_segmentation_labels((1024, 1024))
    alternate_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 2))

    with tempfile.TemporaryDirectory() as temp_dir:
        # create the whole cell and nuclear segmentation label compartments
        io.imsave(os.path.join(temp_dir, '%s_feature_0.tif' % fov), example_labels,
                  check_contrast=False)
        io.imsave(os.path.join(temp_dir, '%s_feature_1.tif' % fov), example_labels,
                  check_contrast=False)

        # save the cell image
        img_dir = os.path.join(temp_dir, 'img_dir')
        os.mkdir(img_dir)
        io.imsave(os.path.join(img_dir, '%s.tif' % fov), example_images,
                  check_contrast=False)

        # test with both nuclear and membrane specified
        contour_mask = plot_utils.create_overlay(
            fov=fov, segmentation_dir=temp_dir, data_dir=img_dir,
            img_overlay_chans=['nuclear_channel', 'membrane_channel'],
            seg_overlay_comp='whole_cell')

        assert contour_mask.shape == (1024, 1024, 3)

        # test with just nuclear specified
        contour_mask = plot_utils.create_overlay(
            fov=fov, segmentation_dir=temp_dir, data_dir=img_dir,
            img_overlay_chans=['nuclear_channel'],
            seg_overlay_comp='whole_cell')

        assert contour_mask.shape == (1024, 1024, 3)

        # test with nuclear compartment
        contour_mask = plot_utils.create_overlay(
            fov=fov, segmentation_dir=temp_dir, data_dir=img_dir,
            img_overlay_chans=['nuclear_channel', 'membrane_channel'],
            seg_overlay_comp='nuclear')

        assert contour_mask.shape == (1024, 1024, 3)

        # test with an alternate contour
        contour_mask = plot_utils.create_overlay(
            fov=fov, segmentation_dir=temp_dir, data_dir=img_dir,
            img_overlay_chans=['nuclear_channel', 'membrane_channel'],
            seg_overlay_comp='whole_cell',
            alternate_segmentation=alternate_labels)

        assert contour_mask.shape == (1024, 1024, 3)

        # invalid alternate contour provided
        with pytest.raises(ValueError):
            plot_utils.create_overlay(
                fov=fov, segmentation_dir=temp_dir, data_dir=img_dir,
                img_overlay_chans=['nuclear_channel', 'membrane_channel'],
                seg_overlay_comp='whole_cell',
                alternate_segmentation=alternate_labels[:100, :100])
