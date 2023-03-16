import os
import tempfile
from pathlib import Path

import matplotlib.colors as colors
import natsort
import numpy as np
import pandas as pd
import pytest
import skimage.io as io
import xarray as xr
from skimage.draw import disk
from alpineer import image_utils, test_utils

from ark.utils import plot_utils


def _generate_segmentation_labels(img_dims, num_cells=20):
    if len(img_dims) != 2:
        raise ValueError("must be image data of shape [rows, cols]")
    labels = np.zeros(img_dims, dtype="int16")
    radius = 20

    for i in range(num_cells):
        r, c = np.random.randint(radius, img_dims[0] - radius, 2)
        rr, cc = disk((r, c), radius)
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
            'pixel_meta_cluster': np.repeat(np.arange(5), 4),
            'pixel_meta_cluster_rename': ['meta' + str(i) for i in np.repeat(np.arange(5), 4)]
        })
        df.to_csv(mapping_path, index=False)

        with pytest.raises(ValueError):
            plot_utils.plot_pixel_cell_cluster_overlay(
                sample_img_xr, ['fov1', 'fov2'], mapping_path, {}
            )

        # rename bad_cluster_col to cluster so it passes that test
        df = df.rename({'bad_cluster_col': 'pixel_som_cluster'}, axis=1)
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
        image_utils.save_image(os.path.join(temp_dir, '%s_whole_cell.tiff' % fov), example_labels)
        image_utils.save_image(os.path.join(temp_dir, '%s_nuclear.tiff' % fov), example_labels)

        # save the cell image
        img_dir = os.path.join(temp_dir, 'img_dir')
        os.mkdir(img_dir)
        image_utils.save_image(os.path.join(img_dir, '%s.tiff' % fov), example_images)

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


def test_set_minimum_color_for_colormap():

    cols = ["green", "orange", "gold", "blue"]
    color_map = colors.ListedColormap(cols)

    # check minimum color is defaulted to black
    default_color_map = plot_utils.set_minimum_color_for_colormap(color_map)
    assert default_color_map(0.0) == (0.0, 0.0, 0.0, 1.0)

    # check for specific min color
    new_color_map = plot_utils.set_minimum_color_for_colormap(color_map, (0.1, 0.2, 0.5, 0.3))
    assert new_color_map(0.0) == (0.1, 0.2, 0.5, 0.3)


def test_create_mantis_dir():

    # Number of FOVs
    fov_count = 6

    # Initial data

    example_labels = xr.DataArray([_generate_segmentation_labels((1024, 1024))
                                   for _ in range(fov_count)],
                                  coords=[range(fov_count), range(1024), range(1024)],
                                  dims=["labels", "rows", "cols"])
    example_masks = xr.DataArray(_generate_image_data((1024, 1024, fov_count)),
                                 coords=[range(1024), range(1024), range(fov_count)],
                                 dims=["rows", "cols", "masks"])

    # Misc paths used
    segmentation_dir = "seg_dir"
    mask_dir = "masks"
    cell_output_dir = "cell_output"
    img_data_path = "img_data"
    img_sub_folder = "normalized"

    with tempfile.TemporaryDirectory() as temp_dir:

        # create the folders
        os.makedirs(os.path.join(temp_dir, cell_output_dir))
        os.makedirs(os.path.join(temp_dir, segmentation_dir))
        os.makedirs(os.path.join(temp_dir, cell_output_dir, mask_dir))
        os.makedirs(os.path.join(temp_dir, img_data_path))

        # mantis project path
        mantis_project_path = os.path.join(temp_dir, 'mantis')

        # mask output dir path
        mask_output_dir = os.path.join(temp_dir, cell_output_dir, mask_dir)

        # image data path, create 2 fovs, with 4 channels each
        fovs, channels = test_utils.gen_fov_chan_names(num_fovs=fov_count, num_chans=4,
                                                       use_delimiter=False, return_imgs=False)

        fov_path = os.path.join(temp_dir, img_data_path)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            fov_path, fovs, channels, img_shape=(10, 10), mode='tiff', delimiter=None,
            sub_dir=img_sub_folder, fills=True, dtype=np.int16
        )

        # Loop over the xarray, save each fov's channels,
        # segmentation label compartments, and sample masks
        fovs = data_xr.fovs.values
        fovs_subset = fovs[1:4]

        for idx, fov in enumerate(fovs):
            # Save the segmentation label compartments for each fov
            image_utils.save_image(
                os.path.join(temp_dir, segmentation_dir, '%s_whole_cell_test.tiff' % fov),
                example_labels.loc[idx, ...].values
            )

            # Save the sample masks
            image_utils.save_image(
                os.path.join(mask_output_dir, '%s_mask.tiff' % fov),
                example_masks.loc[..., idx].values
            )

            # Save each channel per fov
            for idx, chan in enumerate(channels):
                image_utils.save_image(
                    filelocs[fov][idx] + ".tiff",
                    data_xr.loc[fov, :, :, chan].values
                )

        # create the mapping path, and the sample mapping file
        mapping_path = os.path.join(temp_dir, cell_output_dir, 'sample_mapping_path.csv')

        df = pd.DataFrame.from_dict({
            'pixel_som_cluster': np.arange(20),
            'pixel_meta_cluster': np.repeat(np.arange(5), 4),
            'pixel_meta_cluster_rename': ['meta' + str(i) for i in np.repeat(np.arange(5), 4)]
        })
        df.to_csv(mapping_path, index=False)

        # The suffix for finding masks
        mask_suffix = "_mask"

        # Image segmentation full path
        image_segmentation_full_path = os.path.join(temp_dir, segmentation_dir)

        # Test mapping csv, and df
        for mapping in [df, mapping_path]:
            plot_utils.create_mantis_dir(
                fovs=fovs_subset,
                mantis_project_path=mantis_project_path,
                img_data_path=fov_path,
                mask_output_dir=mask_output_dir,
                mask_suffix=mask_suffix,
                mapping=mapping,
                seg_dir=image_segmentation_full_path,
                seg_suffix_name="_whole_cell_test.tiff",
                img_sub_folder=img_sub_folder
            )

            # Testing file existence and correctness
            for idx, fov in enumerate(fovs_subset, start=1):
                # output path for testing
                output_path = os.path.join(mantis_project_path, fov)

                # 1. Mask tiff tests
                mask_path = os.path.join(output_path, "population{}.tiff".format(mask_suffix))
                original_mask_path = os.path.join(mask_output_dir, '%s_mask.tiff' % fov)

                # 1.a. Assert that the mast path exists
                assert os.path.exists(mask_path)
                mask_img = io.imread(mask_path)
                # original_mask_img = io.imread(original_mask_path)
                original_mask_img = example_masks.loc[..., idx].values
                # 1.b. Assert that the mask is the same as the original mask
                np.testing.assert_equal(mask_img, original_mask_img)

                # 2. Cell Segmentation tiff tests
                cell_seg_path = os.path.join(output_path, "cell_segmentation.tiff")
                # 2.a. Assert that the segmentation label compartments exist in the new directory
                assert os.path.exists(cell_seg_path)
                original_cell_seg_path = os.path.join(temp_dir, segmentation_dir,
                                                      '%s_whole_cell_test.tiff' % fov)
                cell_seg_img = io.imread(cell_seg_path)
                original_cell_seg_img = io.imread(original_cell_seg_path)
                # 2.b. Assert that the `cell_segmentation` file is equal to `fov#_whole_cell`
                np.testing.assert_equal(cell_seg_img, original_cell_seg_img)

                # 3. mapping csv tests
                if type(mapping) is pd.DataFrame:
                    original_mapping_df = df
                else:
                    original_mapping_df = pd.read_csv(mapping_path)
                new_mapping_df = pd.read_csv(
                    os.path.join(output_path, "population{}.csv".format(mask_suffix)))

                # 3.a. Assert that metacluster col equals the region_id col
                metacluster_col = original_mapping_df[["pixel_meta_cluster"]]
                region_id_col = new_mapping_df[["region_id"]]
                metacluster_col.eq(region_id_col)

                # 3.b. Assert that mc_name col equals the region_name col
                mc_name_col = original_mapping_df[["pixel_meta_cluster_rename"]]
                region_name = new_mapping_df[["region_name"]]
                mc_name_col.eq(region_name)

                mantis_fov_channels = natsort.natsorted(list(Path(output_path).glob("chan*.tiff")))

                # 4. Test that all fov channels exist and are correct
                for chan_path in mantis_fov_channels:
                    new_chan = io.imread(chan_path)

                    # get the channel name
                    chan, _ = chan_path.name.split('.')
                    original_chan = data_xr.loc[fov, :, :, chan].values
                    np.testing.assert_equal(new_chan, original_chan)
