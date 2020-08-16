import numpy as np
import xarray as xr
import os
import pytest
import tempfile

import skimage.morphology as morph
from skimage.morphology import erosion

from mibidata import mibi_image as mi, tiff

from ark.segmentation import marker_quantification
from ark.utils import data_utils_test

from ark.utils.segmentation_utils_test import _create_test_extraction_data


def test_compute_marker_counts():
    cell_mask, channel_data = _create_test_extraction_data()

    # create xarray containing segmentation mask
    coords = [range(40), range(40), ['whole_cell']]
    dims = ['rows', 'cols', 'compartments']
    segmentation_masks = xr.DataArray(np.expand_dims(cell_mask, axis=-1), coords=coords, dims=dims)

    # create xarray with channel data
    coords = [range(40), range(40), ['chan0', 'chan1', 'chan2', 'chan3', 'chan4']]
    dims = ['rows', 'cols', 'channels']
    input_images = xr.DataArray(channel_data, coords=coords, dims=dims)

    segmentation_output = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks)

    # check that channel 0 counts are same as cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'cell_size'].values,
                          segmentation_output.loc['whole_cell', :, 'chan0'].values)

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'cell_size'].values * 5,
                          segmentation_output.loc['whole_cell', :, 'chan1'].values)

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'chan2'].values,
                          segmentation_output.loc['whole_cell', :, 'chan1'].values)

    # check that only cell1 is negative for channel 3
    assert segmentation_output.loc['whole_cell', :, 'chan3'][1] == 0
    assert np.all(segmentation_output.loc['whole_cell', :, 'chan3'][2:] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output.loc['whole_cell', :, 'chan4'][2] > 0
    assert np.all(segmentation_output.loc['whole_cell', :, 'chan4'][:2] == 0)
    assert np.all(segmentation_output.loc['whole_cell', :, 'chan4'][3:] == 0)

    # check that cell sizes are correct
    sizes = [np.sum(cell_mask == cell_id) for cell_id in [1, 2, 3, 5]]
    assert np.array_equal(sizes, segmentation_output.loc['whole_cell', 1:, 'cell_size'])

    # check that regionprops size matches with cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', 1:, 'cell_size'],
                          segmentation_output.loc['whole_cell', 1:, 'area'])

    # test whole_cell and nuclear compartments with same data
    equal_masks = np.stack((cell_mask, cell_mask), axis=-1)
    # create xarray containing segmentation mask
    coords = [range(40), range(40), ['whole_cell', 'nuclear']]
    dims = ['rows', 'cols', 'compartments']
    segmentation_masks_equal = xr.DataArray(equal_masks, coords=coords, dims=dims)

    segmentation_output_equal = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks_equal,
                                                    nuclear_counts=True)

    assert np.all(segmentation_output_equal[0].values == segmentation_output_equal[1].values)

    # test with different sized nuclear and whole_cell masks

    # nuclear mask is smaller
    nuc_mask = erosion(cell_mask, selem=morph.disk(1))

    unequal_masks = np.stack((cell_mask, nuc_mask), axis=-1)
    coords = [range(40), range(40), ['whole_cell', 'nuclear']]
    dims = ['rows', 'cols', 'compartments']
    segmentation_masks_unequal = xr.DataArray(unequal_masks, coords=coords, dims=dims)

    segmentation_output_unequal = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks_unequal,
                                                    nuclear_counts=True)

    # make sure nuclear segmentations are smaller
    assert np.all(segmentation_output_unequal.loc['nuclear', 1:, 'cell_size'].values <
                  segmentation_output_unequal.loc['whole_cell', 1:, 'cell_size'].values)

    # check that channel 0 counts are same as cell size
    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values,
                          segmentation_output_unequal.loc['nuclear', :, 'chan0'].values)

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values * 5,
                          segmentation_output_unequal.loc['nuclear', :, 'chan1'].values)

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'chan2'].values,
                          segmentation_output_unequal.loc['nuclear', :, 'chan1'].values)

    # check that only cell1 is negative for channel 3
    assert segmentation_output_unequal.loc['nuclear', :, 'chan3'][1] == 0
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'chan3'][2:] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output_unequal.loc['nuclear', :, 'chan4'][2] > 0
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'chan4'][:2] == 0)
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'chan4'][3:] == 0)

    # check that cell sizes are correct
    sizes = [np.sum(nuc_mask == cell_id) for cell_id in [1, 2, 3, 5]]
    assert np.array_equal(sizes, segmentation_output_unequal.loc['nuclear', 1:, 'cell_size'])

    assert np.array_equal(segmentation_output_unequal.loc['nuclear', 1:, 'cell_size'],
                          segmentation_output_unequal.loc['nuclear', 1:, 'area'])


def test_generate_expression_matrix():
    cell_mask, channel_data = _create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask
    cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]

    channel_datas = np.zeros((2, 40, 40, 5), dtype="int16")
    channel_datas[0, :, :, :] = channel_data
    channel_datas[1, 5:, 5:, :] = channel_data[:-5, :-5]

    segmentation_masks = xr.DataArray(cell_masks,
                                      coords=[["Point1", "Point2"], range(40), range(40),
                                              ["whole_cell"]],
                                      dims=["fovs", "rows", "cols", "compartments"])

    channel_data = xr.DataArray(channel_datas,
                                coords=[["Point1", "Point2"], range(40), range(40),
                                        ["chan0", "chan1", "chan2", "chan3", "chan4"]],
                                dims=["fovs", "rows", "cols", "channels"])

    normalized, arcsinh = marker_quantification.generate_expression_matrix(segmentation_masks,
                                                                           channel_data)

    assert normalized.shape[0] == 7

    assert np.all(normalized['chan0'] == np.repeat(1, len(normalized)))
    assert np.all(normalized['chan1'] == np.repeat(5, len(normalized)))
    assert np.all(normalized['chan2'] == normalized['chan2'])


def test_generate_expression_matrix_multiple_compartments():
    cell_mask, channel_data = _create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask
    cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]

    channel_datas = np.zeros((2, 40, 40, 5), dtype="int16")
    channel_datas[0, :, :, :] = channel_data
    channel_datas[1, 5:, 5:, :] = channel_data[:-5, :-5]

    # generate a second set of nuclear masks that are smaller than cell masks
    nuc_masks = np.zeros_like(cell_masks)
    nuc_masks[0, :, :, 0] = erosion(cell_masks[0, :, :, 0], selem=morph.disk(1))
    nuc_masks[1, :, :, 0] = erosion(cell_masks[1, :, :, 0], selem=morph.disk(1))

    # cell 2 in fov0 has no nucleus
    nuc_masks[0, nuc_masks[0, :, :, 0] == 2, 0] = 0

    # all of the nuclei have a label that is 2x the label of the corresponding cell
    nuc_masks *= 2

    unequal_masks = np.concatenate((cell_masks, nuc_masks), axis=-1)
    coords = [["Point0", "Point1"], range(40), range(40), ['whole_cell', 'nuclear']]
    dims = ['fovs', 'rows', 'cols', 'compartments']
    segmentation_masks_unequal = xr.DataArray(unequal_masks, coords=coords, dims=dims)

    channel_data = xr.DataArray(channel_datas,
                                coords=[["Point0", "Point1"], range(40), range(40),
                                        ["chan0", "chan1", "chan2", "chan3", "chan4"]],
                                dims=["fovs", "rows", "cols", "channels"])

    normalized, arcsinh = marker_quantification.generate_expression_matrix(
        segmentation_masks_unequal,
        channel_data,
        nuclear_counts=True)

    # 7 total cells
    assert normalized.shape[0] == 7

    # channel 0 has a constant value of 1
    assert np.all(normalized['chan0'] == np.repeat(1, len(normalized)))

    # channel 1 has a constant value of 5
    assert np.all(normalized['chan1'] == np.repeat(5, len(normalized)))

    # these two channels should be equal for all cells
    assert np.all(normalized['chan1'] == normalized['chan2'])

    # check that cell with missing nucleus has size 0
    index = np.logical_and(normalized['label'] == 2, normalized['fov'] == 'Point0')
    assert normalized.loc[index, 'cell_size_nuclear'].values == 0

    # check that correct nuclear label is assigned to all cells
    normalized_with_nuc = normalized.loc[normalized['label'] != 2, ['label', 'label_nuclear']]
    assert np.all(normalized_with_nuc['label'] * 2 == normalized_with_nuc['label_nuclear'])


def test_compute_complete_expression_matrices():

    # checks if the tree loading is being called correctly when is_mibitiff is False
    # save the actual expression matrix and data loding tests for their respective test functions
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 FOVs and 3 imgs per FOV
        fovs = ["Point1", "Point2", "Point3"]
        imgs = ["img1.tiff", "img2.tiff", "img3.tiff"]

        # define a subset of fovs
        fovs_subset = fovs[:2]

        # since example_dataset exists, lets create a new directory called testing_dataset
        # the rest of the directory structure will be the same
        base_dir = os.path.join(temp_dir, "testing_dataset")
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = os.path.join(input_dir, "single_channel_inputs")
        img_sub_folder = "TIFs"

        # create the directory structure, with a little help from _create_img_dir
        os.mkdir(base_dir)
        os.mkdir(input_dir)
        os.mkdir(tiff_dir)
        data_utils_test._create_img_dir(temp_dir=tiff_dir,
                                        fovs=fovs,
                                        imgs=imgs,
                                        img_sub_folder=img_sub_folder,
                                        dtype="int16")

        # checks that a ValueError is thrown when the user tries to specify points that are not
        # in the original segmentation mask
        with pytest.raises(ValueError):
            # generate a segmentation array with 1 FOV
            cell_masks = np.zeros((1, 50, 50, 1), dtype="int16")
            segmentation_masks = xr.DataArray(cell_masks,
                                              coords=[["Point1"], range(50), range(50),
                                                      ["whole_cell"]],
                                              dims=["fovs", "rows", "cols", "compartments"])

            marker_quantification.compute_complete_expression_matrices(
                segmentation_labels=segmentation_masks, tiff_dir=tiff_dir,
                img_sub_folder=img_sub_folder, is_mibitiff=False, points=["Point1", "Point2"],
                batch_size=5)

        # generate a sample segmentation_mask
        cell_mask, _ = _create_test_extraction_data()
        cell_masks = np.zeros((3, 40, 40, 1), dtype="int16")
        cell_masks[0, :, :, 0] = cell_mask
        cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]
        cell_masks[2, 10:, 10:, 0] = cell_mask[:-10, :-10]
        segmentation_masks = xr.DataArray(cell_masks,
                                          coords=[fovs, range(40), range(40),
                                                  ["whole_cell"]],
                                          dims=["fovs", "rows", "cols", "compartments"])

        # generate sample norm and arcsinh data for all points
        norm_data, arcsinh_data = marker_quantification.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, points=None, batch_size=2)

        assert norm_data.shape[0] > 0 and norm_data.shape[1] > 0
        assert arcsinh_data.shape[0] > 0 and arcsinh_data.shape[1] > 0

        # generate sample norm and arcsinh data for a subset of points
        norm_data, arcsinh_data = marker_quantification.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, points=fovs_subset, batch_size=2)

        assert norm_data.shape[0] > 0 and norm_data.shape[1] > 0
        assert arcsinh_data.shape[0] > 0 and arcsinh_data.shape[1] > 0

    # checks if the loading is being called correctly when is_mibitiff is True
    # save the actual expression matrix and data loding tests for their respective test functions
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 2 FOVs and 2 mibitiff_imgs
        fovs = ["Point1", "Point2", "Point3"]
        mibitiff_imgs = ["Point1.tif", "Point2.tiff", "Point3.tiff"]

        # define a subset of fovs
        fovs_subset = fovs[:2]

        # since example_dataset exists, lets create a new directory called testing_dataset
        # the rest of the directory structure will be the same
        base_dir = os.path.join(temp_dir, "testing_dataset")
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = os.path.join(input_dir, "mibitiff_inputs")

        # create the directory structure
        os.mkdir(base_dir)
        os.mkdir(input_dir)
        os.mkdir(tiff_dir)

        # create sample mibitiff images for each point
        for f, m in zip(fovs, mibitiff_imgs):
            # required metadata for mibitiff writing (double barf)
            METADATA = {
                'run': '20180703_1234_test', 'date': '2017-09-16T15:26:00',
                'coordinates': (12345, -67890), 'size': 500., 'slide': '857',
                'fov_id': f, 'fov_name': 'R1C3_Tonsil',
                'folder': f + '/RowNumber0/Depth_Profile0',
                'dwell': 4, 'scans': '0,5', 'aperture': 'B',
                'instrument': 'MIBIscope1', 'tissue': 'Tonsil',
                'panel': '20170916_1x', 'mass_offset': 0.1, 'mass_gain': 0.2,
                'time_resolution': 0.5, 'miscalibrated': False, 'check_reg': False,
                'filename': '20180703_1234_test', 'description': 'test image',
                'version': 'alpha',
            }

            channels = ["HH3", "Membrane"]
            sample_tif = mi.MibiImage(np.random.rand(40, 40, 2).astype(np.float32),
                                      ((1, channels[0]), (2, channels[1])),
                                      **METADATA)
            tiff.write(os.path.join(tiff_dir, m), sample_tif, dtype=np.float32)

        # generate a sample segmentation_mask
        cell_mask, _ = _create_test_extraction_data()
        cell_masks = np.zeros((3, 40, 40, 1), dtype="int16")
        cell_masks[0, :, :, 0] = cell_mask
        cell_masks[1, 5:, 5:, 0] = cell_mask[:-5, :-5]
        cell_masks[2, 10:, 10:, 0] = cell_mask[:-10, :-10]
        segmentation_masks = xr.DataArray(cell_masks,
                                          coords=[fovs, range(40), range(40),
                                                  ["whole_cell"]],
                                          dims=["fovs", "rows", "cols", "compartments"])

        # generate sample norm and arcsinh data for all points
        norm_data, arcsinh_data = marker_quantification.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=True, points=None, batch_size=2)

        assert norm_data.shape[0] > 0 and norm_data.shape[1] > 0
        assert arcsinh_data.shape[0] > 0 and arcsinh_data.shape[1] > 0

        # generate sample norm and arcsinh data for a subset of points
        norm_data, arcsinh_data = marker_quantification.compute_complete_expression_matrices(
            segmentation_labels=segmentation_masks, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=True, points=fovs_subset, batch_size=2)

        assert norm_data.shape[0] > 0 and norm_data.shape[1] > 0
        assert arcsinh_data.shape[0] > 0 and arcsinh_data.shape[1] > 0
