import numpy as np
import os
import pytest
import tempfile

import skimage.morphology as morph
from skimage.morphology import erosion

from ark.segmentation import marker_quantification
from ark.utils import test_utils


def test_compute_marker_counts():

    cell_mask, channel_data = test_utils.create_test_extraction_data()

    segmentation_masks = test_utils.make_labels_xarray(label_data=cell_mask,
                                                       compartment_names=['whole_cell'])

    input_images = test_utils.make_images_xarray(channel_data)

    # test utils output is 4D but tests require 3D
    segmentation_masks, input_images = segmentation_masks[0], input_images[0]

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
    assert segmentation_output.loc['whole_cell', 1, 'chan3'] == 0
    assert np.all(segmentation_output.loc['whole_cell', 2:, 'chan3'] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output.loc['whole_cell', 2, 'chan4'] > 0
    assert np.all(segmentation_output.loc['whole_cell', :1, 'chan4'] == 0)
    assert np.all(segmentation_output.loc['whole_cell', 3:, 'chan4'] == 0)

    # check that cell sizes are correct
    sizes = [np.sum(cell_mask == cell_id) for cell_id in [1, 2, 3, 5]]
    assert np.array_equal(sizes, segmentation_output.loc['whole_cell', :, 'cell_size'])

    # check that regionprops size matches with cell size
    assert np.array_equal(segmentation_output.loc['whole_cell', :, 'cell_size'],
                          segmentation_output.loc['whole_cell', :, 'area'])

    # test whole_cell and nuclear compartments with same data
    segmentation_masks_equal = test_utils.make_labels_xarray(
        label_data=np.concatenate((cell_mask, cell_mask), axis=-1),
        compartment_names=['whole_cell', 'nuclear']
    )

    # test utils output is 4D but tests require 3D
    segmentation_masks_equal = segmentation_masks_equal[0]

    segmentation_output_equal = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks_equal,
                                                    nuclear_counts=True)

    assert np.all(segmentation_output_equal[0].values == segmentation_output_equal[1].values)

    # test with different sized nuclear and whole_cell masks

    # nuclear mask is smaller
    nuc_mask = \
        np.expand_dims(erosion(cell_mask[0, :, :, 0], selem=morph.disk(1)), axis=0)
    nuc_mask = np.expand_dims(nuc_mask, axis=-1)

    unequal_masks = np.concatenate((cell_mask, nuc_mask), axis=-1)
    segmentation_masks_unequal = test_utils.make_labels_xarray(
        label_data=unequal_masks,
        compartment_names=['whole_cell', 'nuclear']
    )

    # test utils output is 4D but tests require 3D
    segmentation_masks_unequal = segmentation_masks_unequal[0]

    segmentation_output_unequal = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks_unequal,
                                                    nuclear_counts=True)

    # make sure nuclear segmentations are smaller
    assert np.all(segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values <
                  segmentation_output_unequal.loc['whole_cell', :, 'cell_size'].values)

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
    assert segmentation_output_unequal.loc['nuclear', 1, 'chan3'] == 0
    assert np.all(segmentation_output_unequal.loc['nuclear', 2:, 'chan3'] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output_unequal.loc['nuclear', 2, 'chan4'] > 0
    assert np.all(segmentation_output_unequal.loc['nuclear', :1, 'chan4'] == 0)
    assert np.all(segmentation_output_unequal.loc['nuclear', 3:, 'chan4'] == 0)

    # check that cell sizes are correct
    sizes = [np.sum(nuc_mask == cell_id) for cell_id in [1, 2, 3, 5]]
    assert np.array_equal(sizes, segmentation_output_unequal.loc['nuclear', :, 'cell_size'])

    assert np.array_equal(segmentation_output_unequal.loc['nuclear', :, 'cell_size'],
                          segmentation_output_unequal.loc['nuclear', :, 'area'])

    # different object properties can be supplied
    regionprops_features = ['label', 'area']
    excluded_defaults = ['eccentricity']
    segmentation_output_specified = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks_equal,
                                                    nuclear_counts=True,
                                                    regionprops_features=regionprops_features)

    assert np.all(np.isin(['label', 'area'], segmentation_output_specified.features.values))

    assert not np.any(np.isin(excluded_defaults, segmentation_output_specified.features.values))

    # these nuclei are all smaller than the cells, so we should get same result
    segmentation_output_specified_split = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_masks=segmentation_masks_equal,
                                                    nuclear_counts=True,
                                                    regionprops_features=regionprops_features,
                                                    split_large_nuclei=True)

    assert np.all(segmentation_output_specified_split == segmentation_output_specified)


def test_generate_expression_matrix():

    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask[0, :, :, 0]
    cell_masks[1, 5:, 5:, 0] = cell_mask[0, :-5, :-5, 0]

    tif_data = np.zeros((2, 40, 40, 5), dtype="int16")
    tif_data[0, :, :, :] = channel_data[0, :, :, :]
    tif_data[1, 5:, 5:, :] = channel_data[0, :-5, :-5, :]

    segmentation_masks = test_utils.make_labels_xarray(
        label_data=cell_masks,
        compartment_names=['whole_cell']
    )

    channel_data = test_utils.make_images_xarray(tif_data)

    normalized, _ = marker_quantification.generate_expression_matrix(segmentation_masks,
                                                                     channel_data)

    assert normalized.shape[0] == 7

    assert np.array_equal(normalized['chan0'], np.repeat(1, len(normalized)))
    assert np.array_equal(normalized['chan1'], np.repeat(5, len(normalized)))


def test_generate_expression_matrix_multiple_compartments():

    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask[0, :, :, 0]
    cell_masks[1, 5:, 5:, 0] = cell_mask[0, :-5, :-5, 0]

    channel_datas = np.zeros((2, 40, 40, 5), dtype="int16")
    channel_datas[0, :, :, :] = channel_data[0, :, :, :]
    channel_datas[1, 5:, 5:, :] = channel_data[0, :-5, :-5, :]

    # generate a second set of nuclear masks that are smaller than cell masks
    nuc_masks = np.zeros_like(cell_masks)
    nuc_masks[0, :, :, 0] = erosion(cell_masks[0, :, :, 0], selem=morph.disk(1))
    nuc_masks[1, :, :, 0] = erosion(cell_masks[1, :, :, 0], selem=morph.disk(1))

    # cell 2 in fov0 has no nucleus
    nuc_masks[0, nuc_masks[0, :, :, 0] == 2, 0] = 0

    # all of the nuclei have a label that is 2x the label of the corresponding cell
    nuc_masks *= 2

    unequal_masks = np.concatenate((cell_masks, nuc_masks), axis=-1)

    segmentation_masks_unequal = test_utils.make_labels_xarray(
        label_data=unequal_masks,
        compartment_names=['whole_cell', 'nuclear']
    )

    channel_data = test_utils.make_images_xarray(channel_datas)

    normalized, arcsinh = marker_quantification.generate_expression_matrix(
        segmentation_masks_unequal,
        channel_data,
        nuclear_counts=True
    )

    # 7 total cells
    assert normalized.shape[0] == 7

    # channel 0 has a constant value of 1
    assert np.array_equal(normalized['chan0'], np.repeat(1, len(normalized)))

    # channel 1 has a constant value of 5
    assert np.array_equal(normalized['chan1'], np.repeat(5, len(normalized)))

    # these two channels should be equal for all cells
    assert np.array_equal(normalized['chan1'], normalized['chan2'])

    # check that cell with missing nucleus has size 0
    index = np.logical_and(normalized['label'] == 2, normalized['fov'] == 'Point0')
    assert normalized.loc[index, 'cell_size_nuclear'].values == 0

    # check that correct nuclear label is assigned to all cells
    normalized_with_nuc = normalized.loc[normalized['label'] != 2, ['label', 'label_nuclear']]
    assert np.array_equal(normalized_with_nuc['label'] * 2, normalized_with_nuc['label_nuclear'])


def test_compute_complete_expression_matrices():

    # checks if the tree loading is being called correctly when is_mibitiff is False
    # save the actual expression matrix and data loding tests for their respective test functions
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 FOVs and 3 imgs per FOV
        fovs, chans = test_utils.gen_fov_chan_names(3, 3)

        tiff_dir = os.path.join(temp_dir, "single_channel_inputs")
        img_sub_folder = "TIFs"

        os.mkdir(tiff_dir)
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs,
            channel_names=chans,
            img_shape=(40, 40),
            sub_dir=img_sub_folder,
            dtype="int16"
        )

        # define a subset of fovs
        fovs_subset = fovs[:2]

        # generate a sample segmentation_mask
        cell_mask, _ = test_utils.create_test_extraction_data()

        cell_masks = np.zeros((3, 40, 40, 1), dtype="int16")
        cell_masks[0, :, :, 0] = cell_mask[0, :, :, 0]
        cell_masks[1, 5:, 5:, 0] = cell_mask[0, :-5, :-5, 0]
        cell_masks[2, 10:, 10:, 0] = cell_mask[0, :-10, :-10, 0]

        segmentation_masks = test_utils.make_labels_xarray(
            label_data=cell_masks,
            compartment_names=['whole_cell']
        )

        # checks that a ValueError is thrown when the user tries to specify points that are not
        # in the original segmentation mask
        with pytest.raises(ValueError):
            # generate a segmentation array with 1 FOV

            marker_quantification.compute_complete_expression_matrices(
                segmentation_labels=segmentation_masks.loc[["Point1"]], tiff_dir=tiff_dir,
                img_sub_folder=img_sub_folder, is_mibitiff=False, points=["Point1", "Point2"],
                batch_size=5)

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
        # define 3 FOVs and 2 mibitiff_imgs
        fovs, channels = test_utils.gen_fov_chan_names(3, 2)

        # define a subset of fovs
        fovs_subset = fovs[:2]

        tiff_dir = os.path.join(temp_dir, "mibitiff_inputs")

        os.mkdir(tiff_dir)
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs,
            channel_names=channels,
            img_shape=(40, 40),
            mode='mibitiff',
            dtype=np.float32
        )

        # generate a sample segmentation_mask
        cell_mask, _ = test_utils.create_test_extraction_data()
        cell_masks = np.zeros((3, 40, 40, 1), dtype="int16")
        cell_masks[0, :, :, 0] = cell_mask[0, :, :, 0]
        cell_masks[1, 5:, 5:, 0] = cell_mask[0, :-5, :-5, 0]
        cell_masks[2, 10:, 10:, 0] = cell_mask[0, :-10, :-10, 0]
        segmentation_masks = test_utils.make_labels_xarray(
            label_data=cell_masks,
            compartment_names=['whole_cell']
        )

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
