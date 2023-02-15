import copy
import os
import tempfile

import numpy as np
import pytest
import skimage.morphology as morph
import xarray as xr
from skimage.morphology import erosion
from alpineer import image_utils, misc_utils
from alpineer.test_utils import (create_paired_xarray_fovs, gen_fov_chan_names,
                                 make_images_xarray, make_labels_xarray)

import ark.settings as settings
from ark.segmentation import marker_quantification
import test_utils

parametrize = pytest.mark.parametrize


@parametrize('regionprops_single_comp',
             [copy.deepcopy(settings.REGIONPROPS_SINGLE_COMP), []])
def test_get_single_compartment_props(regionprops_single_comp):
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    segmentation_labels = make_labels_xarray(label_data=cell_mask,
                                             compartment_names=['whole_cell'])

    regionprops_base = copy.deepcopy(settings.REGIONPROPS_BASE)

    regionprops_names = copy.deepcopy(regionprops_base)
    regionprops_names.remove('centroid')
    regionprops_names += ['centroid-0', 'centroid-1']

    cell_props = marker_quantification.get_single_compartment_props(
        segmentation_labels.loc['fov0', :, :, 'whole_cell'].values,
        regionprops_base,
        regionprops_single_comp)

    misc_utils.verify_same_elements(
        all_features=regionprops_names + regionprops_single_comp,
        cell_props_columns=cell_props.columns.values
    )

    # test that blank segmentation mask is handled appropriately
    cell_props_blank = marker_quantification.get_single_compartment_props(
        np.zeros((40, 40), dtype='int'),
        regionprops_base,
        regionprops_single_comp)

    misc_utils.verify_same_elements(
        all_features=copy.deepcopy(regionprops_base) + regionprops_single_comp,
        cell_props_columns=cell_props_blank.columns.values
    )


def test_assign_single_compartment_props():
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    segmentation_labels = make_labels_xarray(
        label_data=cell_mask,
        compartment_names=['whole_cell']
    )

    input_images = make_images_xarray(channel_data)

    # define the names of the base features that can be computed directly from regionprops
    regionprops_base = copy.deepcopy(settings.REGIONPROPS_BASE) + ['coords']

    # define the names of the extras
    regionprops_single_comp = copy.deepcopy(settings.REGIONPROPS_SINGLE_COMP)

    # define the names of everything
    regionprops_names = copy.deepcopy(regionprops_base)
    regionprops_names.remove('coords')
    regionprops_names.remove('centroid')
    regionprops_names += ['centroid-0', 'centroid-1'] + regionprops_single_comp

    # get all the cell ids, for testing we'll only use 1 cell id
    unique_cell_ids = np.unique(segmentation_labels[..., 0].values)
    unique_cell_ids = unique_cell_ids[np.nonzero(unique_cell_ids)]
    cell_ids = unique_cell_ids[:2]

    # create the cell properties, easier to use get_single_compartment_props and make_labels_xarray
    cell_props = marker_quantification.get_single_compartment_props(
        segmentation_labels.loc['fov0', :, :, 'whole_cell'].values,
        regionprops_base, regionprops_single_comp
    )

    # set channel names for images
    channel_features = input_images.channels.values

    # create labels for array holding channel counts and morphology metrics
    feature_names = np.concatenate((np.array(settings.PRE_CHANNEL_COL), channel_features,
                                    regionprops_names), axis=None)

    # create np.array to hold compartment x cell x feature info
    marker_counts_array = np.zeros((len(segmentation_labels.compartments), 2,
                                    len(feature_names)))

    marker_counts = xr.DataArray(copy.copy(marker_counts_array),
                                 coords=[segmentation_labels.compartments,
                                         cell_ids,
                                         feature_names],
                                 dims=['compartments', 'cell_id', 'features'])

    # get the cell coordinates of the cell_id
    cell_coords = cell_props.loc[cell_props['label'] == cell_ids[0], 'coords'].values[0]

    # assign the features to that cell id
    marker_counts = marker_quantification.assign_single_compartment_features(
        marker_counts, 'whole_cell', cell_props, cell_coords, cell_ids[0], cell_ids[0],
        input_images.loc['fov0', ...], regionprops_names, 'total_intensity'
    )

    assert marker_counts.loc[:, cell_ids[0], settings.POST_CHANNEL_COL] == 1
    assert marker_counts.loc[:, cell_ids[0], 'cell_size'] == 36
    assert marker_counts.loc[:, cell_ids[0], 'centroid-0'] == 6.5
    assert marker_counts.loc[:, cell_ids[0], 'centroid-1'] == 6.5

    assert marker_counts.loc[:, cell_ids[0], 'area'] == 36

    major_axis_length = marker_counts.loc[:, cell_ids[0], 'major_axis_length']
    minor_axis_length = marker_counts.loc[:, cell_ids[0], 'minor_axis_length']
    assert major_axis_length == minor_axis_length

    assert marker_counts.loc[:, cell_ids[0], 'centroid_dif'] == 0
    assert marker_counts.loc[:, cell_ids[0], 'num_concavities'] == 0

    # get the cell coordinates of the cell_id
    cell_coords = cell_props.loc[cell_props['label'] == cell_ids[1], 'coords'].values[0]

    # assign the features to that cell id
    marker_counts = marker_quantification.assign_single_compartment_features(
        marker_counts, 'whole_cell', cell_props, cell_coords, cell_ids[1], cell_ids[1],
        input_images.loc['fov0', ...], regionprops_names, 'total_intensity',
    )

    assert marker_counts.loc[:, cell_ids[1], settings.POST_CHANNEL_COL] == 2
    assert marker_counts.loc[:, cell_ids[1], 'cell_size'] == 100
    assert marker_counts.loc[:, cell_ids[1], 'centroid-0'] == 19.5
    assert marker_counts.loc[:, cell_ids[1], 'centroid-1'] == 24.5

    assert marker_counts.loc[:, cell_ids[1], 'area'] == 100

    major_axis_length = marker_counts.loc[:, cell_ids[1], 'major_axis_length']
    minor_axis_length = marker_counts.loc[:, cell_ids[1], 'minor_axis_length']
    assert major_axis_length == minor_axis_length

    assert marker_counts.loc[:, cell_ids[1], 'centroid_dif'] == 0
    assert marker_counts.loc[:, cell_ids[1], 'num_concavities'] == 0


@parametrize('regionprops_multi_comp',
             [copy.deepcopy(settings.REGIONPROPS_MULTI_COMP), []])
def test_assign_multi_compartment_features(regionprops_multi_comp):
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # test whole_cell and nuclear compartments with same data
    segmentation_labels_equal = make_labels_xarray(
        label_data=np.concatenate((cell_mask, cell_mask), axis=-1),
        compartment_names=['whole_cell', 'nuclear']
    )

    # create a sample marker count matrix with 2 compartments, 3 cell ids, and 2 features
    sample_marker_counts = np.zeros((2, 3, 2))

    # cell 0: no nucleus
    sample_marker_counts[0, 0, 1] = 5

    # cell 1: equal whole cell and nuclear area
    sample_marker_counts[0, 1, 1] = 10
    sample_marker_counts[1, 1, 1] = 10

    # cell 2: different whole cell and nuclear area
    sample_marker_counts[0, 2, 1] = 10
    sample_marker_counts[1, 2, 1] = 5

    # write marker_counts to xarray
    sample_marker_counts = xr.DataArray(copy.copy(sample_marker_counts),
                                        coords=[['whole_cell', 'nuclear'],
                                                [0, 1, 2],
                                                ['feat_1', 'area']],
                                        dims=['compartments', 'cell_id', 'features'])

    sample_marker_counts = marker_quantification.assign_multi_compartment_features(
        sample_marker_counts, regionprops_multi_comp
    )

    # check nc_ratio stats if regionprops_multi_comp set
    if len(regionprops_multi_comp) > 0:
        # assert we added nc_ratio as a features key
        assert 'nc_ratio' in sample_marker_counts.features.values

        # testing cell 0
        assert sample_marker_counts.loc['whole_cell', 0, 'nc_ratio'] == 0
        assert sample_marker_counts.loc['nuclear', 0, 'nc_ratio'] == 0

        # testing cell 1
        assert sample_marker_counts.loc['whole_cell', 1, 'nc_ratio'] == 1
        assert sample_marker_counts.loc['nuclear', 1, 'nc_ratio'] == 1

        # testing cell 2
        assert sample_marker_counts.loc['whole_cell', 2, 'nc_ratio'] == 0.5
        assert sample_marker_counts.loc['nuclear', 2, 'nc_ratio'] == 0.5
    # otherwise ensure we didn't add nc_ratio
    else:
        assert 'nc_ratio' not in sample_marker_counts.features.values


@parametrize('fast_extraction', [False, True])
def test_compute_marker_counts_base(fast_extraction):
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    segmentation_labels = make_labels_xarray(label_data=cell_mask,
                                             compartment_names=['whole_cell'])

    input_images = make_images_xarray(channel_data)

    # test utils output is 4D but tests require 3D
    segmentation_labels, input_images = segmentation_labels[0], input_images[0]

    segmentation_output = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_labels=segmentation_labels,
                                                    fast_extraction=fast_extraction)

    # check that cell sizes are correct
    sizes = [np.sum(cell_mask == cell_id) for cell_id in [1, 2, 3, 5]]
    assert np.array_equal(sizes, segmentation_output.loc['whole_cell', :, settings.CELL_SIZE])

    # check that channel 0 counts are same as cell size
    assert np.array_equal(
        segmentation_output.loc['whole_cell', :, settings.CELL_SIZE].values,
        segmentation_output.loc['whole_cell', :, 'chan0'].values
    )

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(
        segmentation_output.loc['whole_cell', :, settings.CELL_SIZE].values * 5,
        segmentation_output.loc['whole_cell', :, 'chan1'].values
    )

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(
        segmentation_output.loc['whole_cell', :, 'chan2'].values,
        segmentation_output.loc['whole_cell', :, 'chan1'].values
    )

    # check that only cell1 is negative for channel 3
    assert segmentation_output.loc['whole_cell', 1, 'chan3'] == 0
    assert np.all(segmentation_output.loc['whole_cell', 2:, 'chan3'] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output.loc['whole_cell', 2, 'chan4'] > 0
    assert np.all(segmentation_output.loc['whole_cell', :1, 'chan4'] == 0)
    assert np.all(segmentation_output.loc['whole_cell', 3:, 'chan4'] == 0)

    # check that regionprops size matches with cell size, only check if fast_extraciton=False
    if not fast_extraction:
        assert np.array_equal(
            segmentation_output.loc['whole_cell', :, settings.CELL_SIZE],
            segmentation_output.loc['whole_cell', :, 'area']
        )

    # bad extraction selection
    with pytest.raises(ValueError):
        marker_quantification.compute_marker_counts(
            input_images=input_images,
            segmentation_labels=segmentation_labels,
            extraction='bad_extraction'
        )

    # test different extraction selection
    center_extraction = \
        marker_quantification.compute_marker_counts(
            input_images=input_images,
            segmentation_labels=segmentation_labels,
            extraction='center_weighting'
        )

    assert np.all(
        segmentation_output.loc['whole_cell', :, 'chan0'].values
        > center_extraction.loc['whole_cell', :, 'chan0'].values
    )

    # blank segmentation mask results in the cells column of length 0
    blank_labels = make_labels_xarray(
        label_data=np.zeros((1, 40, 40, 1), dtype='int'),
        compartment_names=['whole_cell']
    )

    blank_output = marker_quantification.compute_marker_counts(
        input_images=input_images,
        segmentation_labels=blank_labels[0]
    )
    assert blank_output.shape[1] == 0


def test_compute_marker_counts_equal_masks():
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # test whole_cell and nuclear compartments with same data
    segmentation_labels_equal = make_labels_xarray(
        label_data=np.concatenate((cell_mask, cell_mask), axis=-1),
        compartment_names=['whole_cell', 'nuclear']
    )

    input_images = make_images_xarray(channel_data)

    # test utils output is 4D but tests require 3D
    segmentation_labels_equal, input_images = segmentation_labels_equal[0], input_images[0]

    segmentation_output_equal = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_labels=segmentation_labels_equal,
                                                    nuclear_counts=True)

    assert np.all(segmentation_output_equal[0].values == segmentation_output_equal[1].values)


@parametrize('fast_extraction', [False, True])
def test_compute_marker_counts_nuc_whole_cell_diff(fast_extraction):
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # nuclear mask is smaller
    nuc_mask = \
        np.expand_dims(erosion(cell_mask[0, :, :, 0], footprint=morph.disk(1)), axis=0)
    nuc_mask = np.expand_dims(nuc_mask, axis=-1)

    unequal_masks = np.concatenate((cell_mask, nuc_mask), axis=-1)
    segmentation_labels_unequal = make_labels_xarray(
        label_data=unequal_masks,
        compartment_names=['whole_cell', 'nuclear']
    )

    input_images = make_images_xarray(channel_data)

    # test utils output is 4D but tests require 3D
    segmentation_labels_unequal, input_images = segmentation_labels_unequal[0], input_images[0]

    segmentation_output_unequal = \
        marker_quantification.compute_marker_counts(
            input_images=input_images,
            segmentation_labels=segmentation_labels_unequal,
            nuclear_counts=True,
            fast_extraction=fast_extraction
        )

    # make sure nuclear segmentations are smaller, this applies even if fast_extraction set
    assert np.all(
        segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values <
        segmentation_output_unequal.loc['whole_cell', :, 'cell_size'].values
    )

    # check that cell sizes are correct, this applies even if fast_extraction set
    sizes = [np.sum(nuc_mask == cell_id) for cell_id in [1, 2, 3, 5]]
    assert np.array_equal(sizes, segmentation_output_unequal.loc['nuclear', :, 'cell_size'])

    # check that channel 0 counts are same as cell size
    assert np.array_equal(
        segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values,
        segmentation_output_unequal.loc['nuclear', :, 'chan0'].values
    )

    # check that channel 1 counts are 5x cell size
    assert np.array_equal(
        segmentation_output_unequal.loc['nuclear', :, 'cell_size'].values * 5,
        segmentation_output_unequal.loc['nuclear', :, 'chan1'].values
    )

    # check that channel 2 counts are the same as channel 1
    assert np.array_equal(
        segmentation_output_unequal.loc['nuclear', :, 'chan2'].values,
        segmentation_output_unequal.loc['nuclear', :, 'chan1'].values
    )

    # check that only cell1 is negative for channel 3
    assert segmentation_output_unequal.loc['nuclear', 1, 'chan3'] == 0
    assert np.all(segmentation_output_unequal.loc['nuclear', 2:, 'chan3'] > 0)

    # check that only cell2 is positive for channel 4
    assert segmentation_output_unequal.loc['nuclear', 2, 'chan4'] > 0
    assert np.all(segmentation_output_unequal.loc['nuclear', :1, 'chan4'] == 0)
    assert np.all(segmentation_output_unequal.loc['nuclear', 3:, 'chan4'] == 0)

    # the following test applies only if fast_extraction not set
    if not fast_extraction:
        assert np.array_equal(
            segmentation_output_unequal.loc['nuclear', :, 'cell_size'],
            segmentation_output_unequal.loc['nuclear', :, 'area']
        )

    # check that splitting large nuclei works as expected

    # swap nuclear and cell masks so that nuc is bigger
    big_nuc_masks = np.concatenate((nuc_mask, cell_mask), axis=-1)
    segmentation_labels_big_nuc = make_labels_xarray(
        label_data=big_nuc_masks,
        compartment_names=['whole_cell', 'nuclear']
    )

    # test utils output is 4D but tests require 3D
    segmentation_labels_big_nuc = segmentation_labels_big_nuc[0]

    segmentation_output_big_nuc = \
        marker_quantification.compute_marker_counts(
            input_images=input_images,
            segmentation_labels=segmentation_labels_big_nuc,
            nuclear_counts=True,
            split_large_nuclei=True)


def test_compute_marker_counts_no_coords():
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # test whole_cell and nuclear compartments with same data
    segmentation_labels_equal = make_labels_xarray(
        label_data=np.concatenate((cell_mask, cell_mask), axis=-1),
        compartment_names=['whole_cell', 'nuclear']
    )

    input_images = make_images_xarray(channel_data)

    segmentation_labels_equal, input_images = segmentation_labels_equal[0], input_images[0]

    # different object properties can be supplied
    regionprops_base = ['label', 'area']
    excluded_defaults = ['eccentricity']

    segmentation_output_specified = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_labels=segmentation_labels_equal,
                                                    nuclear_counts=True,
                                                    regionprops_base=regionprops_base)

    assert np.all(np.isin(['label', 'area'], segmentation_output_specified.features.values))

    assert not np.any(np.isin(excluded_defaults, segmentation_output_specified.features.values))

    # these nuclei are all smaller than the cells, so we should get same result
    segmentation_output_specified_split = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_labels=segmentation_labels_equal,
                                                    nuclear_counts=True,
                                                    regionprops_base=regionprops_base,
                                                    split_large_nuclei=True)

    assert np.all(segmentation_output_specified_split == segmentation_output_specified)


def test_compute_marker_counts_no_labels():
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # test whole_cell and nuclear compartments with same data
    segmentation_labels_equal = make_labels_xarray(
        label_data=np.concatenate((cell_mask, cell_mask), axis=-1),
        compartment_names=['whole_cell', 'nuclear']
    )

    input_images = make_images_xarray(channel_data)

    segmentation_labels_equal, input_images = segmentation_labels_equal[0], input_images[0]

    # different object properties can be supplied
    regionprops_base = ['coords', 'area']
    excluded_defaults = ['eccentricity']

    segmentation_output_specified = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_labels=segmentation_labels_equal,
                                                    nuclear_counts=True,
                                                    regionprops_base=regionprops_base)

    assert np.all(np.isin(['label', 'area'], segmentation_output_specified.features.values))

    assert not np.any(np.isin(excluded_defaults, segmentation_output_specified.features.values))

    # these nuclei are all smaller than the cells, so we should get same result
    segmentation_output_specified_split = \
        marker_quantification.compute_marker_counts(input_images=input_images,
                                                    segmentation_labels=segmentation_labels_equal,
                                                    nuclear_counts=True,
                                                    regionprops_base=regionprops_base,
                                                    split_large_nuclei=True)

    assert np.all(segmentation_output_specified_split == segmentation_output_specified)


@parametrize('fast_extraction', [False, True])
def test_create_marker_count_matrices_base(fast_extraction):
    cell_mask, channel_data = test_utils.create_test_extraction_data()

    # generate data for two fovs offset
    cell_masks = np.zeros((2, 40, 40, 1), dtype="int16")
    cell_masks[0, :, :, 0] = cell_mask[0, :, :, 0]
    cell_masks[1, 5:, 5:, 0] = cell_mask[0, :-5, :-5, 0]

    tif_data = np.zeros((2, 40, 40, 5), dtype="int16")
    tif_data[0, :, :, :] = channel_data[0, :, :, :]
    tif_data[1, 5:, 5:, :] = channel_data[0, :-5, :-5, :]

    segmentation_labels = make_labels_xarray(
        label_data=cell_masks,
        compartment_names=['whole_cell']
    )

    channel_data = make_images_xarray(tif_data)

    # NOTE: use 0:1 instead of 0 to ensure dimension doesn't collapse
    normalized, _ = marker_quantification.create_marker_count_matrices(
        segmentation_labels[0:1, ...],
        channel_data[0:1, ...],
        fast_extraction=fast_extraction
    )

    # 4 total cells
    assert normalized.shape[0] == 4

    # exclude custom regionprops columns from count if fast_extraction set
    assert normalized.shape[1] == 10 if fast_extraction else 23

    assert np.array_equal(normalized['chan0'], np.repeat(1, len(normalized)))
    assert np.array_equal(normalized['chan1'], np.repeat(5, len(normalized)))

    # blank image doesn't cause any issues
    segmentation_labels.values[1, ...] = 0
    _ = marker_quantification.create_marker_count_matrices(
        segmentation_labels[1:2, ...],
        channel_data[1:2, ...],
        fast_extraction=fast_extraction
    )

    # error checking
    with pytest.raises(ValueError):
        # attempt to pass non-xarray for segmentation_labels
        marker_quantification.create_marker_count_matrices(segmentation_labels.values,
                                                           channel_data)

    with pytest.raises(ValueError):
        marker_quantification.create_marker_count_matrices(segmentation_labels,
                                                           channel_data.values)

    segmentation_labels_bad = segmentation_labels.copy()
    segmentation_labels_bad = segmentation_labels_bad.reindex({'fovs': [1, 2]})

    with pytest.raises(ValueError):
        # attempt to pass segmentation_labels and channel_data with different fovs
        marker_quantification.create_marker_count_matrices(segmentation_labels_bad,
                                                           channel_data)


@parametrize('fast_extraction', [False, True])
def test_create_marker_count_matrices_multiple_compartments(fast_extraction):
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
    nuc_masks[0, :, :, 0] = erosion(cell_masks[0, :, :, 0], footprint=morph.disk(1))
    nuc_masks[1, :, :, 0] = erosion(cell_masks[1, :, :, 0], footprint=morph.disk(1))

    # cell 2 in fov0 has no nucleus
    nuc_masks[0, nuc_masks[0, :, :, 0] == 2, 0] = 0

    # all of the nuclei have a label that is 2x the label of the corresponding cell
    nuc_masks *= 2

    unequal_masks = np.concatenate((cell_masks, nuc_masks), axis=-1)

    segmentation_labels_unequal = make_labels_xarray(
        label_data=unequal_masks,
        compartment_names=['whole_cell', 'nuclear']
    )

    channel_data = make_images_xarray(channel_datas)

    # NOTE: use 0:1 instead of 0 to prevent dimension from collapsing
    normalized, arcsinh = marker_quantification.create_marker_count_matrices(
        segmentation_labels_unequal[0:1, ...],
        channel_data[0:1, ...],
        nuclear_counts=True,
        fast_extraction=fast_extraction
    )

    # 4 total cells
    assert normalized.shape[0] == 4

    # exclude custom regionprops columns from count if fast_extraction set
    assert normalized.shape[1] == 19 if fast_extraction else 47

    # check that cell with missing nucleus has size 0, runs even with fast_extraction
    index = np.logical_and(normalized['label'] == 2, normalized['fov'] == 'fov0')
    assert normalized.loc[index, 'cell_size_nuclear'].values == 0

    # check that correct nuclear label is assigned to all cells, runs even with fast_extraction
    normalized_with_nuc = normalized.loc[normalized['label'] != 2, ['label', 'label_nuclear']]
    assert np.array_equal(normalized_with_nuc['label'] * 2, normalized_with_nuc['label_nuclear'])

    # channel 0 has a constant value of 1
    assert np.array_equal(normalized['chan0'], np.repeat(1, len(normalized)))

    # channel 1 has a constant value of 5
    assert np.array_equal(normalized['chan1'], np.repeat(5, len(normalized)))

    # these two channels should be equal for all cells
    assert np.array_equal(normalized['chan1'], normalized['chan2'])

    # blank nuclear segmentation mask doesn't cause any issues
    segmentation_labels_unequal.values[1, ..., 1] = 0
    _ = marker_quantification.create_marker_count_matrices(
        segmentation_labels_unequal[0:1, ...],
        channel_data[0:1, ...],
        nuclear_counts=True
    )


# NOTE: fast_extraction logic handled by create_marker_count_matrices tests
def test_generate_cell_table_tree_loading():
    # is_mibitiff False case, load from directory tree
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 fovs and 3 imgs per fov
        fovs, chans = gen_fov_chan_names(3, 3)

        tiff_dir = os.path.join(temp_dir, "single_channel_inputs")
        img_sub_folder = "TIFs"

        os.mkdir(tiff_dir)

        # this function should work on FOVs with varying sizes
        fov_size_split = 2
        create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs[0:fov_size_split],
            channel_names=chans,
            img_shape=(40, 40),
            sub_dir=img_sub_folder,
            dtype="int16"
        )
        create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs[fov_size_split:],
            channel_names=chans,
            img_shape=(20, 20),
            sub_dir=img_sub_folder,
            dtype="int16"
        )

        # define a subset of fovs
        fovs_subset = fovs[:2]

        # define a subset of fovs with file extensions
        fovs_subset_ext = fovs[:2]
        fovs_subset_ext[0] = str(fovs_subset_ext[0]) + ".tiff"
        fovs_subset_ext[1] = str(fovs_subset_ext[1]) + ".tiff"

        # generate sample segmentation_masks
        cell_masks_40 = np.random.randint(
            low=0, high=5, size=(fov_size_split, 40, 40, 2), dtype="int16"
        )

        # TODO: condense cell mask generation and saving into a helper function
        for i in np.arange(1, cell_masks_40.shape[0]):
            cell_masks_40[i, (5 * i):, (5 * i):, 0] = cell_masks_40[i, :-(5 * i), :-(5 * i), 0]
        cell_masks_40[..., 1] = cell_masks_40[..., 0]

        cell_masks_20 = np.random.randint(
            low=0, high=5, size=(len(fovs) - fov_size_split, 20, 20, 2), dtype="int16"
        )
        for i in np.arange(1, cell_masks_20.shape[0]):
            cell_masks_20[i, (5 * i):, (5 * i):, 0] = cell_masks_20[i, :-(5 * i), :-(5 * i), 0]
        cell_masks_20[..., 1] = cell_masks_20[..., 0]

        for fov in range(cell_masks_40.shape[0]):
            fov_whole_cell = cell_masks_40[fov, :, :, 0]
            fov_nuclear = cell_masks_40[fov, :, :, 1]
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_whole_cell.tiff' % fov),
                                   fov_whole_cell)
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_nuclear.tiff' % fov),
                                   fov_nuclear)

        for fov in range(cell_masks_20.shape[0]):
            fov_whole_cell = cell_masks_20[fov, :, :, 0]
            fov_nuclear = cell_masks_20[fov, :, :, 1]
            image_utils.save_image(
                os.path.join(temp_dir, 'fov%d_whole_cell.tiff' % (fov + fov_size_split)),
                fov_whole_cell
            )
            image_utils.save_image(
                os.path.join(temp_dir, 'fov%d_nuclear.tiff' % (fov + fov_size_split)),
                fov_nuclear
            )

        with pytest.raises(FileNotFoundError):
            # specifying fovs not in the original segmentation mask
            marker_quantification.generate_cell_table(
                segmentation_dir=temp_dir, tiff_dir=tiff_dir,
                img_sub_folder=img_sub_folder, is_mibitiff=False, fovs=["fov2", "fov3"])

        # generate sample norm and arcsinh data for all fovs
        norm_data_all_fov, arcsinh_data_all_fov = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, fovs=None)

        assert norm_data_all_fov.shape[0] > 0 and norm_data_all_fov.shape[1] > 0
        assert arcsinh_data_all_fov.shape[0] > 0 and arcsinh_data_all_fov.shape[1] > 0

        # generate sample norm and arcsinh data for a subset of fovs
        norm_data_fov_sub, arcsinh_data_fov_sub = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, fovs=fovs_subset)

        assert norm_data_fov_sub.shape[0] > 0 and norm_data_fov_sub.shape[1] > 0
        assert arcsinh_data_fov_sub.shape[0] > 0 and arcsinh_data_fov_sub.shape[1] > 0

        # generate sample norm and arcsinh data for a subset of fovs with extensions
        norm_data_fov_ext, arcsinh_data_fov_ext = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, fovs=fovs_subset_ext)

        assert norm_data_fov_ext.shape[0] > 0 and norm_data_fov_ext.shape[1] > 0
        assert arcsinh_data_fov_ext.shape[0] > 0 and arcsinh_data_fov_ext.shape[1] > 0

        # test nuclear_counts True
        norm_data_nuc, arcsinh_data_nuc = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False, fovs=fovs_subset,
            nuclear_counts=True)

        assert norm_data_nuc.shape[0] == norm_data_fov_sub.shape[0]
        assert norm_data_nuc.shape[1] == norm_data_fov_sub.shape[1] * 2 + 1
        misc_utils.verify_in_list(
            nuclear_col='nc_ratio',
            nuc_cell_table_cols=norm_data_nuc.columns.values
        )

        assert arcsinh_data_nuc.shape[0] == arcsinh_data_fov_sub.shape[0]
        assert arcsinh_data_nuc.shape[1] == norm_data_fov_sub.shape[1] * 2 + 1
        misc_utils.verify_in_list(
            nuclear_col='nc_ratio',
            nuc_cell_table_cols=norm_data_nuc.columns.values
        )


# TODO: consider removing since MIBItiffs are being phased out
def test_generate_cell_table_mibitiff_loading():
    # is_mibitiff True case, load from mibitiff file structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 fovs and 2 mibitiff_imgs
        fovs, channels = gen_fov_chan_names(3, 2)

        # define a subset of fovs
        fovs_subset = fovs[:2]

        # define a subset of fovs with file extensions
        fovs_subset_ext = fovs[:2]
        fovs_subset_ext[0] = str(fovs_subset_ext[0]) + ".tiff"
        fovs_subset_ext[1] = str(fovs_subset_ext[1]) + ".tiff"

        tiff_dir = os.path.join(temp_dir, "mibitiff_inputs")

        os.mkdir(tiff_dir)
        create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs,
            channel_names=channels,
            img_shape=(40, 40),
            mode='mibitiff',
            dtype=np.float32
        )

        # generate a sample segmentation_mask
        cell_masks = np.random.randint(low=0, high=5, size=(3, 40, 40, 2), dtype="int16")
        cell_masks[0, :, :, 0] = cell_masks[0, :, :, 0]
        cell_masks[1, 5:, 5:, 0] = cell_masks[0, :-5, :-5, 0]
        cell_masks[2, 10:, 10:, 0] = cell_masks[0, :-10, :-10, 0]
        cell_masks[..., 1] = cell_masks[..., 0]

        for fov in range(cell_masks.shape[0]):
            fov_whole_cell = cell_masks[fov, :, :, 0]
            fov_nuclear = cell_masks[fov, :, :, 1]
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_whole_cell.tiff' % fov),
                                   fov_whole_cell)
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_nuclear.tiff' % fov),
                                   fov_nuclear)

        # generate sample norm and arcsinh data for all fovs
        norm_data_all_fov, arcsinh_data_all_fov = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=tiff_dir, is_mibitiff=True, fovs=None)

        assert norm_data_all_fov.shape[0] > 0 and norm_data_all_fov.shape[1] > 0
        assert arcsinh_data_all_fov.shape[0] > 0 and arcsinh_data_all_fov.shape[1] > 0

        # generate sample norm and arcsinh data for a subset of fovs
        norm_data_fov_sub, arcsinh_data_fov_sub = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=tiff_dir, is_mibitiff=True, fovs=fovs_subset)

        assert norm_data_fov_sub.shape[0] > 0 and norm_data_fov_sub.shape[1] > 0
        assert arcsinh_data_fov_sub.shape[0] > 0 and arcsinh_data_fov_sub.shape[1] > 0

        # generate sample norm and arcsinh data for a subset of fovs with extensions
        norm_data_fov_ext, arcsinh_data_fov_ext = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=tiff_dir, is_mibitiff=True, fovs=fovs_subset_ext)

        assert norm_data_fov_ext.shape[0] > 0 and norm_data_fov_ext.shape[1] > 0
        assert arcsinh_data_fov_ext.shape[0] > 0 and arcsinh_data_fov_ext.shape[1] > 0

        # test nuclear_counts True
        norm_data_nuc, arcsinh_data_nuc = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=tiff_dir, is_mibitiff=True, fovs=fovs_subset,
            nuclear_counts=True)

        assert norm_data_nuc.shape[0] == norm_data_fov_sub.shape[0]
        assert norm_data_nuc.shape[1] == norm_data_fov_sub.shape[1] * 2 + 1
        misc_utils.verify_in_list(
            nuclear_col='nc_ratio',
            nuc_cell_table_cols=norm_data_nuc.columns.values
        )

        assert arcsinh_data_nuc.shape[0] == arcsinh_data_fov_sub.shape[0]
        assert arcsinh_data_nuc.shape[1] == norm_data_fov_sub.shape[1] * 2 + 1
        misc_utils.verify_in_list(
            nuclear_col='nc_ratio',
            nuc_cell_table_cols=norm_data_nuc.columns.values
        )


def test_generate_cell_table_extractions():
    with tempfile.TemporaryDirectory() as temp_dir:
        # define 3 fovs and 3 imgs per fov
        fovs, chans = gen_fov_chan_names(3, 3)

        tiff_dir = os.path.join(temp_dir, "single_channel_inputs")
        img_sub_folder = "TIFs"

        os.mkdir(tiff_dir)
        create_paired_xarray_fovs(
            base_dir=tiff_dir,
            fov_names=fovs,
            channel_names=chans,
            img_shape=(40, 40),
            sub_dir=img_sub_folder,
            fills=True,
            dtype="int16"
        )

        # generate a sample segmentation_mask
        cell_masks = np.random.randint(low=0, high=5, size=(3, 40, 40, 2), dtype="int16")
        cell_masks[0, :, :, 0] = cell_masks[0, :, :, 0]
        cell_masks[1, 5:, 5:, 0] = cell_masks[0, :-5, :-5, 0]
        cell_masks[2, 10:, 10:, 0] = cell_masks[0, :-10, :-10, 0]
        cell_masks[..., 1] = cell_masks[..., 0]

        for fov in range(cell_masks.shape[0]):
            fov_whole_cell = cell_masks[fov, :, :, 0]
            fov_nuclear = cell_masks[fov, :, :, 1]
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_whole_cell.tiff' % fov),
                                   fov_whole_cell)
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_nuclear.tiff' % fov),
                                   fov_nuclear)

        default_norm_data, _ = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False
        )

        # verify total intensity extraction
        assert np.all(
            default_norm_data.loc[default_norm_data[settings.CELL_LABEL] == 1][chans].values
            == np.arange(9).reshape(3, 3)
        )

        # define a specific threshold for positive pixel extraction
        thresh_kwargs = {
            'signal_kwargs':
                {
                    'threshold': 1
                }
        }

        # verify thresh kwarg passes through
        positive_pixel_data, _ = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False,
            extraction='positive_pixel', **thresh_kwargs
        )

        assert np.all(positive_pixel_data.iloc[:4][['chan0', 'chan1']].values == 0)
        assert np.all(positive_pixel_data.iloc[4:][chans].values == 1)

        # verify thresh kwarg passes through and nuclear counts True
        positive_pixel_data_nuc, _ = marker_quantification.generate_cell_table(
            segmentation_dir=temp_dir, tiff_dir=tiff_dir,
            img_sub_folder=img_sub_folder, is_mibitiff=False,
            extraction='positive_pixel', nuclear_counts=True, **thresh_kwargs
        )

        assert np.all(positive_pixel_data_nuc.iloc[:4][['chan0', 'chan1']].values == 0)
        assert np.all(positive_pixel_data_nuc.iloc[4:][chans].values == 1)
        assert positive_pixel_data_nuc.shape[0] == positive_pixel_data.shape[0]
        assert positive_pixel_data_nuc.shape[1] == positive_pixel_data.shape[1] * 2 + 1
        misc_utils.verify_in_list(
            nuclear_col='nc_ratio',
            nuc_cell_table_cols=positive_pixel_data_nuc.columns.values
        )
