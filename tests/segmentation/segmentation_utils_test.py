import os
import os.path
import tempfile

import numpy as np
import pytest
import test_utils
import xarray as xr
from alpineer import image_utils
from alpineer.test_utils import make_images_xarray, make_labels_xarray
from skimage.measure import regionprops

import ark.settings as settings
from ark.segmentation import segmentation_utils


def test_find_nuclear_mask_id():
    # create cell labels with 5 distinct cells
    cell_labels = np.zeros((60, 10), dtype='int')
    for i in range(6):
        cell_labels[(i * 10):(i * 10 + 8), :8] = i + 1

    # create nuc labels with varying degrees of overlap
    nuc_labels = np.zeros((60, 60), dtype='int')

    # perfect overlap
    nuc_labels[:8, :8] = 1

    # greater than majority overlap
    nuc_labels[10:16, :6] = 2

    # only partial overlap
    nuc_labels[20:23, :3] = 3

    # no overlap for cell 4

    # two cells overlapping, larger cell_id correct
    nuc_labels[40:48, :2] = 5
    nuc_labels[40:48, 2:8] = 20

    # two cells overlapping, background is highest
    nuc_labels[50:58, :1] = 21
    nuc_labels[50:58, 1:3] = 6

    true_nuc_ids = [1, 2, 3, None, 20, 6]

    cell_props = regionprops(cell_labels)

    # check that predicted nuclear id is correct for all cells in image
    for idx, prop in enumerate(cell_props):
        predicted_nuc = \
            segmentation_utils.find_nuclear_label_id(nuc_segmentation_labels=nuc_labels,
                                                     cell_coords=prop.coords)

        assert predicted_nuc == true_nuc_ids[idx]


def test_split_large_nuclei():
    cell_mask, _ = test_utils.create_test_extraction_data()
    cell_mask = cell_mask[0, :, :, 0]

    nuc_mask = np.zeros_like(cell_mask)

    # completely contained within the cell
    nuc_mask[4:8, 4:8] = 1

    # same size as cell
    nuc_mask[15:25, 20:30] = 2

    # strictly bigger than the cell
    nuc_mask[25:32, 3:30] = 3

    # only partially overlaps the cell
    nuc_mask[33:37, 12:20] = 5

    split_mask = segmentation_utils.split_large_nuclei(nuc_segmentation_labels=nuc_mask,
                                                       cell_segmentation_labels=cell_mask,
                                                       cell_ids=np.array([1, 2, 3, 5]))

    # nuc 1 and 2 are unchanged
    assert np.array_equal(nuc_mask == 1, split_mask == 1)
    assert np.array_equal(nuc_mask == 2, split_mask == 2)

    # nuc 3 was greater than cell 3
    nuc_3_inner = np.logical_and(nuc_mask == 3, cell_mask == 3)
    nuc_3_outer = np.logical_and(nuc_mask == 3, cell_mask != 3)

    nuc_3_inner_val = np.unique(split_mask[nuc_3_inner])
    nuc_3_outer_val = np.unique(split_mask[nuc_3_outer])

    # the different parts of nuc 3 have a single label
    assert len(nuc_3_inner_val) == 1
    assert len(nuc_3_outer_val) == 1

    # the labels are different
    assert nuc_3_inner_val != nuc_3_outer_val

    # nuc 5 partially overlapped cell 5
    nuc_5_inner = np.logical_and(nuc_mask == 5, cell_mask == 5)
    nuc_5_outer = np.logical_and(nuc_mask == 5, cell_mask != 5)

    nuc_5_inner_val = np.unique(split_mask[nuc_5_inner])
    nuc_5_outer_val = np.unique(split_mask[nuc_5_outer])

    # the different parts of nuc 3 have a single label
    assert len(nuc_5_inner_val) == 1
    assert len(nuc_5_outer_val) == 1

    # the labels are different
    assert nuc_5_inner_val != nuc_5_outer_val


# TODO: refactor to avoid code reuse
def test_transform_expression_matrix():
    # create expression matrix
    cell_data = np.random.choice([0, 1, 2, 3, 4], 70, replace=True)
    cell_data = cell_data.reshape((1, 10, 7)).astype('float')

    coords = [['whole_cell'], list(range(10)),
              [settings.CELL_SIZE, 'chan1', 'chan2', 'chan3', 'label', 'morph_1', 'morph_2']]
    dims = ['compartments', 'cell_id', 'features']

    cell_data = xr.DataArray(cell_data, coords=coords, dims=dims)

    unchanged_cols = [settings.CELL_SIZE, 'label', 'morph_1', 'morph_2']
    modified_cols = ['chan1', 'chan2', 'chan3']

    # test size_norm
    normalized_data = segmentation_utils.transform_expression_matrix(cell_data,
                                                                     transform='size_norm')

    assert np.array_equal(normalized_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        if cell_data.loc['whole_cell', cell, settings.CELL_SIZE] != 0:
            normalized_vals = np.divide(
                cell_data.loc['whole_cell', cell, modified_cols].values,
                cell_data.loc['whole_cell', cell, settings.CELL_SIZE].values
            )
            assert np.array_equal(normalized_data.loc['whole_cell', cell, modified_cols].values,
                                  normalized_vals)

    # test arcsinh transform
    transform_kwargs = {'linear_factor': 1}
    arcsinh_data = segmentation_utils.transform_expression_matrix(
        cell_data,
        transform='arcsinh',
        transform_kwargs=transform_kwargs
    )

    assert np.array_equal(arcsinh_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        arcsinh_vals = np.arcsinh(cell_data.loc[:, cell, modified_cols].values)
        assert np.array_equal(arcsinh_data.loc[:, cell, modified_cols].values, arcsinh_vals)


def test_transform_expression_matrix_multiple_compartments():
    # create expression matrix
    cell_data = np.random.choice([0, 1, 2, 3, 4], 140, replace=True)
    cell_data = cell_data.reshape((2, 10, 7)).astype('float')

    coords = [['whole_cell', 'nuclear'], list(range(10)),
              [settings.CELL_SIZE, 'chan1', 'chan2', 'chan3', 'label', 'morph_1', 'morph_2']]
    dims = ['compartments', 'cell_id', 'features']

    cell_data = xr.DataArray(cell_data, coords=coords, dims=dims)

    unchanged_cols = [settings.CELL_SIZE, 'label', 'morph_1', 'morph_2']
    modified_cols = ['chan1', 'chan2', 'chan3']

    # test size_norm
    normalized_data = segmentation_utils.transform_expression_matrix(cell_data,
                                                                     transform='size_norm')

    assert np.array_equal(normalized_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        if cell_data.loc['whole_cell', cell, settings.CELL_SIZE] != 0:
            normalized_vals = np.divide(
                cell_data.loc['whole_cell', cell, modified_cols].values,
                cell_data.loc['whole_cell', cell, settings.CELL_SIZE].values
            )
            assert np.array_equal(normalized_data.loc['whole_cell', cell, modified_cols].values,
                                  normalized_vals)
        if cell_data.loc['nuclear', cell, settings.CELL_SIZE] != 0:
            normalized_vals = np.divide(
                cell_data.loc['nuclear', cell, modified_cols].values,
                cell_data.loc['nuclear', cell, settings.CELL_SIZE].values
            )
            assert np.array_equal(normalized_data.loc['nuclear', cell, modified_cols].values,
                                  normalized_vals)

    # test arcsinh transform
    transform_kwargs = {'linear_factor': 1}
    arcsinh_data = segmentation_utils.transform_expression_matrix(
        cell_data,
        transform='arcsinh',
        transform_kwargs=transform_kwargs
    )

    assert np.array_equal(arcsinh_data.loc[:, :, unchanged_cols].values,
                          cell_data.loc[:, :, unchanged_cols].values)

    # TODO: In general it's bad practice for tests to call the same function as code under test
    for cell in cell_data.cell_id:
        arcsinh_vals = np.arcsinh(cell_data.loc[:, cell, modified_cols].values)
        assert np.array_equal(arcsinh_data.loc[:, cell, modified_cols].values, arcsinh_vals)


def test_save_segmentation_labels():
    channel_xr = make_images_xarray(
        np.zeros((2, 50, 50, 2)), channel_names=['nuclear_channel', 'membrane_channel']
    )
    segmentation_labels_xr = make_labels_xarray(np.zeros((2, 50, 50, 2)))
    fov_sub = segmentation_labels_xr.fovs.values[:1]
    chan_sub = channel_xr.channels.values[:2]

    # test saving without channel overlay
    with tempfile.TemporaryDirectory() as temp_dir:
        # save the images
        img_dir = os.path.join(temp_dir, 'data_dir')
        os.mkdir(img_dir)

        for fov in channel_xr.fovs.values:
            fov_channel_vals = channel_xr.loc[fov, ...].values
            image_utils.save_image(os.path.join(img_dir, '%s.tiff' % fov), fov_channel_vals)

        # save the segmentation labels
        for fov in segmentation_labels_xr.fovs.values:
            fov_whole_cell = segmentation_labels_xr.loc[fov, :, :, 'whole_cell'].values
            fov_nuclear = segmentation_labels_xr.loc[fov, :, :, 'nuclear'].values
            image_utils.save_image(os.path.join(temp_dir, '%s_whole_cell.tiff' % fov),
                                   fov_whole_cell)
            image_utils.save_image(os.path.join(temp_dir, '%s_nuclear.tiff' % fov),
                                   fov_nuclear)

        # test segmentation without channel overlay
        segmentation_utils.save_segmentation_labels(
            segmentation_dir=temp_dir, data_dir=img_dir, output_dir=temp_dir,
            fovs=fov_sub
        )

        # make sure all files except overlay.tiff exists
        assert os.path.exists(os.path.join(temp_dir,
                                           f'{fov_sub[0]}'
                                           f'_segmentation_borders.tiff'))
        assert not any(f.endswith('_overlay.tiff') for f in os.listdir(temp_dir))

    # test saving with channel overlay
    with tempfile.TemporaryDirectory() as temp_dir:
        # save the images
        img_dir = os.path.join(temp_dir, 'data_dir')
        os.mkdir(img_dir)

        for fov in channel_xr.fovs.values:
            fov_channel_vals = channel_xr.loc[fov, ...].values
            image_utils.save_image(os.path.join(img_dir, '%s.tiff' % fov), fov_channel_vals)

        # save the segmentation labels
        for fov in segmentation_labels_xr.fovs.values:
            fov_whole_cell = segmentation_labels_xr.loc[fov, :, :, 'whole_cell'].values
            fov_nuclear = segmentation_labels_xr.loc[fov, :, :, 'nuclear'].values
            image_utils.save_image(os.path.join(temp_dir, '%s_whole_cell.tiff' % fov),
                                   fov_whole_cell)
            image_utils.save_image(os.path.join(temp_dir, '%s_nuclear.tiff' % fov),
                                   fov_nuclear)

        # invalid channel values passed
        with pytest.raises(ValueError):
            segmentation_utils.save_segmentation_labels(
                segmentation_dir=temp_dir, data_dir=img_dir, output_dir=temp_dir,
                fovs=fov_sub, channels=['bad_chan']
            )

        # test segmentation with channel overlay
        segmentation_utils.save_segmentation_labels(
            segmentation_dir=temp_dir, data_dir=img_dir, output_dir=temp_dir,
            fovs=fov_sub, channels=chan_sub
        )

        # make sure all files and overlay.tiff exists
        assert os.path.exists(os.path.join(temp_dir,
                                           f'{fov_sub[0]}'
                                           f'_segmentation_borders.tiff'))
        assert os.path.exists(os.path.join(temp_dir,
                                           '_'.join([f'{fov_sub[0]}',
                                                     *channel_xr.channels.values[:2],
                                                     'overlay.tiff'])))


def test_concatenate_csv():
    # create sample data
    test_data_1 = test_utils.make_cell_table(n_cells=10, n_markers=5)
    test_data_2 = test_utils.make_cell_table(n_cells=20, n_markers=5)

    with pytest.raises(ValueError):
        # attempt to pass column_values list with different length than number of csv files
        segmentation_utils.concatenate_csv(base_dir="example_base_dir",
                                           csv_files=["example_1.csv", "example_2.csv"],
                                           column_values=["missingno"])

    with tempfile.TemporaryDirectory() as temp_dir:
        # create a sample base_dir
        base_dir = os.path.join(temp_dir, 'base_dir')
        os.mkdir(base_dir)

        # write the sample data
        path_1 = os.path.join(base_dir, "cell_data_1.csv")
        path_2 = os.path.join(base_dir, "cell_data_2.csv")

        test_data_1.to_csv(path_1, index=False, header=False)
        test_data_2.to_csv(path_2, index=False, header=False)

        # create concatenated csv with basic settings
        segmentation_utils.concatenate_csv(base_dir=base_dir,
                                           csv_files=["cell_data_1.csv",
                                                      "cell_data_2.csv"])

        assert os.path.exists(os.path.join(base_dir, "combined_data.csv"))

        # reset for next test
        os.remove(os.path.join(base_dir, "combined_data.csv"))

        # now test with column values
        segmentation_utils.concatenate_csv(base_dir=base_dir,
                                           csv_files=["cell_data_1.csv",
                                                      "cell_data_2.csv"],
                                           column_values=["example_data_1",
                                                          "example_data_2"])

        assert os.path.exists(os.path.join(base_dir, "combined_data.csv"))
