import numpy as np
import xarray as xr

from ark.segmentation import signal_extraction
from ark.utils import synthetic_spatial_datagen

from skimage.measure import regionprops


def test_positive_pixels_extraction():
    # this function tests the functionality of positive pixels extraction
    # where we count the number of non-zero values for each channel

    # configure your parameters here
    size_img = (1024, 1024)
    cell_radius = 10
    nuc_radius = 3
    memb_thickness = 5
    nuc_signal_strength = 10
    memb_signal_strength = 100
    nuc_uncertainty_length = 0
    memb_uncertainty_length = 0

    # generate sample segmentation mask and channel data
    sample_segmentation_mask, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(
            size_img=size_img,
            cell_radius=cell_radius,
            nuc_radius=nuc_radius,
            memb_thickness=memb_thickness,
            nuc_signal_strength=nuc_signal_strength,
            memb_signal_strength=memb_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length,
            memb_uncertainty_length=memb_uncertainty_length
        )

    # extract the cell regions for cells 1 and 2
    coords_1 = np.argwhere(sample_segmentation_mask == 1)
    coords_2 = np.argwhere(sample_segmentation_mask == 2)

    # test default extraction (threshold == 0)
    channel_counts_1 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data)
    )

    channel_counts_2 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data)
    )

    # note that for cell 2 it's higher because of membrane-level expression
    assert np.all(channel_counts_1 == [25, 0])
    assert np.all(channel_counts_2 == [0, 236])

    # test with new threshold == 10 (that's what we currently label nuclear signal)
    # nuclear signal extraction should now be 0
    test_threshold = 10

    channel_counts_1 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data),
        threshold=test_threshold
    )

    channel_counts_2 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data),
        threshold=test_threshold
    )

    assert np.all(channel_counts_1 == [0, 0])
    assert np.all(channel_counts_2 == [0, 236])


def test_center_weighting_extraction():
    # this function tests the functionality of center weighting extraction
    # where we add a weighting scheme with more confidence toward the center
    # before summing across each channel

    # configure your parameters here
    size_img = (1024, 1024)
    cell_radius = 10
    nuc_radius = 3
    memb_thickness = 5
    nuc_signal_strength = 10
    memb_signal_strength = 10
    nuc_uncertainty_length = 1
    memb_uncertainty_length = 1

    # generate sample segmentation mask and channel data
    sample_segmentation_mask, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(
            size_img=size_img,
            cell_radius=cell_radius,
            nuc_radius=nuc_radius,
            memb_thickness=memb_thickness,
            nuc_signal_strength=nuc_signal_strength,
            memb_signal_strength=memb_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length,
            memb_uncertainty_length=memb_uncertainty_length
        )

    # extract the cell regions for cells 1 and 2
    coords_1 = np.argwhere(sample_segmentation_mask == 1)
    coords_2 = np.argwhere(sample_segmentation_mask == 2)

    # generate region info using regionprops, used to extract the centroids and coords
    region_info = regionprops(sample_segmentation_mask.astype(np.int16))
    centroid_1 = region_info[0].centroid
    centroid_2 = region_info[1].centroid

    # could use np.argwhere for this but might as well standardize the entire thing
    coords_1 = region_info[0].coords
    coords_2 = region_info[1].coords

    channel_counts_1_center_weight = signal_extraction.center_weighting_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data),
        centroid=centroid_1
    )

    channel_counts_2_center_weight = signal_extraction.center_weighting_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data),
        centroid=centroid_2
    )

    channel_counts_1_base_weight = signal_extraction.default_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data)
    )

    channel_counts_2_base_weight = signal_extraction.default_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data)
    )

    # assert that the nuclear signal for cell 1 is lower for weighted than for base
    # same for membrane signal for cell 2
    assert channel_counts_1_center_weight[0] < channel_counts_1_base_weight[0]
    assert channel_counts_2_center_weight[1] < channel_counts_2_base_weight[1]

    # we intentionally bled membrane signal from cell 2 into cell 1
    # a weighted signal technique will ensure that this bleeding will be curbed
    # thus the signal noise will be drastically reduced
    # so there will not be as much membrane noise in cell 1 in this case
    assert channel_counts_1_center_weight[1] < channel_counts_1_base_weight[1]


def test_default_extraction():
    # this function tests the functionality of default weighting extraction
    # where we just sum across each channel

    # configure your parameters here
    size_img = (1024, 1024)
    cell_radius = 10
    nuc_radius = 3
    memb_thickness = 5
    nuc_signal_strength = 10
    memb_signal_strength = 10
    nuc_uncertainty_length = 0
    memb_uncertainty_length = 0

    # generate sample segmentation mask and channel data
    sample_segmentation_mask, sample_channel_data = \
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(
            size_img=size_img,
            cell_radius=cell_radius,
            nuc_radius=nuc_radius,
            memb_thickness=memb_thickness,
            nuc_signal_strength=nuc_signal_strength,
            memb_signal_strength=memb_signal_strength,
            nuc_uncertainty_length=nuc_uncertainty_length,
            memb_uncertainty_length=memb_uncertainty_length
        )

    # extract the cell regions for cells 1 and 2
    coords_1 = np.argwhere(sample_segmentation_mask == 1)
    coords_2 = np.argwhere(sample_segmentation_mask == 2)

    channel_counts_1 = signal_extraction.default_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data)
    )

    channel_counts_2 = signal_extraction.default_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data)
    )

    # note that for cell 2 it's higher because of membrane-level expression
    assert np.all(channel_counts_1 == [250, 0])
    assert np.all(channel_counts_2 == [0, 2360])
