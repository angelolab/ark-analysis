import numpy as np
import xarray as xr
from skimage.measure import regionprops

from ark.segmentation import signal_extraction
import synthetic_spatial_datagen


def test_positive_pixels_extraction():
    # sample params
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
        synthetic_spatial_datagen.generate_two_cell_chan_data(
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

    # test signal counts for different channels
    assert np.all(channel_counts_1 == [25, 0])
    assert np.all(channel_counts_2 == [0, 236])

    # test with new threshold == 10
    kwargs = {'threshold': 10}

    channel_counts_1 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data),
        **kwargs
    )

    channel_counts_2 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data),
        **kwargs
    )

    assert np.all(channel_counts_1 == [0, 0])
    assert np.all(channel_counts_2 == [0, 236])

    # test for multichannel thresholds
    kwargs = {'threshold': np.array([0, 10])}

    channel_counts_1 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data),
        **kwargs
    )

    channel_counts_2 = signal_extraction.positive_pixels_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data),
        **kwargs
    )

    assert np.all(channel_counts_1 == [25, 0])
    assert np.all(channel_counts_2 == [0, 236])


def test_center_weighting_extraction():
    # sample params
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
        synthetic_spatial_datagen.generate_two_cell_chan_data(
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

    # extract the centroids and coords
    region_info = regionprops(sample_segmentation_mask.astype(np.int16))
    kwarg_1 = {'centroid': region_info[0].centroid}
    kwarg_2 = {'centroid': region_info[1].centroid}

    coords_1 = region_info[0].coords
    coords_2 = region_info[1].coords

    channel_counts_1_center_weight = signal_extraction.center_weighting_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data),
        **kwarg_1
    )

    channel_counts_2_center_weight = signal_extraction.center_weighting_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data),
        **kwarg_2
    )

    channel_counts_1_base_weight = signal_extraction.total_intensity_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data)
    )

    channel_counts_2_base_weight = signal_extraction.total_intensity_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data)
    )

    # cell 1 and cell 2 nuclear signal should be lower for weighted than default
    assert channel_counts_1_center_weight[0] < channel_counts_1_base_weight[0]
    assert channel_counts_2_center_weight[1] < channel_counts_2_base_weight[1]

    # assert effect of "bleeding" membrane signal is less with weighted than default
    assert channel_counts_1_center_weight[1] < channel_counts_1_base_weight[1]


def test_total_intensity_extraction():
    # sample params
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
        synthetic_spatial_datagen.generate_two_cell_chan_data(
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

    channel_counts_1 = signal_extraction.total_intensity_extraction(
        cell_coords=coords_1,
        image_data=xr.DataArray(sample_channel_data)
    )

    channel_counts_2 = signal_extraction.total_intensity_extraction(
        cell_coords=coords_2,
        image_data=xr.DataArray(sample_channel_data)
    )

    # test signal counts for different channels
    assert np.all(channel_counts_1 == [250, 0])
    assert np.all(channel_counts_2 == [0, 2360])
