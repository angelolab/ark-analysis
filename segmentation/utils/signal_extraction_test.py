import numpy as np
import xarray as xr

from segmentation.utils import signal_extraction
from segmentation.utils import synthetic_spatial_datagen

from skimage.measure import regionprops
from skimage.draw import circle_perimeter


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
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(size_img=size_img,
                                                                                cell_radius=cell_radius,
                                                                                nuc_radius=nuc_radius,
                                                                                memb_thickness=memb_thickness,
                                                                                nuc_signal_strength=nuc_signal_strength,
                                                                                memb_signal_strength=memb_signal_strength,
                                                                                nuc_uncertainty_length=nuc_uncertainty_length,
                                                                                memb_uncertainty_length=memb_uncertainty_length)

    # extract the cell regions for cells 1 and 2
    coords_1 = np.argwhere(sample_segmentation_mask == 1)
    coords_2 = np.argwhere(sample_segmentation_mask == 2)

    # test default extraction (threshold == 0)
    channel_counts_1 = signal_extraction.positive_pixels_extraction(cell_coords=coords_1,
                                                                    image_data=xr.DataArray(sample_channel_data))

    channel_counts_2 = signal_extraction.positive_pixels_extraction(cell_coords=coords_2,
                                                                    image_data=xr.DataArray(sample_channel_data))

    # note that for cell 2 it's higher because of membrane-level expression
    assert np.all(channel_counts_1 == [25, 0])
    assert np.all(channel_counts_2 == [0, 236])

    # test with new threshold == 10 (because that's what we label nuclear signal for the time being)
    # nuclear signal extraction should now be 0
    test_threshold = 10

    channel_counts_1 = signal_extraction.positive_pixels_extraction(cell_coords=coords_1,
                                                                    image_data=xr.DataArray(sample_channel_data),
                                                                    threshold=test_threshold)

    channel_counts_2 = signal_extraction.positive_pixels_extraction(cell_coords=coords_2,
                                                                    image_data=xr.DataArray(sample_channel_data),
                                                                    threshold=test_threshold)

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
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(size_img=size_img,
                                                                                cell_radius=cell_radius,
                                                                                nuc_radius=nuc_radius,
                                                                                memb_thickness=memb_thickness,
                                                                                nuc_signal_strength=nuc_signal_strength,
                                                                                memb_signal_strength=memb_signal_strength,
                                                                                nuc_uncertainty_length=nuc_uncertainty_length,
                                                                                memb_uncertainty_length=memb_uncertainty_length)

    # a lot of this is pretty clunky right now, but essentially we're testing here that
    # weighted extraction indeed leads to less confidence beyond the border of the nucleus
    # or membrane depending on whether

    # extract the cell regions for cells 1 and 2
    coords_1 = np.argwhere(sample_segmentation_mask == 1)
    coords_2 = np.argwhere(sample_segmentation_mask == 2)

    channel_counts_1 = signal_extraction.center_weighting_extraction(cell_coords=coords_1,
                                                                     image_data=xr.DataArray(sample_channel_data))

    channel_counts_2 = signal_extraction.center_weighting_extraction(cell_coords=coords_2,
                                                                     image_data=xr.DataArray(sample_channel_data))

    # the reason we get weird values for channel counts 1 is because cell 2 has membrane-level expression
    # with intentional uncertainty added to it, meaning some of its signal "bleeds" into cell 1's signal

    # TODO: test if this function actually works as intended around the boundaries, aka
    # the effect of wrong signal being added to a cell is reduced with a center weighting technique
    assert np.all(channel_counts_1.astype(np.int16) == [448, 88])
    assert np.all(channel_counts_2.astype(np.int16) == [0, 2329])


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
        synthetic_spatial_datagen.generate_two_cell_test_channel_synthetic_data(size_img=size_img,
                                                                                cell_radius=cell_radius,
                                                                                nuc_radius=nuc_radius,
                                                                                memb_thickness=memb_thickness,
                                                                                nuc_signal_strength=nuc_signal_strength,
                                                                                memb_signal_strength=memb_signal_strength,
                                                                                nuc_uncertainty_length=nuc_uncertainty_length,
                                                                                memb_uncertainty_length=memb_uncertainty_length)

    # extract the cell regions for cells 1 and 2
    coords_1 = np.argwhere(sample_segmentation_mask == 1)
    coords_2 = np.argwhere(sample_segmentation_mask == 2)

    channel_counts_1 = signal_extraction.default_extraction(cell_coords=coords_1,
                                                            image_data=xr.DataArray(sample_channel_data))

    channel_counts_2 = signal_extraction.default_extraction(cell_coords=coords_2,
                                                            image_data=xr.DataArray(sample_channel_data))

    # note that for cell 2 it's higher because of membrane-level expression
    assert np.all(channel_counts_1 == [250, 0])
    assert np.all(channel_counts_2 == [0, 2360])
