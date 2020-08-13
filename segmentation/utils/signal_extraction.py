import numpy as np
from skimage.measure import regionprops

# TODO: work on weighted extraction and implement other discussed techniques of extraction
def positive_pixels_extraction(cell_coords, image_data, threshold=0):
    """
    Extract channel counts by summing over the number of non-zero pixels in the cell,
    improves on default_extraction by not distinguishing between pixel expression values

    Args:
        cell_coords (numpy): values representing pixels within one cell
        image_data (xarray): array containing channel counts

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    # index indo image_data to get th channel values we're interested in
    channel_values = image_data.values[tuple(cell_coords.T)]

    # sum up based on a binary mask that is 1 if the expression value > threshold else 0
    channel_counts = np.sum(channel_values > threshold, axis=0)

    return channel_counts


def center_weighting_extraction(cell_coords, image_data):
    """
    Extract channel counts by summing over weighted expression values based on distance from center,
    improves upon default extraction by including a level of certainty/uncertainty

    Args:
        cell_coords (numpy): values representing pixels within one cell
        image_data (xarray): array containing channel counts
        segmentation_mask (numpy): array containing segmentation labels for each cell
        cell_label (int): the cell number we wish to extract for each cell, important to distinguish
            which centroid we want in regionprops because this function only extracts signal
            for one cell, though we may choose to change this in the future...

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    # we could use regionprops but it's probably better to use this method (ala A-Kag's)
    # because it doesn't require you to know which cell_label to index into
    centroid = np.sum(cell_coords, axis=0) / cell_coords.shape[0]
    centroid = centroid.astype(np.int16)

    # this will ensure we never get zero or negative weighting values in our weight matrix
    # I'm currently rounding but it's sort of arbitrary, may decide to truncate (aka round down)
    # in case of decimal centroid values returned by regionprops
    center_weight = max(round(centroid[0]), round(centroid[1])) + 1

    # TODO: this is still not a perfect way to do this for even image dimensions
    # but I think Adam's suggested method might fix this
    image_row_index, image_col_index = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
    weight_matrix = center_weight - np.maximum(np.abs(image_row_index - round(centroid[0])), np.abs(image_col_index - round(centroid[1])))

    # now center the weight matrix around value 1
    weight_matrix = weight_matrix / center_weight

    # index indo image_data to get th channel values we're interested in
    channel_values = image_data.values[tuple(cell_coords.T)]
    weight_values = np.expand_dims(weight_matrix[tuple(cell_coords.T)], axis=0)

    # multiply channel_values by weight_values and then sum across channels
    channel_counts = weight_values.dot(channel_values)

    return channel_counts


def default_extraction(cell_coords, image_data):
    """
    Extract channel counts for an individual cell via basic summation for each channel

    Args:
        cell_coords (numpy): values representing pixels within one cell
        image_data (xarray): array containing channel counts

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    # index indo image_data to get th channel values we're interested in
    channel_values = image_data.values[tuple(cell_coords.T)]

    # collapse along channels dimension to get counts per channel
    channel_counts = np.sum(channel_values, axis=0)

    return channel_counts
