import numpy as np


def positive_pixels_extraction(cell_coords, image_data, threshold=0):
    """Extract channel counts by summing over the number of non-zero pixels in the cell.

    Improves on default_extraction by not distinguishing between pixel expression values

    Args:
        cell_coords (numpy.ndarray): values representing pixels within one cell
        image_data (xarray.DataArray): array containing channel counts
        threshold (int): where we want to set the cutoff for a positive pixel, default 0

    Returns:
        numpy.ndarray:
            Sums of counts for each channel
    """

    # index into image_data
    channel_values = image_data.values[tuple(cell_coords.T)]

    # create binary mask based on threshold
    channel_counts = np.sum(channel_values > threshold, axis=0)

    return channel_counts


def center_weighting_extraction(cell_coords, image_data, centroid):
    """Extract channel counts by summing over weighted expression values based on distance from
    center.

    Improves upon default extraction by including a level of certainty/uncertainty.

    Args:
        cell_coords (numpy.ndarray):
            values representing pixels within one cell
        image_data (xarray.DataArray):
            array containing channel counts
        centroid (tuple):
            the centroid of the region in question

    Returns:
        numpy.ndarray:
            Sums of counts for each channel
    """

    # compute the distance box-level from the center outward
    weights = np.linalg.norm(cell_coords - centroid, ord=np.inf, axis=1)

    # center the weights around the middle value
    weights = 1 - (weights / (np.max(weights) + 1))

    # retrieve the channel counts
    channel_values = image_data.values[tuple(cell_coords.T)]
    channel_counts = weights.dot(channel_values)

    return channel_counts


def default_extraction(cell_coords, image_data):
    """ Extract channel counts for an individual cell via basic summation for each channel

    Args:
        cell_coords (numpy.ndarray):
            values representing pixels within one cell
        image_data (xarray.DataArray):
            array containing channel counts

    Returns:
        numpy.ndarray:
            Sum of counts for each channel
    """

    # index into image_data to get the channel values we're interested in
    channel_values = image_data.values[tuple(cell_coords.T)]

    # collapse along channels dimension to get counts per channel
    channel_counts = np.sum(channel_values, axis=0)

    return channel_counts
