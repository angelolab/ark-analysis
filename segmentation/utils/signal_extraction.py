import numpy as np


# TODO: work on weighted extraction and implement other discussed techniques of extraction
def positive_pixels(cell_coords, image_data):
    """
    Extract channel counts by summing over the number of non-zero pixels in the cell,
    improves on default_extraction by not distinguishing between pixel expression values

    Args:
        cell_coords (tuple): tuples of values representing pixels within one cell
        image_data (numpy): array containing channel counts

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    pass


def center_weighting(cell_coords, image_data):
    """
    Extract channel counts by summing over weighted expression values based on distance from center,
    improves upon default extraction by including a level of certainty/uncertainty
    """

    pass


def default_extraction(cell_coords, image_data):
    """
    Extract channel counts for an individual cell via basic summation for each channel

    Args:
        cell_coords (tuple): tuples of values representing pixels within one cell
        image_data (numpy): array containing channel counts

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    # transpose coords so they can be used to index an array
    cell_coords = cell_coords.T
    channel_values = image_data.values[tuple(cell_coords)]

    # collapse along channels dimension to get counts per channel
    channel_counts = np.sum(channel_values, axis=0)

    return channel_counts
