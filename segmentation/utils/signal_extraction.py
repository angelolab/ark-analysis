import numpy as np


def default_extraction(cell_coords, image_data):
    """Extract channel counts for an individual cell

    Args:
        cell_coords: tuples of values representing pixels within one cell
        image_data: array containing channel counts

    Returns:
        np.array: sum of counts for each channel
    """

    # transpose coords so they can be used to index an array
    cell_coords = cell_coords.T
    channel_values = image_data.values[tuple(cell_coords)]

    # collapse along channels dimension to get counts per channel
    channel_counts = np.sum(channel_values, axis=0)

    return channel_counts
