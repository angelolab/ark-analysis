import numpy as np


# TODO: work on weighted extraction and implement other discussed techniques of extraction
def positive_pixels_extraction(cell_coords, image_data):
    """
    Extract channel counts by summing over the number of non-zero pixels in the cell,
    improves on default_extraction by not distinguishing between pixel expression values

    Args:
        cell_coords (numpy): values representing pixels within one cell
        image_data (xarray): array containing channel counts

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    # transpose coords so they can be used to index an array
    cell_coords = cell_coords.T
    channel_values = image_data.values[tuple(cell_coords)]

    # sum up based on a binary mask that is 1 if the expression value > 0 else 0
    channel_counts = np.sum(channel_values > 0, axis=0)

    return channel_counts


def center_weighting_extraction(cell_coords, image_data):
    """
    Extract channel counts by summing over weighted expression values based on distance from center,
    improves upon default extraction by including a level of certainty/uncertainty

    Args:
        cell_coords (numpy): values representing pixels within one cell
        image_data (xarray): array containing channel counts

    Returns:
        channel_counts (numpy): sum of counts for each channel
    """

    # create a weighting matrix, we will assign the highest value at the center
    # and decrease by 1 each layer we move away from it
    # I'm going to be assume we get square matrices in image_data
    center = int(image_data.shape[0] / 2)

    # this will ensure we never get zero or negative weighting values in our weight matrix
    center_weight = center + 1

    # now build the weight matrix using ogrid
    image_row_index, image_col_index = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
    weight_matrix = center_weight - np.maximum(np.abs(image_row_index - center), np.abs(image_col_index - center))

    # TODO: the center of even-length arrays is not a single pixel which currently causes
    # minor signal irregularities at different layers
    # ex. in a 4 x 4 matrix the center should be:
    # 0 0 0 0
    # 0 X X 0
    # 0 X X 0
    # 0 0 0 0

    # and not:
    # 0 0 0 0
    # 0 0 0 0
    # 0 0 X 0
    # 0 0 0 0
    # as would be defined without the correction by the regular signal weighting algorithm
    # the easy way to fix this is to use skimage.draw.rectangle

    # center the weight matrix by center_weight
    weight_matrix = weight_matrix / center_weight

    # transpose coords so they can be used to index an array
    cell_coords = cell_coords.T
    channel_values = image_data.values[tuple(cell_coords)]
    weight_values = np.expand_dims(weight_matrix[tuple(cell_coords)], axis=0)

    # multiply channel_values by weight_values and then sum across channels
    channel_counts = np.sum(np.multiply(channel_values, weight_values.T), axis=0)

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

    # transpose coords so they can be used to index an array
    cell_coords = cell_coords.T
    channel_values = image_data.values[tuple(cell_coords)]

    # collapse along channels dimension to get counts per channel
    channel_counts = np.sum(channel_values, axis=0)

    return channel_counts
