import copy

import numpy as np
import pandas as pd

from skimage.measure import label, moments, regionprops_table


def centroid_dif(prop, **kwargs):
    """Return the normalized euclidian distance between the centroid of the cell and the centroid of the corresponding convex hull

    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments

    Returns:
        float:
            The centroid shift for the cell
    """

    cell_image = prop.image
    cell_M = moments(cell_image)
    cell_centroid = np.array([cell_M[1, 0] / cell_M[0, 0], cell_M[0, 1] / cell_M[0, 0]])

    convex_image = prop.convex_image
    convex_M = moments(convex_image)
    convex_centroid = np.array([convex_M[1, 0] / convex_M[0, 0], convex_M[0, 1] / convex_M[0, 0]])

    centroid_dist = np.linalg.norm(cell_centroid - convex_centroid) / np.sqrt(prop.area)

    return centroid_dist


def num_concavities(prop, **kwargs):
    """Return the number of concavities for a cell

    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments

    Returns:
        int:
            The number of concavities for a cell
    """

    cell_image = prop.image
    convex_image = prop.convex_image

    diff_img = convex_image ^ cell_image

    if np.sum(diff_img) > 0:
        labeled_diff_img = label(diff_img)
        hull_prop_df = pd.DataFrame(regionprops_table(labeled_diff_img,
                                                      properties=['area', 'perimeter']))
        hull_prop_df['compactness'] = np.square(hull_prop_df['perimeter']) / hull_prop_df['area']

        small_idx_area_cutoff = kwargs.get('small_idx_area_cutoff', 10)
        compactness_cutoff = kwargs.get('compactness_cutoff', 60)
        large_idx_area_cutoff = kwargs.get('large_idx_area_cutoff', 150)

        small_idx = np.logical_and(hull_prop_df['area'] > small_idx_area_cutoff,
                                   hull_prop_df['compactness'] < compactness_cutoff)
        large_idx = hull_prop_df['area'] > large_idx_area_cutoff
        combined_idx = np.logical_or(small_idx, large_idx)

        concavities = np.sum(combined_idx)
    else:
        concavities = 0

    return concavities


REGIONPROPS_FUNCTION = {
    'centroid_dif': centroid_dif,
    'num_concavities': num_concavities
}
