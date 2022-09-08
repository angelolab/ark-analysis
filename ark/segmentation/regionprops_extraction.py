
import numpy as np
import pandas as pd
from skimage.measure import label, moments, regionprops_table


def major_minor_axis_ratio(prop, **kwargs):
    """Return the ratio of the major axis length to the minor axis length

    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments

    Returns:
        float:
            major axis length / minor axis length
    """
    if prop.minor_axis_length == 0:
        return np.float('NaN')
    else:
        return prop.major_axis_length / prop.minor_axis_length


def perim_square_over_area(prop, **kwargs):
    """Return the ratio of the squared perimeter to the cell area

    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments

    Returns:
        float:
            perimeter^2 / area
    """

    return np.square(prop.perimeter) / prop.area


def major_axis_equiv_diam_ratio(prop, **kwargs):
    """Return the ratio of the major axis length to the equivalent diameter

    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments

    Returns:
        float:
            major axis length / equivalent diameter
    """

    return prop.major_axis_length / prop.equivalent_diameter


def convex_hull_resid(prop, **kwargs):
    """Return the ratio of the difference between convex area and area to convex area

    Args:
        prop (skimage.measure.regionprops):
            The property information for a cell returned by regionprops
        **kwargs:
            Arbitrary keyword arguments

    Returns:
        float:
            (convex area - area) / convex area
    """

    return (prop.convex_area - prop.area) / prop.convex_area


def centroid_dif(prop, **kwargs):
    """Return the normalized euclidian distance between the centroid of the cell
    and the centroid of the corresponding convex hull

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
        labeled_diff_img = label(diff_img, connectivity=1)
        hull_prop_df = pd.DataFrame(regionprops_table(labeled_diff_img,
                                                      properties=['area', 'perimeter']))
        hull_prop_df['compactness'] = np.square(hull_prop_df['perimeter']) / hull_prop_df['area']

        small_idx_area_cutoff = kwargs.get('small_concavity_minimum', 10)
        compactness_cutoff = kwargs.get('max_compactness', 60)
        large_idx_area_cutoff = kwargs.get('large_concavity_minimum', 150)

        small_idx = np.logical_and(hull_prop_df['area'] > small_idx_area_cutoff,
                                   hull_prop_df['compactness'] < compactness_cutoff)
        large_idx = hull_prop_df['area'] > large_idx_area_cutoff
        combined_idx = np.logical_or(small_idx, large_idx)

        concavities = np.sum(combined_idx)
    else:
        concavities = 0

    return concavities


def nc_ratio(marker_counts, **kwargs):
    """Return the ratio of the nuclear area to total cell area

    Args:
        marker_counts (xarray.DataArray):
            xarray containing segmentaed data of cells x markers
        **kwargs:
            Arbitrary keyword arguments
    """

    # get the whole cell and nuclear area information
    whole_cell_areas = marker_counts.loc['whole_cell', :, 'area']
    nuclear_areas = marker_counts.loc['nuclear', :, 'area']

    # compute nc_ratio by dividing nuclear by whole cell area, set infs to 0
    marker_counts.loc['nuclear', :, 'nc_ratio'] = np.nan_to_num(nuclear_areas / whole_cell_areas,
                                                                posinf=0, neginf=0)

    # copy nc_ratio to whole_cell because it applies to both dimensions
    marker_counts.loc['whole_cell', :, 'nc_ratio'] = marker_counts.loc['nuclear', :, 'nc_ratio']

    return marker_counts


REGIONPROPS_FUNCTION = {
    'major_minor_axis_ratio': major_minor_axis_ratio,
    'perim_square_over_area': perim_square_over_area,
    'major_axis_equiv_diam_ratio': major_axis_equiv_diam_ratio,
    'convex_hull_resid': convex_hull_resid,
    'centroid_dif': centroid_dif,
    'num_concavities': num_concavities,
    'nc_ratio': nc_ratio
}
