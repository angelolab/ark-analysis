import numpy as np
import pandas as pd

import ark.settings as settings


def calculate_median_distance_to_cell_type(
        cell_df, dist_xr, cell_cluster, k, cell_type_col=settings.CELL_TYPE,
        cell_label_col=settings.CELL_LABEL):
    """Function to calculate median distance of all cells to a specified cell type
    Args:
        cell_df (pd.DataFrame):
            Dataframe containing all cells and their cell type
        dist_xr (xr.array):
            Cell by cell distances for all cells
        cell_cluster (str):
            Cell cluster to calculate distance to
        k (int):
            Number of nearest neighbours
        cell_type_col (str):
            column with the cell phenotype
        cell_label_col (str):
            column with the cell labels

    Returns:
        np.array:
            mean distances for each cell to the cluster cells
    """

    # get cell ids for all cells of specific cluster
    j = cell_df.loc[cell_df[cell_type_col] == cell_cluster, cell_label_col]

    # get all cells that match specified cell cluster
    dist_xr = dist_xr.loc[:, dist_xr.dim_1.isin(j)]

    # keep the closest k values, not included itself
    dist_xr = dist_xr.where(dist_xr > 0)
    sorted_dist = np.sort(dist_xr.values, axis=1)
    sorted_dist = sorted_dist[:, :k]

    # take the median
    mean_dists = sorted_dist.mean(axis=1)

    return mean_dists


def calculate_median_distance_to_all_cell_types(
        cell_df, dist_xr, k, cell_type_col=settings.CELL_TYPE):
    """Wrapper function to calculate median distance of all cells against all cell types
    Args:
        cell_df (pd.DataFrame):
            Dataframe containing all cells and their cell type
        dist_xr (xr.array):
            Cell by cell distances for all cells
        k (int):
            Number of nearest neighbours
        cell_type_col (str):
            column with the cell phenotype

    Returns:
        pd.DataFrame:
            average distances
    """

    # get all cell clusters in cell table
    all_clusters = np.unique(cell_df[cell_type_col])

    # call calculate_median_distance_to_cell_type for all cell clusters
    avg_dists = pd.DataFrame(index=cell_df.index.values, columns=all_clusters)
    for cell_cluster in all_clusters:
        avg_dists.loc[:, cell_cluster] = calculate_median_distance_to_cell_type(
            cell_df, dist_xr, cell_cluster, k, cell_type_col)
        
    return avg_dists
