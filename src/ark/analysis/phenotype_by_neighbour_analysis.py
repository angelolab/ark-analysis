import numpy as np
import pandas as pd
import xarray as xr

import scipy
import netCDF4


# calculate median distance from a specific cell to all other cells of a specified cell cluster
def calculate_median_distance_to_cell_type(cell_df, dist_xr, cell_cluster, N, cell_cluster_column = 'cell_cluster', cell_id_column = 'label'):
    """Function to calculate median distance of all cels to a specified cell type
    Args:
        cell_df (pandas.DataFrame):
            Dataframe containg all cells and their cell type
        dist_xr (xrarray):
            Cell by cell distance xrarray all cells
        cell_cluster (str):
            Cell cluster to calculate distance to
        N (int):
            Number of nearest neighbours
    """
    # get cell ids for all cells of specific cluster
    j = cell_df.loc[cell_df[cell_cluster_column] == cell_cluster,cell_id_column]
    # get all cells that match specified cell cluster
    dist_xr = dist_xr.loc[:, dist_xr.dim_1.isin(j)]
    # convert self to nans 
    dist_xr = dist_xr.where(dist_xr > 0)
    # sort values 
    sorted_dist = np.sort(dist_xr.values, axis=1)
    # keep the closest N values
    sorted_dist = sorted_dist[:, :N]
    # take the median
    mean_dist = sorted_dist.mean(axis=1)
    return(mean_dist)


# create function that wraps above function to analyze all cell clusters 
def calculate_median_distance_to_all_cell_types(cell_df, dist_xr, N, cell_cluster_column = 'cell_cluster'):
    """Wrapper function to calculate median distance of all cells against all cell types
    Args:
        cell_df (pandas.DataFrame):
            Dataframe containg all cells and their cell type
        dist_xr (xrarray):
            Cell by cell distance xrarray all cells
        N (int):
            Number of nearest neighbours
    """
    # get all cell clusters in cell table
    all_clusters = np.unique(cell_df[cell_cluster_column])
    # call calculate_median_distance_to_cell_type for all cell clusters
    avgdists = pd.DataFrame(index = cell_df.index.values, columns = all_clusters)
    for cell_cluster in all_clusters:
        avgdists.loc[:,cell_cluster] = calculate_median_distance_to_cell_type(cell_df, dist_xr, cell_cluster, N)
        
    return(avgdists)