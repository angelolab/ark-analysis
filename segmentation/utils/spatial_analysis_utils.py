import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
from scipy.spatial.distance import cdist


def calc_dist_matrix(label_map):
    """Generate matrix of distances between center of pairs of cells

    Args:
        label_map: numpy array with unique cells given unique pixel labels
    Returns:
        dist_matrix: cells x cells matrix with the euclidian
        distance between centers of corresponding cells"""
    dist_mats_list = []
    for i in range(0, label_map.shape[0]):
        props = skimage.measure.regionprops(label_map[i, :, :])
        a = []
        for j in range(len(props)):
            a.append(props[j].centroid)
        centroids = np.array(a)
        dist_matrix = cdist(centroids, centroids)
        dist_mats_list.append(dist_matrix)
    dist_mats = np.stack(dist_mats_list, axis=0)
    # label_map.coords["fovs"]
    coords = [range(len(dist_mats)), range(dist_mats[0].data.shape[0]), range(dist_mats[0].data.shape[1])]
    dims = ["points", "rows", "cols"]
    dist_mats_xr = xr.DataArray(dist_mats, coords=coords, dims=dims)
    return dist_mats_xr
