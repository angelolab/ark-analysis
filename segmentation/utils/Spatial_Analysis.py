import numpy as np
import math
import skimage.measure
from scipy.spatial.distance import cdist


def calc_dist_matrix(label_map):
    """Generate matrix of distances between center of pairs of cells
        Args
            label_map: numpy array with unique cells given unique pixel labels
        Returns
            dist_matrix: cells x cells matrix with the euclidian
            distance between centers of corresponding cells"""

    if len(np.unique(label_map)) < 3:
        raise ValueError("Array must be provided in labeled format")

    props = skimage.measure.regionprops(label_map)
    a = [None] * len(props)
    for i in range(len(props)):
        a[i] = props[i].centroid
    centroids = np.array(a)
    dist_matrix = cdist(centroids, centroids)
    return dist_matrix