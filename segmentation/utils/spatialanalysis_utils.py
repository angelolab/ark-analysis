import numpy as np
import pandas as pd
import skimage.measure
from scipy.spatial.distance import cdist


def calc_dist_matrix(label_map):
    """Generate matrix of distances between center of pairs of cells
        Args
            label_map: numpy array with unique cells given unique pixel labels
        Returns
            dist_matrix: cells x cells matrix with the euclidian
            distance between centers of corresponding cells"""

    props = skimage.measure.regionprops(label_map)
    a = []
    for i in range(len(props)):
        a.append(props[i].centroid)
    centroids = np.array(a)
    dist_matrix = cdist(centroids, centroids)
    return dist_matrix


def make_distance_matrix(typeOfEnfrichment):
    if(typeOfEnfrichment == "none"):
        randMat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(randMat, 0)
        return randMat
    elif(typeOfEnfrichment == "positive"):
        distMatP = np.zeros((80,80))
        distMatP[10:20, :10] = 50
        distMatP[:10, 10:20] = 50
        distMatP[20:40, :20] = 200
        distMatP[:20, 20:40] = 200
        distMatP[40:80, :40] = 300
        distMatP[:40, 40:80] = 300
        return distMatP
    elif(typeOfEnfrichment == "negative"):
        distMatN = np.zeros((60, 60))
        distMatN[20:40, :20] = 300
        distMatN[:20, 20:40] = 300
        distMatN[40:50, :40] = 50
        distMatN[:40, 40:50] = 50
        distMatN[50:60, :50] = 200
        distMatN[:50, 50:60] = 200
        return distMatN


def make_expression_matrix(typeOfEncrichment):
    if(typeOfEncrichment == "none"):
        cellArray = pd.DataFrame(np.zeros((60, 53)))
        cellArray[0] = 6
        cellArray[1] = np.arange(len(cellArray[1])) + 1
        cellArray.iloc[0:20, 7] = 1
        cellArray.iloc[20:40, 8] = 1
        return cellArray
    elif(typeOfEncrichment == "positive"):
        cellArrayP = pd.DataFrame(np.zeros((80, 53)))
        cellArrayP[0] = 6
        cellArrayP[1] = np.arange(len(cellArrayP[1])) + 1
        cellArrayP.iloc[0:8, 7] = 1
        cellArrayP.iloc[28:30, 7] = 1
        cellArrayP.iloc[38:40, 7] = 1
        cellArrayP.iloc[10:18, 8] = 1
        cellArrayP.iloc[27, 8] = 1
        cellArrayP.iloc[30, 8] = 1
        cellArrayP.iloc[36:38, 8] = 1
        return cellArrayP
    elif(typeOfEncrichment == "negative"):
        cellArrayN = pd.DataFrame(np.zeros((60, 53)))
        cellArrayN[0] = 6
        cellArrayN[1] = np.arange(len(cellArrayN[1])) + 1
        cellArrayN.iloc[0:20, 7] = 1
        cellArrayN.iloc[20:40, 8] = 1
        return cellArrayN
