import pandas as pd
import numpy as np
from skimage import io
from segmentation.utils import spatialanalysis_utils

cell_array = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEnrichment/granA_cellpheno_CS-asinh-norm_revised.csv")
marker_thresholds = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
dist_matrix = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header = None))

test_cellarray = pd.read_csv("/Users/jaiveersingh/Desktop/tests/celllabels.csv")
test_thresholds = pd.read_csv("/Users/jaiveersingh/Desktop/tests/thresholds.csv")
test_distmat = np.asarray(pd.read_csv("/Users/jaiveersingh/Desktop/tests/distmat.csv", header = None))

def spatial_analysis(dist_matrix, marker_thresholds, cell_array):
    ###Setup input and parameters
    point = 6
    dataAll = cell_array
    markerInds = [7, 8] + list(range(10,44))
    dataMarkers = dataAll.loc[:, dataAll.columns[markerInds]]
    markerTitles = dataAll.columns[markerInds]
    markerNum = len(markerTitles)

    bootstrapNum = 3
    distLim = 100

    closeNum = np.zeros((markerNum, markerNum), dtype = 'int')
    closeNumRand = np.zeros((markerNum, markerNum, bootstrapNum), dtype = 'int')

    patientIdx = 0
    cellLabelIdx = 1

    markerThresh = marker_thresholds
    thresh_vec = markerThresh.iloc[1:38, 1]

    ###Enrichment Analysis

    dist_mat = dist_matrix
    patientIds = dataAll.iloc[:, patientIdx] == point
    patientData = dataAll[patientIds]
    patientDataMarkers = dataMarkers[patientIds]

    #later implement outside for loop for dif points
    for j in range(0, markerNum):
        marker1_thresh = thresh_vec.iloc[j]
        marker1PosInds = patientDataMarkers[patientDataMarkers.columns[j]] > marker1_thresh
        marker1PosLabels = patientData.loc[marker1PosInds, patientData.columns[cellLabelIdx]]
        marker1Num = len(marker1PosLabels)
        for k in range(0, markerNum):
            marker2_thresh = thresh_vec.iloc[k]
            marker2PosInds = patientDataMarkers[patientDataMarkers.columns[k]] > marker2_thresh
            marker2PosLabels = patientData.loc[marker2PosInds, patientData.columns[cellLabelIdx]]
            marker2Num = len(marker2PosLabels)
            truncDistMat = dist_mat[np.ix_(np.asarray(marker1PosLabels-1), np.asarray(marker2PosLabels-1))]
            #truncDistMat = dist_mat[marker1PosLabels-1, marker2PosLabels-1] ?

            truncDistMatBin = np.zeros(truncDistMat.shape, dtype = 'int')
            truncDistMatBin[truncDistMat < distLim] = 1

            closeNum[j,k] = np.sum(np.sum(truncDistMatBin))
            for r in range(0, bootstrapNum):
                marker1LabelsRand = patientData[patientData.columns[cellLabelIdx]].sample(n = marker1Num, replace = True)
                marker2LabelsRand = patientData[patientData.columns[cellLabelIdx]].sample(n = marker2Num, replace = True)
                randTruncDistMat = dist_mat[np.ix_(np.asarray(marker1LabelsRand - 1), np.asarray(marker2LabelsRand - 1))]
                #randTruncDistMat = dist_mat[marker1LabelsRand-1, marker2LabelsRand-1]  ?
                randTruncDistMatBin = np.zeros(randTruncDistMat.shape)
                randTruncDistMatBin[randTruncDistMat < distLim] = 1

                closeNumRand[j,k,r] = np.sum(np.sum(randTruncDistMatBin))
    return closeNum
#z score, pval, and adj pval


def test_spatial_analysis():
    #_closeNum = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/true_closeNum.csv", header = None))
    # closeNum = spatial_analysis()
    test_closeNum = spatial_analysis(test_distmat, test_thresholds, test_cellarray)
    assert test_closeNum[0, 0] == 16
    assert test_closeNum[0, 1] == 16
    assert test_closeNum[1, 0] == 16
    assert test_closeNum[1, 1] == 16
    assert test_closeNum[2, 2] == 25
    assert test_closeNum[2, 3] == 25
    assert test_closeNum[3, 2] == 25
    assert test_closeNum[3, 3] == 25
    assert test_closeNum[4, 4] == 1
    assert test_closeNum[5, 4] == 1
    assert test_closeNum[4, 5] == 1
    assert test_closeNum[5, 5] == 1

    #Now test with Erin's output
    closeNum = spatial_analysis(dist_matrix, marker_thresholds, cell_array)
    real_closeNum = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/closeNum.csv"))
    assert np.array_equal(closeNum, real_closeNum)

test_spatial_analysis()