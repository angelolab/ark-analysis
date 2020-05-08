import pandas as pd
import numpy as np
from skimage import io
from segmentation.utils import spatialanalysis_utils

###Setup input and parameters
point = 6
dataAll = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEnrichment/granA_cellpheno_CS-asinh-norm_revised.csv")
markerInds = [7,8] + list(range(10,44))
dataMarkers = dataAll.loc[:, dataAll.columns[markerInds]]
markerTitles = dataAll.columns[markerInds]
markerNum = len(dataAll.columns[markerInds])

bootstrapNum = 1000
distLim = 113

closeNum = np.zeros((markerNum, markerNum))
closeNumRand = np.zeros((markerNum, markerNum, bootstrapNum))

patientIdx = 0
cellLabelIdx = 1

markerThresh = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
thresh_vec = markerThresh.iloc[1:38, 1]

###Enrichment Analysis

dist_mat = skimage.io.imread("/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/distancesMat.tiff")
patientIds = dataAll.iloc[:, patientIdx] == point
patientData = dataAll[patientIds]
patientDataMarkers = dataMarkers[patientIds]

#later implement outside for loop for dif points
for j in range(0, markerNum):
    marker1_thresh = thresh_vec.iloc[j]
    marker1PosInds = patientDataMarkers[patientDataMarkers.columns[j]] > thresh_vec.iloc[j]
    marker1PosLabels = patientData.loc[marker1PosInds, patientData.columns[cellLabelIdx]]
    marker1Num = len(marker1PosLabels)
    for k in range(0, markerNum):
        marker2_thresh = thresh_vec.iloc[k]
        marker2PosInds = patientDataMarkers[patientDataMarkers.columns[k]] > thresh_vec.iloc[k]
        marker2PosLabels = patientData.loc[marker2PosInds, patientData.columns[cellLabelIdx]]
        marker2Num = len(marker2PosLabels)
        truncDistMat = dist_mat[marker1PosLabels-1, marker2PosLabels-1] #?

        truncDistMapBin = np.zeros(truncDistMat.shape)
        truncDistMapBin[truncDistMat < distLim] = 1

        closeNum[j,k] = sum(sum(dist_mat))
        for r in range(0, bootstrapNum):
            marker1LabelsRand = random.sample(patientData[patientData.columns[cellLabelIdx]], marker1Num)
            marker2LabelsRand = random.sample(patientData[patientData.columns[cellLabelIdx]], marker2Num)
            randTruncDistMat = dist_mat[marker1LabelsRand-1, marker2LabelsRand-1]  #?
            randTruncDistMatBin = np.zeros(randTruncDistMat.shape)
            randTruncDistMatBin[randTruncDistMat < distLim] = 1

            closeNumRand[j,k,r] = sum(sum(randTruncDistMatBin))

#z score, pval, and adj pval


def test_closeNum():
    real_closeNum = pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/closeNum.csv")
    assert np.array_equal(closeNum, real_closeNum)

test_closeNum()