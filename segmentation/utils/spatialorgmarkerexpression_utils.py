import pandas as pd
import numpy as np
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

# Erin's Data Inputs
cell_array = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEn"
                         "richment/granA_cellpheno_CS-asinh-norm_revised.csv")
marker_thresholds = pd.read_csv("/Users/jaiveersingh/Downloads/Sp"
                                "atialEnrichment/markerThresholds.csv")
dist_matrix = np.asarray(pd.read_csv("/Users/jaiveersingh/Documen"
                                     "ts/MATLAB/distancesMat5.csv",
                                     header=None))

# pv_cellarray = pd.read_csv("/Users/jaive
# ersingh/Desktop/tests/pvcelllabel.csv")

# pv_distmat = np.asarray(pd.read_csv("/Us
# ers/jaiveersingh/Desktop/tests/pvdistmat.csv", header = None))

# pv_cellarrayN = pd.read_csv("/Users/jaiv
# eersingh/Desktop/tests/pvcellarrayN.csv")

# pv_distmatN = np.asarray(pd.read_csv("/Use
# rs/jaiveersingh/Desktop/tests/pvdistmatN.csv", header = None))

# randMat = np.random.randint(0,200,size=(60,60))
# np.fill_diagonal(randMat, 0)

# pv_cellarrayR = pd.read_csv("/Users/jaiveers
# ingh/Desktop/tests/pvcellarrayR.csv")


def spatial_analysis(dist_matrix, marker_thresholds, cell_array):
    # Setup input and parameters
    point = 6
    dataAll = cell_array
    markerInds = [7, 8] + list(range(10, 44))
    dataMarkers = dataAll.loc[:, dataAll.columns[markerInds]]
    markerTitles = dataAll.columns[markerInds]
    markerNum = len(markerTitles)

    bootstrapNum = 100
    distLim = 100

    closeNum = np.zeros((markerNum, markerNum), dtype='int')
    closeNumRand = np.zeros((markerNum, markerNum, bootstrapNum), dtype='int')

    patientIdx = 0
    cellLabelIdx = 1

    markerThresh = marker_thresholds
    thresh_vec = markerThresh.iloc[1:38, 1]

    # Enrichment Analysis

    dist_mat = dist_matrix
    patientIds = dataAll.iloc[:, patientIdx] == point
    patientData = dataAll[patientIds]
    patientDataMarkers = dataMarkers[patientIds]

    # later implement outside for loop for dif points
    for j in range(0, markerNum):
        marker1_thresh = thresh_vec.iloc[j]
        marker1PosInds = patientDataMarkers
        [patientDataMarkers.columns[j]] > marker1_thresh
        marker1PosLabels = patientData.loc[
            marker1PosInds, patientData.columns[cellLabelIdx]]
        marker1Num = len(marker1PosLabels)
        for k in range(0, markerNum):
            marker2_thresh = thresh_vec.iloc[k]
            marker2PosInds = patientDataMarkers
            [patientDataMarkers.columns[k]] > marker2_thresh
            marker2PosLabels = patientData.loc[
                marker2PosInds, patientData.columns[cellLabelIdx]]
            marker2Num = len(marker2PosLabels)
            truncDistMat = dist_mat[np.ix_(
                np.asarray(marker1PosLabels-1), np.asarray(
                    marker2PosLabels-1))]

            truncDistMatBin = np.zeros(truncDistMat.shape, dtype='int')
            truncDistMatBin[truncDistMat < distLim] = 1

            closeNum[j, k] = np.sum(np.sum(truncDistMatBin))
            for r in range(0, bootstrapNum):
                marker1LabelsRand = patientData[
                    patientData.columns[cellLabelIdx]].sample(
                    n=marker1Num, replace=True)
                marker2LabelsRand = patientData[
                    patientData.columns[cellLabelIdx]].sample(
                    n=marker2Num, replace=True)
                randTruncDistMat = dist_mat[np.ix_(
                    np.asarray(marker1LabelsRand - 1), np.asarray(
                        marker2LabelsRand - 1))]
                randTruncDistMatBin = np.zeros(
                    randTruncDistMat.shape, dtype='int')
                randTruncDistMatBin[randTruncDistMat < distLim] = 1

                closeNumRand[j, k, r] = np.sum(np.sum(randTruncDistMatBin))

    # z score, pval, and adj pval
    z = np.zeros((markerNum, markerNum))
    muhat = np.zeros((markerNum, markerNum))
    sigmahat = np.zeros((markerNum, markerNum))

    p = np.zeros((markerNum, markerNum, 2))

    for j in range(0, markerNum):
        for k in range(0, markerNum):
            tmp = np.reshape(closeNumRand[j, k, :], (bootstrapNum, 1))
            (muhat[j, k], sigmahat[j, k]) = scipy.stats.norm.fit(tmp)
            z[j, k] = (closeNum[j, k] - muhat[j, k]) / sigmahat[j, k]
            p[j, k, 0] = (1 + (
                np.sum(tmp >= closeNum[j, k]))) / (bootstrapNum + 1)
            p[j, k, 1] = (1 + (
                np.sum(tmp <= closeNum[j, k]))) / (bootstrapNum + 1)

    p_summary = p[:, :, 0]
    for j in range(0, markerNum):
        for k in range(0, markerNum):
            if z[j, k] > 0:
                p_summary[j, k] = p[j, k, 0]
            else:
                p_summary[j, k] = p[j, k, 1]
    (h, adj_p, aS, aB) = statsmodels.stats.multitest.multipletests(
        p_summary, alpha=.05)

    return \
        closeNum, closeNumRand, z, muhat, sigmahat, p, h, adj_p, markerTitles
    # end
