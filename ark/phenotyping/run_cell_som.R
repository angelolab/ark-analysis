# Assigns cluster labels to cell data using a trained SOM weights matrix

# Usage: Rscript run_cell_som.R {clusterCountsPath} {clusterCountsNormPath} {cellWeightsPath} {cellClusterPath}

# - clusterCountsPath: path to file with counts of unique cells (rows) by unique SOM pixel/meta clusters (columns)
# - clusterCountsNormPath: same as clusterCountsPath, but with counts normalized by cell size
# - cellWeightsPath: path to the SOM weights file
# - cellClusterPath: path to file where the cell SOM labeled data will be written to

library(arrow)
library(data.table)
library(FlowSOM)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# get the path to the cluster counts data
clusterCountsPath <- args[1]

# get the path to the cluster counts norm data
clusterCountsNormPath <- args[2]

# get the weights write path
cellWeightsPath <- args[3]

# get the cluster write path
cellClusterPath <- args[4]

# read the cluster counts data (norm and un-norm)
print("Reading the cluster counts data")
clusterCountsData <- arrow::read_feather(clusterCountsPath)
clusterCountsNorm <- arrow::read_feather(clusterCountsNormPath)

# read the weights
print("Reading the weights matrix")
somWeights <- as.matrix(arrow::read_feather(cellWeightsPath))

clusterCols <- colnames(clusterCountsNorm)[grepl("cluster_|hCluster_cap_",
                                           colnames(clusterCountsNorm))]

# keep just the cluster columns
clusterCountsNorm <- as.matrix(clusterCountsNorm[,clusterCols])

# 99.9% normalize
print("Perform 99.9% normalization")
for (clusterCol in clusterCols) {
    normVal <- quantile(clusterCountsNorm[,clusterCol], 0.999)

    if (normVal == 0) {
        normVal <- quantile(clusterCountsNorm[,clusterCol], 1)
    }

    clusterCountsNorm[,clusterCol] <- clusterCountsNorm[,clusterCol] / normVal
}

# map FlowSOM data
print("Mapping cluster labels")
clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(clusterCountsNorm))

# assign cluster labels to pixel data
clusterCountsData$cluster <- as.integer(clusters[,1])

# write to feather
print("Writing clustered data")
arrow::write_feather(as.data.table(clusterCountsData), cellClusterPath)
