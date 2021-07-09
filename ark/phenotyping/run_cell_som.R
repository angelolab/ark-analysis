# Assigns cluster labels to cell data using a trained SOM weights matrix

# Usage: Rscript run_cell_som.R {clusterCountsPath} {cellWeightsPath} {cellClusterPath}

# - clusterCountsPath: path to file with counts of unique cells (rows) by unique SOM pixel/meta clusters (columns)
# - cellWeightsPath: path to the SOM weights file
# - cellClusterPath: path to file where the cell SOM labeled data will be written to

library(arrow)
library(data.table)
library(FlowSOM)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# get the path to the cluster counts data
clusterCountsPath <- args[1]

# get the weights write path
cellWeightsPath <- args[2]

# get the cluster write path
cellClusterPath <- args[3]

# read the cluster counts data
print("Reading the cluster counts data")
clusterCountsData <- arrow::read_feather(clusterCountsPath)

# read the weights
print("Reading the weights matrix")
somWeights <- as.matrix(arrow::read_feather(cellWeightsPath))

# get the column names of the SOM pixel/meta clusters
clusterCols <- colnames(clusterCountsData)[grepl("cluster_|hCluster_cap_",
                                           colnames(clusterCountsData))]

# keep just the cluster columns
clusterCountsNorm <- clusterCountsData[, clusterCols]

# 99.9% normalize
print("Perform 99.9% normalization")
for (clusterCol in clusterCols) {
    normVal <- quantile(clusterCountsNorm[,clusterCol])

    # prevent normalizing by 0
    if (normVal != 0) {
        clusterCountsNorm[,clusterCol] <- clusterCountsNorm[,clusterCol] / normVal
    }
}

# map FlowSOM data
print("Mapping cluster labels")
clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(clusterCountsNorm))

# assign cluster labels to pixel data
clusterCountsData$cluster <- clusters[,1]

# write to feather
print("Writing clustered data")
arrow::write_feather(as.data.table(clusterCountsData), cellClusterPath)
