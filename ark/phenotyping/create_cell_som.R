# Trains a SOM matrix using cluster counts per cell

# Usage: Rscript create_cell_som.R {fovs} {xdim} {ydim} {lr_start} {lr_end} {numPasses} {clusterCountsPath} {cellWeightsPath} {seed}

# - fovs: list of fovs to cluster
# - xdim: number of x nodes to use for SOM
# - ydim: number of y nodes to use for SOM
# - lr_start: the start learning rate
# - lr_end: the end learning rate
# - numPasses: passes to make through dataset for training
# - clusterCountsPath: path to file with counts of unique cells (rows) by unique SOM pixel/meta clusters (columns)
# - cellWeightsPath: path to the SOM weights file
# - seed: the random seed to use for training

library(arrow)
library(data.table)
library(FlowSOM)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# get the number of x nodes to use for the SOM
xdim <- strtoi(args[2])

# get the number of y nodes to use for the SOM
ydim <- strtoi(args[3])

# get the start learning rate
lr_start <- as.double(args[4])

# get the end learning rate
lr_end <- as.double(args[5])

# get the number of passes to make through SOM training
numPasses <- strtoi(args[6])

# get the path to the cluster counts data
clusterCountsPath <- args[7]

# get the weights write path
cellWeightsPath <- args[8]

# set the random seed
seed <- strtoi(args[9])
set.seed(seed)

# read the cluster counts data
print("Reading the cluster counts data for SOM training")
clusterCountsData <- as.data.frame(arrow::read_feather(clusterCountsPath))

# get the column names of the SOM pixel/meta clusters
clusterCols <- colnames(clusterCountsData)[grepl(pattern="cluster_|hCluster_cap_",
                                           colnames(clusterCountsData))]

# normalize the rows by their cell size
print("Normalizing each cell's cluster counts by cell size")
clusterCountsNorm <- as.matrix(clusterCountsData[,clusterCols] / clusterCountsData$cell_size)

# 99.9% normalize
print("Perform 99.9% normalization")
for (clusterCol in clusterCols) {
    normVal <- quantile(clusterCountsNorm[,clusterCol])

    # prevent normalizing by 0
    if (normVal != 0) {
        clusterCountsNorm[,clusterCol] <- clusterCountsNorm[,clusterCol] / normVal
    }
}

# create the cell SOM
print("Run the SOM training")
somResults <- SOM(data=clusterCountsNorm, xdim=xdim, ydim=ydim,
                  rlen=numPasses, alpha=c(lr_start, lr_end))

# write the weights to feather
print("Save trained weights")
arrow::write_feather(as.data.table(somResults$codes), cellWeightsPath)
