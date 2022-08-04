# Trains a SOM matrix using cluster counts per cell

# Usage: Rscript create_cell_som.R {fovs} {xdim} {ydim} {lr_start} {lr_end} {numPasses} {clusterCountsNormPath} {cellWeightsPath} {seed}

# - fovs: list of fovs to cluster
# - xdim: number of x nodes to use for SOM
# - ydim: number of y nodes to use for SOM
# - lr_start: the start learning rate
# - lr_end: the end learning rate
# - numPasses: passes to make through dataset for training
# - clusterCountsNormPath: path to file with counts of unique cells (rows) by unique SOM pixel/meta clusters (columns), normalized by cell size
# - cellWeightsPath: path to the SOM weights file
# - seed: the random seed to use for training

suppressPackageStartupMessages({
    library(arrow)
    library(data.table)
    library(FlowSOM)
})

# a helper function for computing 99.9%
percentile_99_9_helper <- function(x) {
    if (quantile(as.numeric(x[x > 0]), 0.999) == 0) {
        return(quantile(as.numeric(x), 1))
    }

    return(quantile(as.numeric(x[x > 0]), 0.999))
}

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

# get the path to the cluster counts norm data
clusterCountsNormPath <- args[7]

# get the weights write path
cellWeightsPath <- args[8]

# set the random seed
seed <- strtoi(args[9])
set.seed(seed)

# read the cluster counts data
print("Reading the cluster counts data for SOM training")
clusterCountsNorm <- as.data.frame(arrow::read_feather(clusterCountsNormPath))

# get the column names of the SOM pixel/meta clusters
clusterCols <- colnames(clusterCountsNorm)[grepl(pattern="pixel_som_cluster_|pixel_meta_cluster_",
                                           colnames(clusterCountsNorm))]

# keep just the cluster columns
clusterCountsNormSub <- clusterCountsNorm[,clusterCols]

# get the 99.9% normalized values
clusterCountsNormVals <- sapply(clusterCountsNormSub, percentile_99_9_helper)

# 99.9% normalize the values
clusterCountsNormSub <- as.matrix(sweep(clusterCountsNormSub, 2, clusterCountsNormVals, '/'))

# create the cell SOM
print("Run the SOM training")
somResults <- SOM(data=as.matrix(clusterCountsNormSub), xdim=xdim, ydim=ydim,
                  rlen=numPasses, alpha=c(lr_start, lr_end), map=FALSE)

# write the weights to feather
print("Save trained weights")
arrow::write_feather(as.data.table(somResults$codes), cellWeightsPath, compression='uncompressed')
