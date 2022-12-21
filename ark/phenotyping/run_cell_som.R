# Assigns cluster labels to cell data using a trained SOM weights matrix

# Usage: Rscript run_cell_som.R {clusterCountsNormPath} {cellWeightsPath}

# - clusterCountsNormPath: path to file with counts of unique cells (rows) by unique pixel SOM/meta clusters (columns), with counts normalized by cell size
# - cellWeightsPath: path to the SOM weights file

suppressPackageStartupMessages({
    library(arrow)
    library(data.table)
    library(FlowSOM)
})

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# get the path to the cluster counts data (norm)
clusterCountsPath <- args[1]

# get the weights write path
cellWeightsPath <- args[2]

# read the cluster counts data (norm)
print("Reading the cluster counts data")
clusterCountsNorm <- as.data.frame(arrow::read_feather(clusterCountsPath))

# read the weights
print("Reading the weights matrix")
somWeights <- as.matrix(arrow::read_feather(cellWeightsPath))

# define the subset of count columns to cluster on
clusterCols <- colnames(clusterCountsNorm)[grepl("pixel_som_cluster_|pixel_meta_cluster_",
                                           colnames(clusterCountsNorm))]

# subset on just the clusterCols for normalization
clusterCountsNormSub <- as.matrix(clusterCountsNorm[,clusterCols])

# 99.9% normalize
# normalize by max (100%) instead of 99.9% if 99.9% = 0
print("Perform 99.9% normalization")
for (clusterCol in clusterCols) {
    normVal <- quantile(clusterCountsNormSub[,clusterCol], 0.999)

    if (normVal == 0) {
        normVal <- quantile(clusterCountsNormSub[,clusterCol], 1)
    }

    clusterCountsNormSub[,clusterCol] <- clusterCountsNormSub[,clusterCol] / as.numeric(normVal)
}

# assign column-normalized values back to main norm dataframe
clusterCountsNorm[,clusterCols] <- clusterCountsNormSub

# map FlowSOM data
print("Mapping cluster labels")
clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(clusterCountsNorm[,clusterCols]))

# assign cluster labels to pixel data
clusterCountsNorm$cell_som_cluster <- as.integer(clusters[,1])

# delete the original cell counts norm path, done to prevent corruption
unlink(clusterCountsPath)

# write to feather
print("Writing clustered data")
arrow::write_feather(as.data.table(clusterCountsNorm), clusterCountsPath, compression='uncompressed')
