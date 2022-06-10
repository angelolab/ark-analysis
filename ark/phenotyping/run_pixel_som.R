# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript run_pixel_som.R {fovs} {pixelMatDir} {normValsPath} {pixelWeightsPath}

# - fovs: list of fovs to cluster
# - pixelMatDir: path to directory containing the complete pixel data
# - normValsPath: path to the 99.9% normalization values file (created during preprocessing)
# - pixelWeightsPath: path to the SOM weights file

library(arrow)
library(data.table)
library(doParallel)
library(FlowSOM)
library(foreach)
library(parallel)

# helper function to map a FOV to its SOM labels
mapSOMLabels <- function(fov, somWeights, pixelMatDir) {
    fileName <- paste0(fov, ".feather")
    matPath <- file.path(pixelMatDir, fileName)
    fovPixelData_all <- data.table(arrow::read_feather(matPath))

    # 99.9% normalization
    fovPixelData <- fovPixelData_all[,..markers]
    fovPixelData <- fovPixelData[,Map(`/`,.SD,normVals)]

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(fovPixelData))

    # add back other columns
    to_add <- colnames(fovPixelData_all)[!colnames(fovPixelData_all) %in% markers]
    fovPixelData <- cbind(fovPixelData_all[,..to_add],fovPixelData)

    # assign cluster labels column to pixel data
    fovPixelData$pixel_som_cluster <- as.integer(clusters[,1])

    # write to feather
    arrow::write_feather(as.data.table(fovPixelData),  matPath)

    # inform user that a fov has been processed
    print(paste("Processed fov:", fov))
}

# get the number of cores
nCores <- parallel::detectCores() - 1

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# get path to pixel mat directory
pixelMatDir <- args[2]

# get path to the 99.9% normalized values
normValsPath <- args[3]

# get path to the weights
pixelWeightsPath <- args[4]

# TODO: set batch size to be customizable by user with default arg
# batchSize <- args[6]
batchSize <- 5

# read the weights
somWeights <- as.matrix(arrow::read_feather(pixelWeightsPath))

# read the normalization values
normVals <- data.table(arrow::read_feather(normValsPath))

# get the marker names from the weights matrix
markers <- colnames(somWeights)

# set order of normVals to make sure they match with weights
normVals <- normVals[,..markers]

# using trained SOM, assign SOM labels to each fov
print("Mapping pixel data to SOM cluster labels")
for (batchStart in seq(1, length(fovs), batchSize)) {
    # define the parallel cluster for this batch of fovs
    parallelCluster <- parallel::makeCluster(nCores, type="FORK")

    # register parallel cluster for dopar
    doParallel::registerDoParallel(cl=parallelCluster)

    # need to prevent overshooting end of fovs list when batching
    batchEnd <- min(batchStart + batchSize - 1, length(fovs))

    # run the multithreaded batch process for mapping to SOM labels and saving
    foreach(
        i=batchStart:batchEnd,
        .combine='c'
    ) %dopar% {
        mapSOMLabels(fovs[i], somWeights, pixelMatDir)
    }

    # unregister the parallel cluster
    parallel::stopCluster(cl=parallelCluster)
}

# for (i in 1:length(fovs)) {
#     # read in pixel data
#     fileName <- paste0(fovs[i], ".feather")
#     matPath <- file.path(pixelMatDir, fileName)
#     fovPixelData_all <- data.table(arrow::read_feather(matPath))

#     # 99.9% normalization
#     fovPixelData <- fovPixelData_all[,..markers]
#     fovPixelData <- fovPixelData[,Map(`/`,.SD,normVals)]

#     # map FlowSOM data
#     clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(fovPixelData))

#     # add back other columns
#     to_add <- colnames(fovPixelData_all)[!colnames(fovPixelData_all) %in% markers]
#     fovPixelData <- cbind(fovPixelData_all[,..to_add],fovPixelData)

#     # assign cluster labels column to pixel data
#     fovPixelData$pixel_som_cluster <- as.integer(clusters[,1])

#     # write to feather
#     clusterPath <- file.path(pixelClusterDir, fileName)
#     arrow::write_feather(as.data.table(fovPixelData), clusterPath)

#     # print an update every 10 fovs
#     # TODO: find a way to capture sprintf to the console
#     if (i %% 10 == 0) {
#         print("# fovs clustered:")
#         print(i)
#     }
# }

print("Done!")
