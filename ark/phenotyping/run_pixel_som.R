# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript run_pixel_som.R {fovs} {pixelMatDir} {normValsPath} {pixelWeightsPath} {pixelClusterDir}

# - fovs: list of fovs to cluster
# - pixelMatDir: path to directory containing the complete pixel data
# - normValsPath: path to the 99.9% normalization values file (created during preprocessing)
# - pixelWeightsPath: path to the SOM weights file
# - pixelClusterDir: path to directory where the clustered data will be written to

library(arrow)
library(data.table)
library(FlowSOM)

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

# get the cluster write path directory
pixelClusterDir <- args[5]

# read the weights
somWeights <- as.matrix(arrow::read_feather(pixelWeightsPath))

# read the normalization values
normVals <- data.table(arrow::read_feather(normValsPath))

# get the marker names from the weights matrix
markers <- colnames(somWeights)

# set order of normVals to make sure they match with weights
normVals <- normVals[,..markers]

# using trained SOM, batch cluster the original dataset by fov
print("Mapping data to cluster labels")
for (i in 1:length(fovs)) {
    # read in pixel data
    fileName <- paste0(fovs[i], ".feather")
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
    clusterPath <- file.path(pixelClusterDir, fileName)
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)

    # print an update every 10 fovs
    # TODO: find a way to capture sprintf to the console
    if (i %% 10 == 0) {
        print("# fovs clustered:")
        print(i)
    }
}

print("Done!")
