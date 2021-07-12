# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript run_pixel_som.R {fovs} {markers} {pixelMatDir} {normValsPath} {pixelWeightsPath} {pixelClusterDir}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - pixelMatDir: path to directory containing the complete pixel data
# - normValsPath: path to the 99.9% normalized values file
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
normVals <- as.matrix(arrow::read_feather(normValsPath))

# get the marker names from the weights matrix
markers <- colnames(somWeights)

# using trained SOM, batch cluster the original dataset by fov
print("Mapping data to cluster labels")
for (i in 1:length(fovs)) {
    # read in pixel data
    fileName <- file.path(fovs[i], "feather", fsep=".")
    matPath <- file.path(pixelMatDir, fileName)
    fovPixelData <- arrow::read_feather(matPath)

    # 99.9% normalize pixel data
    for (marker in markers) {
        # this prevents all- or mostly-zero columns from getting normalized and becoming NA/Inf
        if (normVals[1, marker] != 0) {
            fovPixelData[, marker] <- fovPixelData[, marker] / normVals[1, marker]
        }
    }

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(fovPixelData[,markers]))

    # assign cluster labels column to pixel data
    fovPixelData$cluster <- clusters[,1]

    # write to feather
    clusterPath <- file.path(pixelClusterDir, fileName)
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)

    # print an update every 10 fovs
    if (i %% 10 == 0) {
        print("# fovs clustered:")
        print(i)
    }
}

print("Done!")
