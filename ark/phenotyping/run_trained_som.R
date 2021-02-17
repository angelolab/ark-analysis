# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript {fovs} {markers} {pixelMatDir} {pixelWeightsPath} {pixelClusterDir}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - pixelMatDir: path to directory containing the complete pixel data
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

# get path to the weights
pixelWeightsPath <- args[3]

# get the cluster write path directory
pixelClusterDir <- args[4]

# read the weights
print("Reading the weights matrix")
somWeights <- as.matrix(arrow::read_feather(pixelWeightsPath))

# get the marker names from the weights matrix
markers <- colnames(somWeights)

# using trained SOM, batch cluster the original dataset by fov
print("Mapping data to cluster labels")
for (i in 1:length(fovs)) {
    # read in pixel data
    fileName <- paste(fovs[i], ".feather", sep="")
    matPath <- paste(pixelMatDir, fileName, sep="/")
    fovPixelData <- as.matrix(arrow::read_feather(matPath, col_select=all_of(markers)))

    # 99.9% normalize pixel data
    for (marker in markers) {
        marker_quantile <- quantile(fovPixelData[, marker], 0.999)

        # this prevents all-zero columns from getting normalized and becoming NA/Inf
        if (marker_quantile != 0) {
            fovPixelData[, marker] = fovPixelData[, marker] / marker_quantile
        }
    }

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somWeights, fovPixelData)

    # assign cluster labels column to pixel data
    fovPixelData <- as.matrix(cbind(as.matrix(fovPixelData), clusters=clusters[,1]))

    # write to feather
    clusterPath <- paste(pixelClusterDir, fileName, sep="/")
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)

    # print an update every 10 fovs
    if (i %% 10 == 0) {
        sprintf("Finished clustering %s fovs", i)
    }
}

print("Done!")
