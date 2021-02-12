# Trains a SOM matrix using subsetted pixel data

# Usage: Rscript {fovs} {markers} {numPasses} {pixelSubsetDir} {pixelWeightsPath}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - numPasses: passes to make through dataset for training
# - pixelSubsetDir: path to directory containing the subsetted pixel data
# - pixelWeightsPath: path to the SOM weights file

library(arrow)
library(data.table)
library(FlowSOM)
library(rhdf5)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# create a vector out of the list of channels provided
markers <- unlist(strsplit(args[2], split=","))

# get the number of passes to make through SOM training
numPasses <- strtoi(args[3])

# get path to subsetted mat directory
pixelSubsetDir <- args[4]

# get the weights write path
pixelWeightsPath <- args[5]

# read the subsetted pixel mat data for training
print("Reading the subsetted pixel matrix data for SOM training")
pixelSubsetData <- NULL

for (i in 1:length(fovs)) {
    # subset each matrix with only the markers columns
    fileName <- paste(fovs[i], ".feather", sep="")
    subPath <- paste(pixelSubsetDir, fileName, sep="/")
    fovSubsetData <- arrow::read_feather(subPath, col_select=all_of(markers))

    # attach each fov's dataset to pixelSubsetData
    if (is.null(pixelSubsetData)) {
        # pixelSubsetData <- as.matrix(fovSubsetData)
        pixelSubsetData <- as.matrix(fovSubsetData)
    }
    else {
        # pixelSubsetData <- rbind(pixelSubsetData, as.matrix(fovSubsetData))
        pixelSubsetData <- rbind(pixelSubsetData, as.matrix(fovSubsetData))
    }
}

# perform 99.9% normalization on the subsetted data
print("Performing 99.9% normalization")

# TODO: need to one-liner this
for (marker in markers) {
    marker_quantile <- quantile(pixelSubsetData[, marker], 0.999)

    # this prevents all-zero columns from getting normalized and becoming NA/Inf
    if (marker_quantile != 0) {
        pixelSubsetData[, marker] = pixelSubsetData[, marker] / marker_quantile
    }
}

# run the SOM training step
print("Run the SOM training")
somResults <- SOM(data=pixelSubsetData, rlen=numPasses)

# write the weights to HDF5
print("Save trained weights")
arrow::write_feather(as.data.table(somResults$codes), pixelWeightsPath)
