# Trains a SOM matrix using subsetted pixel data

# Usage: Rscript {fovs} {markers} {numPasses} {pixelSubsetDir} {pixelWeightsPath}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - numPasses: passes to make through dataset for training
# - pixelSubsetDir: path to directory containing the subsetted pixel data
# - normValsPath: path to the 99.9% normalized values file
# - pixelWeightsPath: path to the SOM weights file

library(arrow)
library(data.table)
library(FlowSOM)

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

# get the normalized values write path
normValsPath <- args[5]

# get the weights write path
pixelWeightsPath <- args[6]

# if a seed is set, get it and set
if (length(args) == 7) {
    seed <- strtoi(args[7])
    set.seed(seed)
}

# read the subsetted pixel mat data for training
print("Reading the subsetted pixel matrix data for SOM training")
pixelSubsetData <- NULL

for (fov in fovs) {
    # subset each matrix with only the markers columns
    fileName <- paste(fov, ".feather", sep="")
    subPath <- paste(pixelSubsetDir, fileName, sep="/")
    fovSubsetData <- arrow::read_feather(subPath, col_select=all_of(markers))

    # attach each fov's dataset to pixelSubsetData
    if (is.null(pixelSubsetData)) {
        pixelSubsetData <- as.matrix(fovSubsetData)
    }
    else {
        pixelSubsetData <- rbind(pixelSubsetData, as.matrix(fovSubsetData))
    }
}

# perform 99.9% normalization on the subsetted data
normVals <- data.frame(matrix(NA, nrow=1, ncol=length(markers)))
colnames(normVals) <- markers

print("Performing 99.9% normalization")

for (marker in markers) {
    marker_quantile <- quantile(pixelSubsetData[, marker], 0.999)

    # this prevents all-zero columns from getting normalized and becoming NA/Inf
    if (marker_quantile != 0) {
        pixelSubsetData[, marker] = pixelSubsetData[, marker] / marker_quantile
    }

    normVals[marker] = marker_quantile
}

# write 99.9% normalized values to feather
print("Save 99.9% normalized values for each marker")
arrow::write_feather(as.data.table(normVals), normValsPath)

# run the SOM training step
print("Run the SOM training")
somResults <- SOM(data=pixelSubsetData, rlen=numPasses, alpha=c(0.05, 0.01))

# write the weights to HDF5
print("Save trained weights")
arrow::write_feather(as.data.table(somResults$codes), pixelWeightsPath)
