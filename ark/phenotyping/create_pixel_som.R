# Trains a SOM matrix using subsetted pixel data

# Usage: Rscript create_pixel_som.R {fovs} {markers} {xdim} {ydim} {lr_start} {lr_end} {numPasses} {pixelSubsetDir} {normValsPath} {pixelWeightsPath} {seed}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - xdim: number of x nodes to use for SOM
# - ydim: number of y nodes to use for SOM
# - lr_start: the start learning rate
# - lr_end: the end learning rate
# - numPasses: passes to make through dataset for training
# - pixelSubsetDir: path to directory containing the subsetted pixel data
# - normValsPath: path to the 99.9% normalized values file
# - pixelWeightsPath: path to the SOM weights file
# - seed: the random seed to use for training

library(arrow)
library(data.table)
library(FlowSOM)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# create a vector out of the list of channels provided
markers <- unlist(strsplit(args[2], split=","))

# get the number of x nodes to use for the SOM
xdim <- strtoi(args[3])

# get the number of y nodes to use for the SOM
ydim <- strtoi(args[4])

# get the start learning rate
lr_start <- as.double(args[5])

# get the end learning rate
lr_end <- as.double(args[6])

# get the number of passes to make through SOM training
numPasses <- strtoi(args[7])

# get path to subsetted mat directory
pixelSubsetDir <- args[8]

# get the normalized values write path
normValsPath <- args[9]

# get the weights write path
pixelWeightsPath <- args[10]

# set the random seed
seed <- strtoi(args[11])
set.seed(seed)

# read the subsetted pixel mat data for training
print("Reading the subsetted pixel matrix data for SOM training")
pixelSubsetData <- NULL

for (fov in fovs) {
    # subset each matrix with only the markers columns
    fileName <- file.path(fov, "feather", fsep=".")
    subPath <- file.path(pixelSubsetDir, fileName)
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
somResults <- SOM(data=pixelSubsetData, rlen=numPasses,
                  xdim=xdim, ydim=ydim, alpha=c(lr_start, lr_end))

# write the weights to feather
print("Save trained weights")
arrow::write_feather(as.data.table(somResults$codes), pixelWeightsPath)
