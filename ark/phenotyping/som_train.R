library(data.table)
library(FlowSOM)
library(rhdf5)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# create a vector out of the list of channels provided
markers <- unlist(strsplit(args[2], split=","))

# get path to subsetted mat data
pixelSubsetPath <- args[3]

# get the weights write path
pixelWeightsPath <- args[4]

# read the subsetted pixel mat data for training
print("Reading the subsetted pixel matrix data for SOM training")
pixelSubsetData <- NULL

for (i in 1:length(fovs)) {
    # subset each matrix with only the markers columns
    fovSubsetData <- h5read(pixelSubsetPath, fovs[i])
    fovSubsetData <- fovSubsetData$table[,markers]

    # attach each fov's dataset to pixelSubsetData
    if (is.null(pixelSubsetData)) {
        pixelSubsetData <- as.matrix(fovSubsetData)
    }
    else {
        pixelSubsetData <- rbind(pixelSubsetData, as.matrix(fovSubsetData))
    }
}

# run the SOM training step
print("Run the SOM training")
somResults <- SOM(data=pixelSubsetData)

# write the weights to HDF5
print("Save trained weights")
h5write(somResults$codes, pixelWeightsPath, 'weights')
