library(data.table)
library(FlowSOM)
library(rhdf5)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# create a vector out of the list of channels provided
markers <- unlist(strsplit(args[2], split=","))

# get path to pixel mat data
pixelMatPath <- args[3]

# get path to the weights
pixelWeightsPath <- args[4]

# get the cluster write path
pixelClusterPath <- args[5]

# read teh weights
somWeights <- h5read(pixelWeightsPath, 'weights')

# using trained SOM, batch cluster the original dataset by fov
print("Mapping cluster labels")
for (i in 1:length(fovs)) {
    # read in pixel data
    fovPixelData <- h5read(pixelMatPath, fovs[i])
    fovPixelData <- fovPixelData$table

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somWeights,
                                         as.matrix(fovPixelData[,markers]))

    # assign cluster labels column to pixel data
    fovPixelData <- cbind(as.matrix(fovPixelData), clusters=clusters[,1])

    # write to hdf5
    h5write(as.matrix(fovPixelData), pixelClusterPath, fovs[i])

    # print an update every 10 fovs
    if (i %% 10 == 0) {
        sprintf("Finished clustering %s fovs", i)
    }
}

print("Done!")
