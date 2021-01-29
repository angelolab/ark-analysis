library(data.table)
library(FlowSOM)
library(rhdf5)

# get the command line arguments
print("Reading command args")
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
print("Creating fovs list")
fovs <- unlist(strsplit(args[1], split=","))
print(fovs)

# create a vector out of the list of channels provided
print("Creating marker list")
markers <- unlist(strsplit(args[2], split=","))
print(markers)

# get path to pixel mat data
print("Getting pixel mat data path")
pixelMatPath <- args[3]

print("Getting the pixel subsetted data path")
pixelSubsetPath <- args[4]

# get the path to the write path
print("Getting the write path")
pixelWritePath <- args[5]

# read the pixel mat data
print("Reading the subsetted pixel matrix data for SOM training")
pixelSubsetData <- fread(pixelSubsetPath, select=markers)

# run the SOM training step
print("Run the SOM training")
somResults <- SOM(data=as.matrix(pixelSubsetData))

# using trained SOM, batch cluster the original dataset by fov
print("Mapping cluster labels")

for (fov in fovs) {
    # read in pixel data
    fovPixelData <- h5read(pixelMatPath, fov)
    fovPixelData <- fovPixelData$table

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somResults$codes,
                                         as.matrix(fovPixelData[,markers]))

    # assign cluster labels column to pixel data
    fovPixelData <- cbind(as.matrix(fovPixelData), clusters=clusters[,1])

    # write to hdf5
    h5write(as.matrix(fovPixelData), pixelWritePath, fov)
}

print("Done!")
