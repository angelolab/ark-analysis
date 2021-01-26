library(data.table)
library(FlowSOM)
library(rhdf5)

print("Reading command args")
args <- commandArgs(trailingOnly=TRUE)

# get path to pixel mat data
print("Getting pixel mat data path")
pixelMatPath <- args[1]

# create a vector out of the list of channels provided
print("Creating marker list")
markers <- unlist(strsplit(args[2], split=","))

# get the path to the write directory
print("Getting the write path")
writePath <- args[3]

# read the pixel mat data
print("Reading the pixel matrix data")
pixelMatData <- fread(pixelMatPath, select=markers)

# TODO: subset the data for SOM clustering only on the channels we're interested in

# run the SOM clustering step
print("Run the SOM clustering")
somResults <- SOM(data=as.matrix(pixelMatData))

# add original cluster labels to pixelMatData and save, used for verification
print("Mapping original SOM data")
clusters <- FlowSOM:::MapDataToCodes(somResults$codes, as.matrix(pixelMatData))
pixelMatData <- cbind(pixelMatData, clusters=clusters)
fwrite(as.matrix(pixelMatData), writePath)
