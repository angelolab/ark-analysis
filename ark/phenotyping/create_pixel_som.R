# Trains a SOM (self-organizing map) using subsetted pixel data

# Usage: Rscript create_pixel_som.R {fovs} {markers} {xdim} {ydim} {lr_start} {lr_end} {numPasses} {pixelSubsetDir} {normValsPath} {pixelWeightsPath} {seed}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - xdim: number of x nodes to use for SOM
# - ydim: number of y nodes to use for SOM
# - lr_start: the start learning rate
# - lr_end: the end learning rate
# - numPasses: passes to make through dataset for training
# - pixelSubsetDir: path to directory containing the subsetted pixel data
# - normValsPath: path to the 99.9% normalization values file (created during preprocessing)
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

# get the normalization values path
normValsPath <- args[9]

# get the weights write path
pixelWeightsPath <- args[10]

# set the random seed
seed <- strtoi(args[11])
set.seed(seed)

# get normalization values
normVals <- data.table(arrow::read_feather(normValsPath))
normVals <- normVals[,..markers]

# train the som
print("Reading the subsetted pixel matrix data")
pixelSubsetData <- rbindlist(lapply(fovs, function(x) {
                                          one_tab = data.table(read_feather(
                                          file.path(pixelSubsetDir, paste0(x,".feather"))))
                                          return(one_tab[,..markers])}))

# perform 99.9% normalization on the subsetted data
print("Performing 99.9% normalization")
pixelSubsetData <- pixelSubsetData[,Map(`/`,.SD,normVals)]

# run the SOM training step
print("Training the SOM")
somResults <- SOM(data=as.matrix(pixelSubsetData), rlen=numPasses,
                  xdim=xdim, ydim=ydim, alpha=c(lr_start, lr_end), map=FALSE)

# write the weights to feather
print("Save trained weights")
arrow::write_feather(as.data.table(somResults$codes), pixelWeightsPath, compression='uncompressed')
