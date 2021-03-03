# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript {fovs} {markers} {pixelMatDir} {pixelWeightsPath} {pixelClusterDir} {batchSize}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - pixelMatDir: path to directory containing the complete pixel data
# - normValsPath: path to the 99.9% normalized values file
# - pixelWeightsPath: path to the SOM weights file
# - pixelClusterDir: path to directory where the clustered data will be written to
# - batchSize: number of fovs to cluster at once

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

# get the batch size
batchSize <- strtoi(args[6])

# read the weights
somWeights <- as.matrix(arrow::read_feather(pixelWeightsPath))

# read the normalization values
normVals <- as.matrix(arrow::read_feather(normValsPath))

# get the marker names from the weights matrix
markers <- colnames(somWeights)

# divide the fovs into batches
fovBatches <- split(fovs, cut(seq_along(fovs), length(fovs) / batchSize, labels=FALSE))

print("Mapping data to cluster labels")
start <- proc.time()
batchNum <- 1
for (fovs in fovBatches) {
    # create pixel data to cluster
    batchPixelData <- matrix(nrow=0, ncol=length(markers))
    colnames(batchPixelData) <- markers

    batchFileName <- sprintf("cluster_%s.feather", batchNum)

    for (fov in fovs) {
        fovFileName <- paste(fov, ".feather", sep="")
        matPath <- paste(pixelMatDir, fovFileName, sep="/")
        fovPixelData <- as.matrix(arrow::read_feather(matPath, col_select=all_of(markers)))

        batchPixelData <- rbind(batchPixelData, fovPixelData)
    }

    # 99.9% normalize pixel data
    for (marker in markers) {
        # this prevents all- or mostly-zero columns from getting normalized and becoming NA/Inf
        if (normVals[1, marker] != 0) {
            batchPixelData[, marker] = batchPixelData[, marker] / normVals[1, marker]
        }
    }

    print(batchPixelData)

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somWeights, batchPixelData)

    # assign cluster labels column to pixel data
    batchPixelData <- as.matrix(cbind(as.matrix(batchPixelData), cluster=clusters[,1]))

    # write to feather
    clusterPath <- paste(pixelClusterDir, batchFileName, sep="/")
    arrow::write_feather(as.data.table(batchPixelData), clusterPath)

    batchNum <- batchNum + 1

    # print an update every 10 fovs
    if (batchNum %% 10 == 0) {
        sprintf("Finished clustering %s fovs", i)
    }
}
print(proc.time() - start)

# using trained SOM, batch cluster the original dataset by fov
print("Mapping data to cluster labels")
start <- proc.time()
for (i in 1:length(fovs)) {
    # read in pixel data
    fileName <- paste(fovs[i], ".feather", sep="")
    matPath <- paste(pixelMatDir, fileName, sep="/")
    fovPixelData <- as.matrix(arrow::read_feather(matPath, col_select=all_of(markers)))

    # 99.9% normalize pixel data
    for (marker in markers) {
        # this prevents all- or mostly-zero columns from getting normalized and becoming NA/Inf
        if (normVals[1, marker] != 0) {
            fovPixelData[, marker] = fovPixelData[, marker] / normVals[1, marker]
        }
    }

    # map FlowSOM data
    clusters <- FlowSOM:::MapDataToCodes(somWeights, fovPixelData)

    # assign cluster labels column to pixel data
    fovPixelData <- as.matrix(cbind(as.matrix(fovPixelData), cluster=clusters[,1]))

    # write to feather
    clusterPath <- paste(pixelClusterDir, fileName, sep="/")
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)

    # print an update every 10 fovs
    if (i %% 10 == 0) {
        sprintf("Finished clustering %s fovs", i)
    }
}
print(proc.time() - start)

print("Done!")
