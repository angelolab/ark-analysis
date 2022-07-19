# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript run_pixel_som.R {fovs} {pixelMatDir} {normValsPath} {pixelWeightsPath}

# - fovs: list of fovs to cluster
# - pixelMatDir: path to directory containing the complete pixel data
# - normValsPath: path to the 99.9% normalization values file (created during preprocessing)
# - pixelWeightsPath: path to the SOM weights file

suppressPackageStartupMessages({
    library(arrow)
    library(data.table)
    library(doParallel)
    library(FlowSOM)
    library(foreach)
    library(parallel)
})

# get the number of cores
nCores <- parallel::detectCores() - 1

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

# retrieve the batch size to determine number of threads to run in parallel
batchSize <- strtoi(args[5])

# read the weights
somWeights <- as.matrix(arrow::read_feather(pixelWeightsPath))

# read the normalization values
normVals <- data.table(arrow::read_feather(normValsPath))

# get the marker names from the weights matrix
markers <- colnames(somWeights)

# set order of normVals to make sure they match with weights
normVals <- normVals[,..markers]

# define variable to keep track of number of fovs processed
fovsProcessed <- 0

# using trained SOM, assign SOM labels to each fov
print("Mapping pixel data to SOM cluster labels")
for (batchStart in seq(1, length(fovs), batchSize)) {
    # define the parallel cluster for this batch of fovs
    # NOTE: to prevent the occassional hanging first FOV issue, we need to log to an outfile
    # to "force" a return out of the foreach loop in this case
    parallelCluster <- parallel::makeCluster(nCores, type="FORK", outfile='log.txt')

    # register parallel cluster for dopar
    doParallel::registerDoParallel(cl=parallelCluster)

    # need to prevent overshooting end of fovs list when batching
    batchEnd <- min(batchStart + batchSize - 1, length(fovs))

    # run the multithreaded batch process for mapping to SOM labels and saving
    fovStatuses <- foreach(
        i=batchStart:batchEnd,
        .combine=rbind
    ) %dopar% {
        fileName <- paste0(fovs[i], ".feather")
        matPath <- file.path(pixelMatDir, fileName)

        status <- tryCatch(
            {
                fovPixelData_all <- data.table(arrow::read_feather(matPath))
                fovPixelData <- fovPixelData_all[,..markers]
                fovPixelData <- fovPixelData[,Map(`/`,.SD,normVals)]

                # map FlowSOM data
                clusters <- FlowSOM:::MapDataToCodes(somWeights, as.matrix(fovPixelData))

                # add back other columns
                to_add <- colnames(fovPixelData_all)[!colnames(fovPixelData_all) %in% markers]
                fovPixelData <- cbind(fovPixelData_all[,..to_add], fovPixelData)

                # assign cluster labels column to pixel data
                fovPixelData$pixel_som_cluster <- as.integer(clusters[,1])

                # write data with SOM labels
                tempPath <- file.path(paste0(pixelMatDir, '_temp'), fileName)
                arrow::write_feather(as.data.table(fovPixelData), tempPath, compression='uncompressed')

                # this won't be displayed to the user but is used as a helper to break out
                # in the rare first FOV hang issue
                print(paste('Done writing fov', fovs[i]))
                0
            },
            error=function(cond) {
                # this won't be displayed to the user but is used as a helper to break out
                # in the rare first FOV hang issue
                print(paste('Error encountered for fov', fovs[i]))
                1
            }
        )

        data.frame(fov=fovs[i], status=status)
    }

    # report any erroneous feather files
    for (i in 1:nrow(fovStatuses)) {
        if (fovStatuses[i, 'status'] == 1) {
            print(paste("The data for FOV", fovStatuses[i, 'fov'], "has been corrupted, removing"))
            fovsProcessed <- fovsProcessed - 1
        }
    }

    # unregister the parallel cluster
    parallel::stopCluster(cl=parallelCluster)

    # update number of fovs processed
    fovsProcessed <- fovsProcessed + (batchEnd - batchStart + 1)

    # inform user that batchSize fovs have been processed
    print(paste("Processed", as.character(fovsProcessed), "fovs"))

    # remove log.txt
    unlink('log.txt')
}
