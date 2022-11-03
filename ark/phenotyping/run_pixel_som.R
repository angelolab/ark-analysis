# Assigns cluster labels to pixel data using a trained SOM weights matrix

# Usage: Rscript run_pixel_som.R {fovs} {pixelMatDir} {normValsPath} {pixelWeightsPath} {multiprocess} {batchSize} {nCores}

# - fovs: list of fovs to cluster
# - pixelMatDir: path to directory containing the complete pixel data
# - normValsPath: path to the 99.9% normalization values file (created during preprocessing)
# - pixelWeightsPath: path to the SOM weights file
# - multiprocess: whether to run cluster assignment using multiprocessing
# - batchSize: if multiprocess is TRUE, the number of FOVs to process at once
# - nCores: if multiprocess is TRUE, the number of Docker CPUs to use

suppressPackageStartupMessages({
    library(arrow)
    library(data.table)
    library(doParallel)
    library(FlowSOM)
    library(foreach)
    library(parallel)
})

# helper function for assigning SOM cluster labels
assignSOMLabels <- function(fov, pixelMatDir, fileName, matPath, markers, normVals, somWeights) {
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

            # write data with SOM cluster labels
            tempPath <- file.path(paste0(pixelMatDir, '_temp'), fileName)
            arrow::write_feather(as.data.table(fovPixelData), tempPath, compression='uncompressed')

            # this won't be displayed to the user but is used as a helper to break out
            # in the rare first FOV hang issue
            # print(paste('Done writing fov', fov))
            c(0, '')
        },
        error=function(cond) {
            # this won't be displayed to the user but is used as a helper to break out
            # in the rare first FOV hang issue
            # print(paste('Error encountered for fov', fov))
            c(1, cond)
        }
    )

    return(status)
}

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

# define if multiprocessing should be turned on
multiprocess <- args[5]

# retrieve the batch size to determine number of threads to run in parallel
batchSize <- strtoi(args[6])

# get the number of cores
nCores <- strtoi(args[7])

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

# using trained SOM, assign SOM cluster labels to each fov
print("Mapping pixel data to SOM cluster labels")

# handle multiprocess passed in as a string argument
if (multiprocess == "True") {
    print("Running multiprocessing")
    for (batchStart in seq(1, length(fovs), batchSize)) {
        # define the parallel cluster for this batch of fovs
        # NOTE: to prevent the occassional hanging first FOV issue, we need to log to an outfile
        # to "force" a return out of the foreach loop in this case
        parallelStatus <- tryCatch(
            {
                parallelCluster <- parallel::makeCluster(nCores, type="FORK", outfile='log.txt')
                0
            },
            error=function(cond) {
                1
            }
        )

        if (parallelStatus == 1) {
            print(paste0("Too many cores (", nCores, ") specified, reduce this using the ncores parameter"))
            quit(status=1)
        }

        # register parallel cluster for dopar
        doParallel::registerDoParallel(cl=parallelCluster)

        # need to prevent overshooting end of fovs list when batching
        batchEnd <- min(batchStart + batchSize - 1, length(fovs))

        # run the multithreaded batch process for mapping to SOM cluster labels and saving
        fovStatuses <- foreach(
            i=batchStart:batchEnd,
            .combine=rbind
        ) %dopar% {
            fileName <- paste0(fovs[i], ".feather")
            matPath <- file.path(pixelMatDir, fileName)

            # generate the SOM cluster labels
            status <- assignSOMLabels(fovs[i], pixelMatDir, fileName, matPath, markers, normVals, somWeights)

            # append to data frame status checker
            data.frame(fov=fovs[i], status=status[1], errCond=status[2])
        }

        # report any erroneous feather files
        for (i in 1:nrow(fovStatuses)) {
            if (fovStatuses[i, 'status'] == 1) {
                print(paste("Processing for FOV", fovStatuses[i, 'fov'], "failed, removing from pipeline. Error message:"))
                print(fovStatuses[i, 'errCond'])
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
} else {
    for (i in 1:length(fovs)) {
        fileName <- paste0(fovs[i], ".feather")
        matPath <- file.path(pixelMatDir, fileName)

        # generate the SOM cluster labels
        status <- assignSOMLabels(fovs[i], pixelMatDir, fileName, matPath, markers, normVals, somWeights)

        # report any erroneous feather files
        if (status[1] == 1) {
            print(paste("Processing for FOV", fovs[i], "failed, removing from pipeline. Error message:"))
            print(status[2])
            fovsProcessed <- fovsProcessed - 1
        }

        # update number of fovs processed
        fovsProcessed <- fovsProcessed + 1

        # inform user every 10 fovs that get processed
        if (fovsProcessed %% 10 == 0 | fovsProcessed == length(fovs)) {
            print(paste("Processed", as.character(fovsProcessed), "fovs"))
        }
    }
}
