# Runs consensus clustering on the pixel data averaged across all channels

# Usage: Rscript {markers} {maxK} {cap} {pixelClusterDir} {clusterAvgPath} {pixelMatConsensus} {seed}

# - markers: list of channel columns to use
# - maxK: number of consensus clusters
# - cap: max z-score cutoff
# - pixelClusterDir: path to the pixel data with SOM clusters
# - clusterAvgPath: path to the averaged cluster data
# - pixelMatConsensus: path to file where the consensus cluster results will be written
# - seed: random factor

library(arrow)
library(data.table)
library(ConsensusClusterPlus)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# create the list of fovs
fovs <- unlist(strsplit(args[1], split=","))

# create the list of channels
markers <- unlist(strsplit(args[2], split=","))

# get number of consensus clusters
maxK <- strtoi(args[3])

# get z-score scaling factor
cap <- strtoi(args[4])

# get path to the clustered pixel data
pixelClusterDir <- args[5]

# get path to the averaged channel data
clusterAvgPath <- args[6]

# get consensus clustered write path
pixelMatConsensus <- args[7]

# set the random seed
seed <- strtoi(args[8])
set.seed(seed)

# read cluster averaged data
print("Reading cluster averaged data")
clusterAvgs <- arrow::read_feather(clusterAvgPath)

# scale and cap the data accordingly
print("Scaling data")
clusterAvgsScale <- pmin(scale(clusterAvgs[markers]), cap)

# run the consensus clustering
print("Running consensus clustering")
consensusClusterResults <- ConsensusClusterPlus(t(clusterAvgsScale), maxK=maxK, seed=seed)
hClust <- consensusClusterResults[[maxK]]$consensusClass
names(hClust) <- clusterAvgs$cluster

# append hClust to each fov's data
print("Writing consensus clustering results")
for (i in 1:length(fovs)) {
    # read in pixel data, we'll need the cluster column for mapping
    fileName <- file.path(fovs[i], "feather", fsep=".")
    matPath <- file.path(pixelClusterDir, fileName)
    fovPixelData <- arrow::read_feather(matPath)

    # assign hierarchical cluster labels
    fovPixelData$hCluster_cap <- hClust[as.character(fovPixelData$cluster)]

    # write consensus clustered data
    clusterPath <- file.path(pixelMatConsensus, fileName)
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)

    # print an update every 10 fovs
    if (i %% 10 == 0) {
        print("# fovs clustered:")
        print(i)
    }
}
