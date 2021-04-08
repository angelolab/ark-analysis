# Runs consensus clustering on the pixel data averaged across all channels

# Usage: Rscript {markers} {maxK} {cap} {clusterAvgPath} {consensusClusterPath}

# - markers: list of channel columns to use
# - maxK: number of consensus clusters
# - cap: z-score scaling factor
# - clusterAvgPath: path to the averaged cluster data
# - consensusClusterPath: path to file where the consensus cluster results will be written

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

# if a seed is set, get it and set
seed <- NULL
if (length(args) == 8) {
    seed <- strtoi(args[8])
    set.seed(seed)
}

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
print('Writing consensus clustering results')
for (fov in fovs) {
    # read in pixel data, we'll need the cluster column for mapping
    fileName <- paste(fov, ".feather", sep="")
    matPath <- paste(pixelClusterDir, fileName, sep="/")
    fovPixelData <- arrow::read_feather(matPath)

    # assign hierarchical cluster labels
    fovPixelData$hCluster_cap <- hClust[as.character(fovPixelData$cluster)]

    # overwrite old cluster file with new one containing hCluster_cap
    clusterPath <- paste(pixelMatConsensus, fileName, sep="/")
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)
}
