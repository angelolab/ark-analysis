# Runs consensus clustering on the pixel data averaged across all channels

# Usage: Rscript {fovs} {markers} {maxK} {cap} {pixelClusterDir} {clusterAvgPath} {pixelMatConsensus} {clustToMeta} {seed}

# - fovs: list of fovs to cluster
# - markers: list of channel columns to use
# - maxK: number of consensus clusters
# - cap: max z-score cutoff
# - pixelClusterDir: path to the pixel data with SOM clusters
# - clusterAvgPath: path to the averaged cluster data
# - pixelMatConsensus: path to file where the consensus cluster results will be written
# - clustToMeta: path to file where the SOM cluster to meta cluster mapping will be written
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

# get the clust to meta write path
clustToMeta <- args[8]

# set the random seed
seed <- strtoi(args[9])
set.seed(seed)

# read cluster averaged data
print("Reading cluster averaged data")
clusterAvgs <- as.data.frame(read.csv(clusterAvgPath, check.names=FALSE))

# scale and cap the data accordingly
# NOTE: z-scoring (done in Python) and capping cluster avg data produces better clustering results
# NOTE: need to cap with sapply because pmin sets out-of-range values to NA on non-vectors
clusterAvgsScale <- clusterAvgs[,markers]
clusterAvgsScale <- sapply(clusterAvgsScale, pmin, cap)

# run the consensus clustering
# TODO: look into suppressing output for Rs (invisible), not urgent
print("Running consensus clustering")
consensusClusterResults <- ConsensusClusterPlus(t(clusterAvgsScale), maxK=maxK, seed=seed)
som_to_meta_map <- consensusClusterResults[[maxK]]$consensusClass
names(som_to_meta_map) <- clusterAvgs$pixel_som_cluster

# append hClust to each fov's data
print("Writing consensus clustering results")
for (i in 1:length(fovs)) {
    # read in pixel data, we'll need the cluster column for mapping
    fileName <- file.path(fovs[i], "feather", fsep=".")
    matPath <- file.path(pixelClusterDir, fileName)
    fovPixelData <- arrow::read_feather(matPath)

    # assign hierarchical cluster labels
    fovPixelData$pixel_meta_cluster <- som_to_meta_map[as.character(fovPixelData$pixel_som_cluster)]

    # write consensus clustered data
    clusterPath <- file.path(pixelMatConsensus, fileName)
    arrow::write_feather(as.data.table(fovPixelData), clusterPath)

    # print an update every 10 fovs
    if (i %% 10 == 0) {
        print("# fovs clustered:")
        print(i)
    }
}

# save the mapping from cluster to som_to_meta_map
print("Writing SOM to meta cluster mapping table")
som_to_meta_map <- as.data.table(som_to_meta_map)

# assign pixel_som_cluster column, then rename som_to_meta_map to pixel_meta_cluster
som_to_meta_map$pixel_som_cluster <- as.integer(rownames(som_to_meta_map))
som_to_meta_map <- setnames(som_to_meta_map, "som_to_meta_map", "pixel_meta_cluster")
arrow::write_feather(som_to_meta_map, clustToMeta)
