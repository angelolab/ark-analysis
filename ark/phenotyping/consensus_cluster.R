# Runs consensus clustering on the pixel data averaged across all channels

# Usage: Rscript {fovs} {channels} {maxK} {cap} {clusterAvgPath} {consensusClusterPath}

# - fovs: list of fovs to cluster
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

# get path to the averaged channel data
clusterAvgPath <- args[5]

# get consensus clustered write path
consensusClusterPath <- args[6]

# read cluster averaged data
print("Reading cluster averaged data")
clusterAvgs <- arrow::read_feather(clusterAvgPath)

# scale and cap the data accordingly
print("Scaling data")
clusterAvgsScale <- pmin(scale(clusterAvgs[markers]), cap)

# run the consensus clustering
print("Running consensus clustering")
consensusClusterResults <- ConsensusClusterPlus(t(clusterAvgsScale), maxK=maxK)
hClust <- consensusClusterResults[[maxK]]$consensusClass
clusterAvgs$hCluster_cap = hClust[clusterAvgs$cluster]

# saving consensus clustering results
print('Writing consensus clustering results')
arrow::write_feather(as.data.table(clusterAvgs), consensusClusterPath)
