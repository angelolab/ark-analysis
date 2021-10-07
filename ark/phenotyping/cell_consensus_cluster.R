# Runs consensus clustering on the averaged cell data table (created by compute_cell_cluster_avg) 
# defined as the mean counts of each SOM pixel/meta cluster across all cell SOM clusters in each fov
# (m x n table, where m is the number of cell SOM/meta clusters and n is the number of pixel SOM/meta clusters).

# Usage: Rscript {maxK} {cap} {cellClusterPath} {clusterAvgPath} {cellConsensusPath} {seed}

# - maxK: number of consensus clusters
# - cap: maximum z-score cutoff
# - cellClusterPath: path to the cell-level data containing the counts of each SOM pixel/meta clusters per cell, labeled with cell SOM clusters
# - clusterAvgPath: path to the averaged cell data table (as defined above)
# - cellConsensusPath: path to file where the cell consensus cluster results will be written
# - clustToMeta: path to file where the SOM cluster to meta cluster mapping will be written
# - seed: random factor

library(arrow)
library(data.table)
library(ConsensusClusterPlus)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# get the cluster cols to subset over
clusterCols <- unlist(strsplit(args[1], split=","))

# get number of consensus clusters
maxK <- strtoi(args[2])

# get z-score scaling factor
cap <- strtoi(args[3])

# get the cell cluster path
cellClusterPath <- args[4]

# get path to the averaged cluster data
clusterAvgPath <- args[5]

# get consensus cluster write path
cellConsensusPath <- args[6]

# get the clust to meta write path
clustToMeta <- args[7]

# set the random seed
seed <- strtoi(args[8])
set.seed(seed)

print("Reading cluster averaged data")
clusterAvgs <- arrow::read_feather(clusterAvgPath)

# scale and cap the data respectively
# note: z-scoring and capping cluster avg data produces better clustering results
print("Scaling data")
clusterAvgsScale <- pmin(scale(clusterAvgs[clusterCols]), cap)

# run the consensus clustering
# TODO: also look into invisible() function here (not urgent, just to prevent printout)
print("Running consensus clustering")
consensusClusterResults <- ConsensusClusterPlus(t(clusterAvgsScale), maxK=maxK, seed=seed)
hClust <- consensusClusterResults[[maxK]]$consensusClass
names(hClust) <- clusterAvgs$cluster

# append hClust to data
print("Writing consensus clustering")
cellClusterData <- arrow::read_feather(cellClusterPath)
cellClusterData$hCluster_cap <- hClust[as.character(cellClusterData$cluster)]
arrow::write_feather(as.data.table(cellClusterData), cellConsensusPath)

# save the mapping from cluster to hCluster_cap
print("Writing SOM to meta cluster mapping table")
hClustLabeled <- as.data.table(hClust)
hClustLabeled$cluster <- as.integer(rownames(hClustLabeled))
hClustLabeles <- setnames(hClustLabeled, "hClust", "hCluster_cap")
arrow::write_feather(hClustLabeled, clustToMeta)
