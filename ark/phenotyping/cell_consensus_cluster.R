# Runs consensus clustering on the cell data averaged across all columns

# Usage: Rscript {maxK} {cap} {cellClusterPath} {clusterAvgPath} {cellConsensusPath} {seed}

# - maxK: number of consensus clusters
# - cap: z-score scaling factor
# - cellClusterPath: path to the data containing the data with cell clusters
# - clusterAvgPath: path to the data containing averages of each column grouped by cell SOM cluster
# - cellConsensusPath: path to file where the cell consensus cluster results will be written
# - seed: random factor

library(arrow)
library(data.table)
library(ConsensusClusterPlus)

# get the command line arguments
args <- commandArgs(trailingOnly=TRUE)

# get number of consensus clusters
maxK <- strtoi(args[1])

# get z-score scaling factor
cap <- strtoi(args[2])

# get the cell cluster path
cellClusterPath <- args[3]

# get path to the averaged cluster data
clusterAvgPath <- args[4]

# get consensus cluster write path
cellConsensusPath <- args[5]

# set the random seed
seed <- strtoi(args[6])
set.seed(seed)

print("Reading cluster averaged data")
clusterAvgs <- arrow::read_feather(clusterAvgPath)

# scale and cap the data respectively
print("Scaling data")
clusterAvgsScale <- pmin(scale(clusterAvgs), cap)

# run the consensus clustering
print("Running consensus clustering")
consensusClusterResults <- ConsensusClusterPlus(t(clusterAvgsScale), maxK=maxK, seed=seed)
hClust <- consensusClusterResults[[maxK]]$consensusClass
names(hClust) <- clusterAvgs$cluster

# append hClust to data
print("Writing consensus clustering")
cellClusterData <- arrow::read_feather(cellClusterPath)
cellClusterData$hCluster_cap <- hClust[as.character(cellClusterData$cluster)]
arrow::write_feather(as.data.table(cellClusterData), cellConsensusPath)
