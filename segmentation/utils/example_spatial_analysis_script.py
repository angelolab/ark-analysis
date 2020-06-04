import numpy as np
import pandas as pd
import h5py
from segmentation.utils import spatial_analysis
from segmentation.utils import visualize
from segmentation.utils import spatial_analysis_utils
from skimage import io
import importlib
importlib.reload(spatial_analysis)
importlib.reload(visualize)
importlib.reload(spatial_analysis_utils)

# This script takes a cell expression matrix, label maps for the fovs to be included in the analysis, and, depending
# on the analysis, a threshold matrix for channel or cluster spatial analysis. In channel spatial analysis, cells for
# a specific fov are identified, and cells of particular phenotypes are compared to each other to check for positive,
# negative, or no enrichment. To do this, a distance matrix is created from the label_maps, cell phenotypes are
# identified by their labels in the image and then significant interactions between different populations of phenotypes
# are recorded. Similar analysis is also done for channel spatial enrichment; however, instead of looking at cell
# phenotypes, markers positive for specific thresholds are identified and specific interactions are then characterized
# between them.

# Import data
# This is the cell expression matrix including data for fovs 6 and 7, their cell labels, marker expression,
# cell phenotypes, and cell phenotype IDs.
all_data = pd.read_csv("/Users/jaiveersingh/Desktop/tests/example_expression_matrix.csv")
# This is the threshold matrix with all marker thresholds - for channel cpatial enrichment
marker_thresholds = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
marker_thresholds = marker_thresholds.drop(0, axis=0)
# This is the label maps from which the distance matrix will be computed
label_map_six = io.imread("/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmod.tiff")
label_map_seven = io.imread("/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmodPoint7.tiff")

# Reshape the label maps and stack them to use as inputs to compute their distance matrices
label_map_six = label_map_six.reshape(1, label_map_six.shape[0], label_map_six.shape[1])
label_map_seven = label_map_seven.reshape(1, label_map_seven.shape[0], label_map_seven.shape[1])
label_maps = np.stack((label_map_six, label_map_seven), axis=0)
# Get an H5py object with the respective distance matrices for the points
dist_mats = spatial_analysis_utils.calc_dist_matrix(label_maps)

# Now with the distance matrix run the distance matrix, threshold values, and the expression matrix through
# channel spatial enrichment

# Columns, other than the marker columns, in the expression data
# These columns will be excluded from the analysis, so that a matrix of only markers can be extracted
excluded_colnames = ["SampleID", "cellLabelInImage", "cellSize", "C", "Na", "Si", "Background", "HH3",
                     "Ta", "Au", "Tissue", "PatientID", "lineage", "cell_type",
                     "cell_lin", "lintype_num", "FlowSOM_ID"]

values_channel, stats_channel = spatial_analysis.calculate_channel_spatial_enrichment(
    dist_mats, marker_thresholds, all_data, excluded_colnames)

# Now with the same parameters, cluster spatial analysis (based on cell types rather than positive marker expression
# by thresholds) will be done

values_cluster, stats_cluster = spatial_analysis.calculate_cluster_spatial_enrichment(
    all_data, dist_mats)

# To then visualize the z scores, a clustermap can be produced

# First, find all the cell phenotypes in the data to use as an input for the clustergram
pheno_titles = all_data["cell_type"].drop_duplicates()
visualize.visualize_z_scores(stats_cluster.loc[6, "z", :, :].values, pheno_titles)
