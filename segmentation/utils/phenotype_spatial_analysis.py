import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import seaborn as sns
from segmentation.utils import spatial_analysis_utils
import importlib
importlib.reload(spatial_analysis_utils)

# Erin's Data Inputs
# all_patient_data = pd.read_csv("/Users/jaiveersingh/Desktop/granA_cellpheno_CS-asinh-norm_matlab_revised.csv")
# pheno = pd.read_csv("/Users/jaiveersingh/Downloads/CellType_GlobalSpatialEnrichment/cellpheno_numkey.csv")
# dist_mat = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))


def visualize_z_scores(z, pheno_titles):
    # visualize
    zplot = np.array(z)
    # Replace the NA's and inf values with 0s
    zplot[np.isnan(zplot)] = 0
    zplot[np.isinf(zplot)] = 0
    # Assign numpy values respective phenotype labels
    zplot = pd.DataFrame(zplot, columns=pheno_titles, index=pheno_titles)
    sns.set(font_scale=.7)
    sns.clustermap(zplot, figsize=(8, 8), cmap="vlag")


def calculate_phenotype_spatial_enrichment(all_patient_data, pheno, dist_mat, points=None,
                                           patient_idx=0, cell_label_idx=1, flowsom_idx=52,
                                           bootstrap_num=1000, dist_lim=100):
    # Setup input and parameters
    num_points = 0
    if points is None:
        points = list(set(all_patient_data.iloc[:, patient_idx]))
        num_points = len(points)
    else:
        num_points = len(points)

    cell_pheno_idx = 2
    values = []

    # Error Checking
    if not np.isin(points, all_patient_data.iloc[:, patient_idx]).all():
        raise ValueError("Points were not found in Expression Matrix")

    # Subset matrix to only include the columns with the patient label, cell label, and cell phenotype
    all_patient_data = all_patient_data[all_patient_data.columns[[patient_idx, cell_label_idx, flowsom_idx]]]
    # Extract the names of the cell phenotypes
    pheno_titles = pheno.iloc[:, 0]
    # Extract the columns with the cell phenotype codes
    pheno_codes = pheno.iloc[:, 1]
    # Get the total number of phenotypes
    pheno_num = len(pheno_codes)

    # Create stats Xarray with the dimensions (points, stats variables, number of markers, number of markers)
    stats_raw_data = np.zeros((num_points, 7, pheno_num, pheno_num))
    coords = [points, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"], pheno_titles, pheno_titles]
    dims = ["points", "stats", "pheno1", "pheno2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    for point in points:
        # Subsetting expression matrix to only include patients with correct label
        patient_ids = all_patient_data.iloc[:, 0] == point
        patient_data = all_patient_data[patient_ids]

        # Get close_num and close_num_rand
        close_num, pheno1_num, pheno2_num = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_mat, dist_lim=dist_lim, num=pheno_num, analysis_type="Cell Label",
            cell_label_idx=cell_label_idx, patient_data=patient_data, cell_pheno_idx=cell_pheno_idx,
            pheno_codes=pheno_codes)
        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            pheno1_num, pheno2_num, dist_mat, pheno_num, dist_lim, bootstrap_num)
        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[point, :, :] = stats_xr.values
    return values, stats
