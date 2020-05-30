import pandas as pd
import numpy as np
import xarray as xr
# import seaborn as sns
from segmentation.utils import spatial_analysis_utils
import importlib
importlib.reload(spatial_analysis_utils)

# Erin's Data Inputs
# all_patient_data = pd.read_csv("/Users/jaiveersingh/Desktop/granA_cellpheno_CS-asinh-norm_matlab_revised.csv")
# pheno = pd.read_csv("/Users/jaiveersingh/Downloads/CellType_GlobalSpatialEnrichment/cellpheno_numkey.csv")
# dist_mat = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))


# def visualize_clustermap(z, pheno_titles):
# visualize
# zplot = np.array(z)
# zplot[np.isnan(zplot)] = 0
# zplot[np.isnan(zplot)] = 0
# zplot = pd.DataFrame(zplot, columns=pheno_titles, index=pheno_titles)
# sns.clustermap(zplot)


def cell_label_spatial_enrichment(all_patient_data, pheno, dist_mat, points=None,
                                  patient_idx=0, cell_label_idx=1, flowsom_idx=52, bootstrap_num=1000, dist_lim=100):
    num_points = 0
    if points is None:
        points = list(set(all_patient_data.iloc[:, patient_idx]))
        num_points = len(points)
    else:
        num_points = len(points)

    # Error Checking
    if not np.isin(points, all_patient_data.iloc[:, patient_idx]).all():
        raise ValueError("Points were not found in Expression Matrix")

    cell_pheno_idx = 2
    values = []

    all_patient_data = all_patient_data[all_patient_data.columns[[patient_idx, cell_label_idx, flowsom_idx]]]
    pheno_titles = pheno.iloc[:, 0]
    pheno_codes = pheno.iloc[:, 1]
    pheno_num = len(pheno_codes)

    stats_raw_data = np.zeros((num_points, 7, pheno_num, pheno_num))
    coords = [points, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"], pheno_titles,
              pheno_titles]
    dims = ["points", "stats", "pheno1", "pheno2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    for point in points:
        patient_ids = all_patient_data.iloc[:, patient_idx] == point
        patient_data = all_patient_data[patient_ids]

        close_num, pheno1_num, pheno2_num = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_mat, dist_lim=dist_lim, num=pheno_num, analysis_type="Cell Label",
            cell_label_idx=cell_label_idx, patient_data=patient_data, cell_pheno_idx=cell_pheno_idx,
            pheno_codes=pheno_codes)
        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            pheno1_num, pheno2_num, dist_mat, pheno_num, dist_lim, bootstrap_num)
        values.append((close_num, close_num_rand))

        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[point, :, :] = stats_xr.values
    return values, stats
