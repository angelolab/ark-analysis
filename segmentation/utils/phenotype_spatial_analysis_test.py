import numpy as np
import pandas as pd
import random
from segmentation.utils import spatial_analysis_test
from segmentation.utils import phenotype_spatial_analysis
import importlib
importlib.reload(phenotype_spatial_analysis)
importlib.reload(spatial_analysis_test)


def make_pheno_codes():
    # Create example phenotype list with IDs for test

    pheno = pd.DataFrame(np.zeros((2, 2)))
    # Assigns the two cell phenotypes to ID 1 and 2
    pheno[1] = np.arange(len(pheno[1])) + 1
    # Creates 2 phenotype names
    pheno[0] = ["Pheno1", "Pheno2"]
    return pheno


def test_visualize_z_scores():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z')+1)]
    plot = phenotype_spatial_analysis.visualize_z_scores(z, pheno_titles)


def test_calculate_phenotype_spatial_enrichment():
    # Test z and p values

    pheno = make_pheno_codes()

    # Positive enrichment
    all_patient_data_pos = spatial_analysis_test.make_expression_matrix("positive")
    dist_mat_pos = spatial_analysis_test.make_distance_matrix("positive")
    values, stats = \
        phenotype_spatial_analysis.calculate_phenotype_spatial_enrichment(
            all_patient_data_pos, pheno, dist_mat_pos,
            patient_idx=30, cell_label_idx=24, flowsom_idx=31, bootstrap_num=100, dist_lim=100)
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] < .05
    assert stats.loc["Point8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert stats.loc["Point8", "z", "Pheno1", "Pheno2"] > 0
    # Negative enrichment
    all_patient_data_neg = spatial_analysis_test.make_expression_matrix("negative")
    dist_mat_neg = spatial_analysis_test.make_distance_matrix("negative")
    values, stats = \
        phenotype_spatial_analysis.calculate_phenotype_spatial_enrichment(
            all_patient_data_neg, pheno, dist_mat_neg,
            patient_idx=30, cell_label_idx=24, flowsom_idx=31, bootstrap_num=100, dist_lim=100)
    assert stats.loc["Point8", "p_neg", "Pheno1", "Pheno2"] < .05
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats.loc["Point8", "z", "Pheno1", "Pheno2"] < 0
    # No enrichment
    all_patient_data = spatial_analysis_test.make_expression_matrix("none")
    dist_mat = spatial_analysis_test.make_distance_matrix("none")
    values, stats = \
        phenotype_spatial_analysis.calculate_phenotype_spatial_enrichment(
            all_patient_data, pheno, dist_mat,
            patient_idx=30, cell_label_idx=24, flowsom_idx=31, bootstrap_num=100, dist_lim=100)
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert abs(stats.loc["Point8", "z", "Pheno1", "Pheno2"]) < 2
