import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns


def visualize_z_scores(z, pheno_titles):
    """Plots the z scores between all phenotypes as a clustermap.

    Args:
        z: array of z scores of shape (pheno_num x pheno_num)
        pheno_titles: list of all the names of the phenotypes"""
    # visualize
    # Replace the NA's and inf values with 0s
    z[np.isnan(z)] = 0
    z[np.isinf(z)] = 0
    # Assign numpy values respective phenotype labels
    zplot = pd.DataFrame(z, columns=pheno_titles, index=pheno_titles)
    sns.set(font_scale=.7)
    sns.clustermap(zplot, figsize=(8, 8), cmap="vlag")
