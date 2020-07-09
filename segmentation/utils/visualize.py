import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

from matplotlib.patches import Ellipse


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

def draw_ellipsoids(centroid_info):
    """
    Visualize ellipses as defined by a list of information such as the center.

    Args:
        centroid_info: a list of tuples in the format:
            (center, width, height, angle)
    """

    for c in centroid_info:
        ell = Ellipse(xy=c[0], width=c[1], height=c[2], angle=c[3])
        ell.set_facecolor('red')
