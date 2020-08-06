import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap


def visualize_dimensionality_reduction(cell_data, columns, category, color_map="Spectral", algorithm="UMAP", save_dir=None):
    """Plots the dimensionality reduction of specified population columns
                Args:
                    cell_data (pd df): Dataframe containing columns for dimensionality reduction and category
                    columns (list): List of column names that are included for dimensionality reduction
                    category (str): Name of column in dataframe containing categorical Population or Patient data
                    color_map (str): Name of MatPlotLib ColorMap used, default is Spectral
                    algorithm (str): Name of dimensionality reduction algorithm, default is UMAP
                    save_dir (str): Directory to save plots, default is None"""
    cell_data = cell_data.dropna()

    if algorithm == "UMAP":
        # Instantiate UMAP
        reducer = umap.UMAP()

        column_data = cell_data[columns].values
        scaled_column_data = StandardScaler().fit_transform(column_data)
        embedding = reducer.fit_transform(scaled_column_data)

        plt.scatter(embedding[:, 0], embedding[:, 1], cmap='color_map',
                    c=sns.color_palette(color_map, n_colors=len(cell_data)))
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of data', fontsize=24);
        plt.colorbar()
        plt.legend()
        if save_dir is not None:
          plt.savefig(save_dir + "UmapVisualization.png")

    elif algorithm == "PCA":
        pca = PCA()
        pca_result = pca.fit_transform(cell_data[columns].values)

        cell_data['pca-one'] = pca_result[:, 0]
        cell_data['pca-two'] = pca_result[:, 1]
        sns.scatterplot(x="pca-one", y="pca-two", hue=cell_data[category], palette=color_map, data=cell_data, legend="full",
                        alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if save_dir is not None:
          plt.savefig(save_dir + "PCAVisualization.png")

    elif algorithm == "tSNE":
        tsne = TSNE()
        tsne_results = tsne.fit_transform(cell_data[columns].values)

        cell_data['tsne-2d-one'] = tsne_results[:, 0]
        cell_data['tsne-2d-two'] = tsne_results[:, 1]

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=cell_data[category],
            palette=color_map,
            data=cell_data,
            legend="full",
            alpha=0.3
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if save_dir is not None:
          plt.savefig(save_dir + "tSNEVisualization.png")