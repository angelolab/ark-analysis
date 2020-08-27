from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import os


def visualize_dimensionality_reduction(cell_data, columns, category, color_map="Spectral",
                                       algorithm="UMAP", save_dir=None):
    """Plots the dimensionality reduction of specified population columns

    Args:
        cell_data (pandas.DataFrame):
            Dataframe containing columns for dimensionality reduction and category
        columns (list):
            List of column names that are included for dimensionality reduction
        category (str):
            Name of column in dataframe containing population or patient data
        color_map (str):
            Name of MatPlotLib ColorMap used, default is Spectral
        algorithm (str):
            Name of dimensionality reduction algorithm, default is UMAP
        save_dir (str):
            Directory to save plots, default is None
    """
    cell_data = cell_data.dropna()

    if algorithm not in ["UMAP", "PCA", "tSNE"]:
        raise ValueError(f"The algorithm specified must be one of the following: "
                         f"{['UMAP', 'PCA', 'tSNE']}")

    if algorithm == "UMAP":
        reducer = umap.UMAP()

        column_data = cell_data[columns].values
        scaled_column_data = StandardScaler().fit_transform(column_data)
        embedding = reducer.fit_transform(scaled_column_data)

        fig1 = plt.figure(1)
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=cell_data[category],
                        palette=color_map, data=cell_data, legend="full", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('UMAP projection of data', fontsize=24)
        fig1.show()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "UMAPVisualization.png"))

    elif algorithm == "PCA":
        pca = PCA()
        pca_result = pca.fit_transform(cell_data[columns].values)

        fig2 = plt.figure(2)
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=cell_data[category],
                        palette=color_map, data=cell_data, legend="full", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('PCA projection of data', fontsize=24)
        fig2.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "PCAVisualization.png"))

    elif algorithm == "tSNE":
        tsne = TSNE()
        tsne_results = tsne.fit_transform(cell_data[columns].values)

        fig3 = plt.figure(3)
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=cell_data[category],
            palette=color_map,
            data=cell_data,
            legend="full",
            alpha=0.3
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('tSNE projection of data', fontsize=24)
        fig3.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "tSNEVisualization.png"))
