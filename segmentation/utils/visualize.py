import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def visualize_z_scores(z, pheno_titles):
    """Plots the z scores between all phenotypes as a clustermap.
    Args:
        z: array of z scores of shape (pheno_num x pheno_num)
        pheno_titles: list of all the names of the phenotypes"""
    # Replace the NA's and inf values with 0s
    z[np.isnan(z)] = 0
    z[np.isinf(z)] = 0
    # Assign numpy values respective phenotype labels
    zplot = pd.DataFrame(z, columns=pheno_titles, index=pheno_titles)
    sns.set(font_scale=.7)
    sns.clustermap(zplot, figsize=(8, 8), cmap="vlag")


def getSortedData(cell_data, patient_col_name, population_col_name, is_normalized=False):
    """Gets the cell data and generates a new Sorted DataFrame with each row representing a patient and column representing Population categories
                  Args:
                      cell_data: Pandas Dataframe containing columns with Patient ID and Cell Name
                      patient_col_name: Name of column containing categorical Patient data
                      population_col_name: Name of column in dataframe containing categorical Population data
                      is_normalized: Boolean specifying whether to normalize cell counts or not, default is False
                  Outputs:
                      cell_data_stacked: DataFrame with rows and columns sorted by population"""
    ids = cell_data[patient_col_name]
    names = cell_data[population_col_name].unique().tolist()
    unique_ids = cell_data[patient_col_name].unique()

    cell_data_stacked = pd.DataFrame()
    if is_normalized:
        cell_data_stacked = pd.crosstab(cell_data[patient_col_name], cell_data[population_col_name], normalize='index')
    else:
        cell_data_stacked = pd.crosstab(cell_data[patient_col_name], cell_data[population_col_name])

    # Sorts by Kagel Method
    id_order = cell_data.groupby(patient_col_name).count().sort_values(by=population_col_name, ascending=False).index.values
    pop_order = cell_data.groupby(population_col_name).count().sort_values(by=patient_col_name, ascending=False).index.values
    cell_data_stacked = cell_data_stacked.reindex(id_order, axis='index')
    cell_data_stacked = cell_data_stacked.reindex(pop_order, axis='columns')

    return cell_data_stacked


def visualize_patient_population_distribution(cell_data, patient_col_name, population_col_name, color_map="jet",
                                              show_total_count=True, show_distribution=True, show_proportion=True, save_dir=""):
    """Plots the distribution of the population given by total count, direct count, and proportion
              Args:
                  cell_data: Pandas Dataframe containing columns with Patient ID and Cell Name
                  patient_col_name: Name of column containing categorical Patient data
                  population_col_name: Name of column in dataframe containing categorical Population data
                  color_map: Name of MatPlotLib ColorMap used, default is jet
                  show_total_count: Boolean specifying whether to show graph of total population count, default is true
                  show_distribution: Boolean specifying whether to show graph of population distribution, default is true
                  show_proportion: Boolean specifying whether to show graph of total count, default is true
                  save_dir: Directory to save plots, default is blank"""
    cell_data = cell_data.dropna()

    # Plot by total count
    if show_total_count:
        population_values = cell_data[population_col_name].value_counts()
        population_values.plot.bar(colormap=color_map)
        plt.title("Distribution of Population in all patients")
        plt.xlabel("Population Type")
        plt.ylabel("Population Count")
        plt.savefig(save_dir + "TotalPopulationDistribution.png")

    # Plot by count
    if show_distribution:
        getSortedData(cell_data, patient_col_name, population_col_name).plot.bar(stacked=True, colormap=color_map)
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel("Patient ID")
        plt.ylabel("Cell Count")
        plt.title("Distribution of Population Count in Patients")
        plt.savefig(save_dir + "PopulationDistribution.png")

    # Plot by Proportion
    if show_proportion:
        getSortedData(cell_data, patient_col_name, population_col_name, is_normalized=True).plot.bar(stacked=True,
                                                                                                     legend=False,
                                                                                                     colormap=color_map)
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel("Patient ID")
        plt.ylabel("Population Proportion")
        plt.title("Distribution of Population Count Proportion in Patients")
        plt.savefig(save_dir + "PopulationProportion.png")


def visualize_dimensionality_reduction(cell_data, columns, category, color_map="Spectral", algorithm="UMAP", save_dir=""):
    """Plots the dimensionality reduction of specified population columns
                Args:
                    cell_data: Pandas Dataframe containing columns for dimensionality reduction and category
                    columns: List of column names that are included for dimensionality reduction
                    category: Name of column in dataframe containing categorical Population or Patient data
                    color_map: Name of MatPlotLib ColorMap used, default is Spectral
                    algorithm: Name of dimensionality reduction algorithm, default is UMAP
                    save_dir: Directory to save plots, default is blank"""
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
        plt.savefig(save_dir + "UmapVisualization.png")

    elif algorithm == "PCA":
        pca = PCA()
        pca_result = pca.fit_transform(cell_data[columns].values)

        cell_data['pca-one'] = pca_result[:, 0]
        cell_data['pca-two'] = pca_result[:, 1]
        sns.scatterplot(x="pca-one", y="pca-two", hue=cell_data[category], palette=color_map, data=cell_data, legend="full",
                        alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(save_dir + "PCAVisualization.png")

    elif algorithm == "tSNE":
        tsne = TSNE()
        tsne_results = tsne.fit_transform(cell_data[columns].values)

        cell_data['tsne-2d-one'] = tsne_results[:, 0]
        cell_data['tsne-2d-two'] = tsne_results[:, 1]

        # plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=cell_data[category],
            palette=color_map,
            data=cell_data,
            legend="full",
            alpha=0.3
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(save_dir + "tSNEVisualization.png")
