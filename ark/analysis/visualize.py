import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_z_scores(z, pheno_titles):
    """Plots the z scores between all phenotypes as a clustermap.

    Args:
        z (numpy.ndarray): array of z scores of shape (pheno_num, pheno_num)
        pheno_titles (list): list of all the names of the phenotypes
    """
    # Replace the NA's and inf values with 0s
    z[np.isnan(z)] = 0
    z[np.isinf(z)] = 0
    # Assign numpy values respective phenotype labels
    zplot = pd.DataFrame(z, columns=pheno_titles, index=pheno_titles)
    sns.set(font_scale=.7)
    sns.clustermap(zplot, figsize=(8, 8), cmap="vlag")


def get_sorted_data(cell_data, patient_col_name, population_col_name, is_normalized=False):
    """Gets the cell data and generates a new Sorted DataFrame with each row representing a
    patient and column representing Population categories

    Args:
        cell_data (pandas.DataFrame):
            Dataframe containing columns with Patient ID and Cell Name
        patient_col_name (str):
            Name of column containing categorical Patient data
        population_col_name (str):
            Name of column in dataframe containing categorical Population data
        is_normalized (bool):
            Boolean specifying whether to normalize cell counts or not, default is False

    Returns:
        pandas.DataFrame:
            DataFrame with rows and columns sorted by population
    """

    cell_data_stacked = pd.crosstab(
        cell_data[patient_col_name],
        cell_data[population_col_name],
        normalize='index' if is_normalized else False
    )

    # Sorts by Kagel Method :)
    id_order = cell_data.groupby(patient_col_name).count().sort_values(
        by=population_col_name,
        ascending=False
    ).index.values

    pop_order = cell_data.groupby(population_col_name).count().sort_values(
        by=patient_col_name,
        ascending=False
    ).index.values

    cell_data_stacked = cell_data_stacked.reindex(id_order, axis='index')
    cell_data_stacked = cell_data_stacked.reindex(pop_order, axis='columns')

    return cell_data_stacked


def visualize_patient_population_distribution(cell_data, patient_col_name, population_col_name,
                                              color_map="jet", show_total_count=True,
                                              show_distribution=True, show_proportion=True,
                                              save_dir=None):
    """Plots the distribution of the population given by total count, direct count, and proportion

    Args:
        cell_data (pandas.DataFrame):
            Dataframe containing columns with Patient ID and Cell Name
        patient_col_name (str):
            Name of column containing categorical Patient data
        population_col_name (str):
            Name of column in dataframe containing Population data
        color_map (str):
            Name of MatPlotLib ColorMap used. Default is jet
        show_total_count (bool):
            Boolean specifying whether to show graph of total population count, default is true
        show_distribution (bool):
            Boolean specifying whether to show graph of population distribution, default is true
        show_proportion (bool):
            Boolean specifying whether to show graph of total count, default is true
        save_dir (str):
            Directory to save plots, default is None
    """
    cell_data = cell_data.dropna()

    # Plot by total count
    if show_total_count:
        population_values = cell_data[population_col_name].value_counts()
        population_values.plot.bar(colormap=color_map)
        plt.title("Distribution of Population in all patients")
        plt.xlabel("Population Type")
        plt.ylabel("Population Count")
        if save_dir is not None:
            plt.savefig(save_dir + "TotalPopulationDistribution.png")

    # Plot by count
    if show_distribution:
        get_sorted_data(cell_data, patient_col_name,
                        population_col_name).plot.bar(stacked=True,
                                                      colormap=color_map)

        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel(patient_col_name)
        plt.ylabel(population_col_name)
        plt.title("Distribution of Population Count in Patients")
        if save_dir is not None:
            plt.savefig(save_dir + "PopulationDistribution.png")

    # Plot by Proportion
    if show_proportion:
        get_sorted_data(cell_data, patient_col_name,
                        population_col_name, is_normalized=True).plot.bar(stacked=True,
                                                                          legend=False,
                                                                          colormap=color_map)

        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel(patient_col_name)
        plt.ylabel(population_col_name)
        plt.title("Distribution of Population Count Proportion in Patients")
        if save_dir is not None:
            plt.savefig(save_dir + "PopulationProportion.png")
