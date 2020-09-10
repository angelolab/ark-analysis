import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_boxplot(cell_data, col_name, col_split=None, split_vals=None, save_dir=None):
    """Draws a boxplot for a given column, optionally with help from a split column

    Args:
        cell_data (pandas.DataFrame):
            Dataframe containing columns with Patient ID and Cell Name
        col_name (str):
            Name of the column we wish to draw a box-and-whisker plot for
        col_split (str):
            If specified, used for additional box-and-whisker plot faceting
        split_vals (list):
            If specified, only visualize the specified values in the col_split column
        save_dir (str):
            If specified, a directory where we will save the plot
    """

    # the col_name must be valid
    if col_name not in cell_data.columns.values:
        raise ValueError("col_name specified does not exist in data provided")

    # basic error checks if split_vals is set
    if split_vals:
        # the user cannot specify split_vales without specifying col_split
        if split_vals and not col_split:
            raise ValueError("If split_vals is set, then col_split must also be set")

        # all the values in split_vals must exist in the col_name of cell_data
        if not all(val in cell_data[col_split].unique() for val in split_vals):
            raise ValueError("Some values in split_vals do not exist in the col_split column of data")

    # don't modify cell_data in anyway
    data_to_viz = cell_data.copy(deep=True)

    # ignore values in col_split not in split_vals if split_vals is set
    if split_vals:
        data_to_viz = data_to_viz[data_to_viz[col_split].isin(split_vals)]

    if col_split:
        # if col_split, then we explicitly facet the visualization
        # labels are automatically generated in Seaborn
        sns.boxplot(x=col_split, y=col_name, data=cell_data)
        plt.title("Distribution of %s, faceted by %s" % (col_name, col_split))
    else:
        # otherwise, we don't facet anything, but we have to explicitly make vertical
        sns.boxplot(x=col_name, data=cell_data, orient="v")
        plt.title("Distribution of %s" % col_name)

    # save visualization to a directory if specified
    if save_dir is not None:
        if not os.path.exists(save_dir):
            raise ValueError("save_dir %s does not exist" % save_dir)

        plt.savefig(os.path.join(save_dir, "sample_boxplot_viz.png"))


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
            if not os.path.exists(save_dir):
                raise ValueError("save_dir %s does not exist" % save_dir)

            plt.savefig(os.path.join(save_dir, "TotalPopulationDistribution.png"))

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
            if not os.path.exists(save_dir):
                raise ValueError("save_dir %s does not exist" % save_dir)

            plt.savefig(os.path.join(save_dir, "PopulationDistribution.png"))

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
            if not os.path.exists(save_dir):
                raise ValueError("save_dir %s does not exist" % save_dir)

            plt.savefig(os.path.join(save_dir, "PopulationProportion.png"))
