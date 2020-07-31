import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def swap_columns(df, c1, c2):
  df['temp'] = df[c1]
  df[c1] = df[c2]
  df[c2] = df['temp']
  df.drop(columns=['temp'], inplace=True)
  return df


def getSortedDf(df, patient_col_name, population_col_name, isNormalized=False):
    ids = df[patient_col_name]
    names = df[population_col_name].value_counts().index.tolist()
    unique_ids = df[patient_col_name].unique()

    df_stacked=pd.DataFrame()
    if isNormalized:
      df_stacked = pd.crosstab(df[patient_col_name], df[population_col_name], normalize='index')
    else:
      df_stacked = pd.crosstab(df[patient_col_name], df[population_col_name])

    # Sorts by Kagel Method
    id_order = df.groupby(patient_col_name).count().sort_values(by=population_col_name, ascending=False).index.values
    pop_order = df.groupby(population_col_name).count().sort_values(by=patient_col_name, ascending=False).index.values
    df_stacked = df_stacked.reindex(id_order, axis='index')
    df_stacked = df_stacked.reindex(pop_order, axis='columns')

    return df_stacked


def visualize_patient_population_distribution(df, patient_col_name, population_col_name, color_map="jet"):
    """Plots the distribution of the population given by total count, direct count, and proportion
              Args:
                  df: Pandas Dataframe containing columns with Patient ID and Cell Name
                  patient_col_name: Name of column containing categorical Patient data
                  population_col_name: Name of column in dataframe containing categorical Population data
                  color_map: Name of MatPlotLib ColorMap used, default is jet"""
    df = df.dropna()

    # Plot by total count
    population_values = df[population_col_name].value_counts()
    population_values.plot.bar(colormap=color_map)
    plt.title("Distribution of Population in all patients")
    plt.xlabel("Population Type")
    plt.ylabel("Population Count")

    # Plot by count
    getSortedDf(df, patient_col_name, population_col_name).plot.bar(stacked=True, colormap=color_map)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Patient ID")
    plt.ylabel("Cell Count")
    plt.title("Distribution of Population Count in Patients")

    # Plot by Proportion
    getSortedDf(df, patient_col_name, population_col_name,isNormalized=True).plot.bar(stacked=True, legend=False, colormap=color_map)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Patient ID")
    plt.ylabel("Population Proportion")
    plt.title("Distribution of Population Count Proportion in Patients")
    