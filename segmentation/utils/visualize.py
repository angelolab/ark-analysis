import pandas as pd
import numpy as np
import matplotlib as plt
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


def newDict(names):
    result = {}
    for name in names:
        result[name] = [0]
    return result


def visualize_distribution_of_cell_count(df, id_col_name, cell_col_name):
    ids = set(df[id_col_name])
    print(ids)
    cell_names = df[cell_col_name].value_counts().index.tolist()

    unique_ids = []
    for item in ids:
        if type(item) is float or type(item) is int:
            unique_ids.append(item)

    unique_ids = [x for x in unique_ids if x == x]

    print(unique_ids)

    df_stacked = pd.DataFrame(newDict(cell_names))

    first = True
    for id in unique_ids:

        df_rows = df.loc[df["PatientID"] == id]
        row_cell_types = df_rows["cell_type"].value_counts()

        if first:
            df_stacked = pd.DataFrame([row_cell_types.values], columns=row_cell_types.index)
            first = False
        else:
            df_stacked = pd.concat([df_stacked, (pd.DataFrame([row_cell_types.values], columns=row_cell_types.index))],
                                   axis=0, ignore_index=True)

    df_stacked.plot.bar(stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Patient ID")
    plt.ylabel("Cell Count")
    plt.title("Distribution of Cell Count in Patients")


def visualize_proportion_of_cell_count(df, id_col_name, cell_col_name):
    ids = set(df[id_col_name])
    cell_names = df[cell_col_name].value_counts().index
    cell_names = cell_names.tolist()
    unique_ids = []

    for item in ids:
        if type(item) is float or type(item) is int:
            unique_ids.append(item)

    unique_ids = [x for x in unique_ids if x == x]

    df_stacked = pd.DataFrame(newDict(cell_names))

    first = True
    for id in unique_ids:

        df_rows = df.loc[df["PatientID"] == id]
        row_cell_types = df_rows["cell_type"].value_counts(normalize=True) * 100

        if first:
            df_stacked = pd.DataFrame([row_cell_types.values], columns=row_cell_types.index)
            first = False
        else:
            df_stacked = pd.concat([df_stacked, (pd.DataFrame([row_cell_types.values], columns=row_cell_types.index))],
                               axis=0, ignore_index=True)


    df_stacked.plot.bar(stacked=True, legend=False)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Patient ID")
    plt.ylabel("Cell Proportion")
    plt.title("Distribution of Cell Count Proportion in Patients")


def visualize_cell_distribution_in_all_patients(df, cell_col_name):
    cell_types = df[cell_col_name].value_counts()

    cell_types.plot.bar()
    plt.title("Distribution of Cells in all patients")
    plt.xlabel("Cell Type")
    plt.ylabel("Cell Count")
