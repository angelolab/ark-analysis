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


def visualize_patient_population_distribution(df, patient_col_name, population_col_name, color_map="jet"):
    """Plots the distribution of the population given by total count, direct count, and proportion
              Args:
                  df: Pandas Dataframe containing columns with Patient ID and Cell Name
                  id_col_name: Name of column containing categorical Patient data
                  cell_col_name: Name of column in dataframe containing categorical Population data
                  color_map: Name of MatPlotLib ColorMap used, default is jet"""
    ids = set(df[patient_col_name])
    names = df[population_col_name].value_counts().index.tolist()

    # Gets unique IDs, accounting for NaNs
    unique_ids = []
    for item in ids:
        if type(item) is float or type(item) is int:
            unique_ids.append(item)

    unique_ids = [x for x in unique_ids if x == x]

    def getSortedDf(isNormalized=False):
        # Flipping the data so it's easily graphable
        df_stacked = pd.DataFrame()
        for id in unique_ids:
            df_rows = df.loc[df[patient_col_name] == id]
            row_pop_types = None

            # Normalize it if asking for proportion, otherwise don't
            if (isNormalized):
                row_pop_types = df_rows[population_col_name].value_counts(normalize=True) * 100
            else:
                row_pop_types = df_rows[population_col_name].value_counts()

            df_stacked = pd.concat([df_stacked, (pd.DataFrame([row_pop_types.values], columns=row_pop_types.index))],
                                   axis=0, ignore_index=True)

        # Gathering sums to sort dataframe by population count
        sums = []
        for index, row in df_stacked.iterrows():
            sum = 0
            for name in names:
                val = row[name]
                if (val == val):
                    sum += val
            sums.append(sum)

        # Sort rows by total population count
        df_stacked.insert(0, "Total Population Count", sums, True)
        df_stacked = df_stacked.sort_values(by="Total Population Count", ascending=False)
        df_stacked = df_stacked.drop(columns=["Total Population Count"])

        # Sort columns by total count
        unsorted_pop_names = df_stacked.columns.tolist()
        pop_sums = df_stacked.sum().sort_values(ascending=False)

        def swap_columns(df, c1, c2):
            df['temp'] = df[c1]
            df[c1] = df[c2]
            df[c2] = df['temp']
            df.drop(columns=['temp'], inplace=True)
            return df

        x = 0
        for pop in pop_sums.index.tolist():
            col_name = unsorted_pop_names[x]
            if (col_name != pop):
                df_stacked = swap_columns(df_stacked, col_name, pop)

        df_stacked = df_stacked[pop_sums.index.tolist()]
        return df_stacked

    # Plot by total count
    population_values = df[population_col_name].value_counts()
    population_values.plot.bar(colormap=color_map)
    plt.title("Distribution of Population in all patients")
    plt.xlabel("Population Type")
    plt.ylabel("Population Count")

    # Plot by count
    getSortedDf().plot.bar(stacked=True, colormap=color_map)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Patient ID")
    plt.ylabel("Cell Count")
    plt.title("Distribution of Population Count in Patients")

    # Plot by Proportion
    getSortedDf(isNormalized=True).plot.bar(stacked=True, legend=False, colormap=color_map)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Patient ID")
    plt.ylabel("Population Proportion")
    plt.title("Distribution of Population Count Proportion in Patients")
