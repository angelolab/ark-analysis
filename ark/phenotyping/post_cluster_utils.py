import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ark.utils import data_utils, load_utils, plot_utils


def plot_hist_thresholds(cell_table, populations, marker, pop_col='cell_meta_cluster',
                         threshold=None, percentile=0.999):
    """Create histograms to compare marker distributions across cell populations

    Args:
        cell_table (pd.DataFrame): cell table with clustered cell populations
        populations (list): populations to plot as stacked histograms
        marker (str): the marker used to generate the histograms
        pop_col (str): the column containing the names of the cell populations
        threshold (float, None): optional value to plot a horizontal line for visualization
        percentile (float): cap used to control x axis limits of the plot
    """
    all_populations = cell_table[pop_col].unique()

    # input validation
    if type(populations) != list:
        raise ValueError("populations argument must be a list of populations to plot")

    # check that provided populations are present in dataframe
    for pop in populations:
        if pop not in all_populations:
            raise ValueError("Invalid population name found in populations: {}".format(pop))

    if marker not in cell_table.columns:
        raise ValueError("Could not find {} as a column in cell table".format(marker))

    # determine max value based on first positive population
    vals = cell_table.loc[cell_table[pop_col] == populations[0], marker].values
    x_max = np.quantile(vals, percentile)

    # plot each pop histogram
    pop_num = len(populations)
    fig, ax = plt.subplots(pop_num, 1, figsize=[6.4, 2.2 * pop_num])
    for i in range(pop_num):
        plot_vals = cell_table.loc[cell_table[pop_col] == populations[i], marker].values
        ax[i].hist(plot_vals, 50, density=True, facecolor='g', alpha=0.75, range=(0, x_max))
        ax[i].set_title("Distribution of {} in {}".format(marker, populations[i]))

        if threshold is not None:
            ax[i].axvline(x=threshold)
    plt.tight_layout()


def create_mantis_project(cell_table, fovs, seg_dir, pop_col, mask_dir, image_dir, mantis_dir):
    """Create a complete Mantis project for viewing cell labels

    Args:
        cell_table (pd.DataFrame): dataframe of extracted cell features and subtypes
        fovs (list): list of FOVs to use for creating the project
        seg_dir (path): path to the directory containing the segmentations
        pop_col (str): the column containing the distinct cell populations
        mask_dir (path): path to the directory where the masks will be stored
        image_dir (path): path to the directory containing the raw image data
        mantis_dir (path): path to the directory where the mantis project will be created """

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # create small df compatible with FOV function
    small_table = cell_table.loc[:, [pop_col, 'label', 'fov']]

    # generate unique numeric value for each population
    small_table['pop_vals'] = pd.factorize(small_table[pop_col].tolist())[0] + 1

    # label and save the cell mask for each FOV
    for fov in fovs:
        whole_cell_file = [fov + '_feature_0.tiff' for fov in fovs]

        # load the segmentation labels in for the FOV
        label_map = load_utils.load_imgs_from_dir(
            data_dir=seg_dir, files=whole_cell_file, xr_dim_name='compartments',
            xr_channel_names=['whole_cell'], trim_suffix='_feature_0'
        ).loc[fov, ...]

        # use label_cells_by_cluster to create cell masks
        mask_data = data_utils.label_cells_by_cluster(
            fov, small_table, label_map, fov_col='fov',
            cell_label_column='label', cluster_column='pop_vals'
        )

        # save the cell mask for each FOV
        data_utils.save_fov_mask(
            fov,
            mask_dir,
            mask_data,
            sub_dir=None,
            name_suffix='_cell_mask'
        )

    # rename the columns of small_table
    mantis_df = small_table.rename({'pop_vals': 'metacluster', pop_col: 'mc_name'}, axis=1)

    # create the mantis project
    plot_utils.create_mantis_dir(fovs=fovs, mantis_project_path=mantis_dir,
                                 img_data_path=image_dir, mask_output_dir=mask_dir,
                                 mask_suffix='_cell_mask', mapping=mantis_df,
                                 seg_dir=seg_dir, img_sub_folder='')
