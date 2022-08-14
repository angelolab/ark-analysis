import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

from ark.utils import io_utils, data_utils, load_utils


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
    fig, ax = plt.subplots(pop_num, 1)
    for i in range(pop_num):
        plot_vals = cell_table.loc[cell_table[pop_col] == populations[i], marker].values
        ax[i].hist(plot_vals, 50, density=True, facecolor='g', alpha=0.75, range=(0, x_max))
        ax[i].set_title("Distribution of {} in {}".format(marker, populations[i]))

        if threshold is not None:
            ax[i].axvline(x=threshold)
    plt.tight_layout()



# identify_thresholds(cell_table=cell_table, top_populations=['tumor_sma'], bottom_populations=['ck17_tumor', 'tumor_ecad'],
#                     marker='ECAD', threshold=0.001)
#
# # create CD4+ cells from CD3_noise population
# marker = 'CD4'
# threshold = 0.001
# target_pop = 'CD3_noise_split'
# new_pop = 'CD3_noise_CD4s'
# selected_idx = cell_table[marker] > threshold
# cell_table.loc[np.logical_and(cell_table['cell_meta_cluster'] == target_pop, selected_idx), 'cell_meta_cluster'] = new_pop
#
## # update cell table with post-inspection decisions
# cell_table.loc[cell_table['cell_meta_cluster'] == 'noise', 'cell_meta_cluster'] = 'tumor_other'


def create_updated_cell_masks(cell_table, fovs, seg_dir, pop_col, mask_dir):
    """Creates masks with the updated cell labels"""

    os.makedirs(mask_dir)

    # create small df compatible with FOV function
    small_table = cell_table.loc[:, [pop_col, 'label', 'fov']]

    # generate unique numeric value for each population
    unique_pops = small_table[pop_col].unique()
    small_table['pop_vals'] = small_table[pop_col].replace(to_replace=unique_pops,
                                                            value=list(range(1, len(unique_pops) + 1)))

    # define the file names for segmentation masks
    whole_cell_files = [fov + '_feature_0.tif' for fov in fovs]

    # load the segmentation labels in
    label_maps = load_utils.load_imgs_from_dir(data_dir=seg_dir,
                                               files=whole_cell_files,
                                               xr_dim_name='compartments',
                                               xr_channel_names=['whole_cell'],
                                               trim_suffix='_feature_0')

    # use label_cells_by_cluster to create cell masks
    img_data = data_utils.label_cells_by_cluster(
        fovs, small_table, label_maps, fov_col='fov',
        cell_label_column='label', cluster_column='pop_vals'
    )

    data_utils.save_fov_images(
            fovs,
            mask_dir,
            img_data,
            sub_dir=None,
            name_suffix='_cell_mask'
        )

#
# unique_vals = cell_table['cell_meta_cluster'].unique()
# cell_table['unique_vals'] = cell_table['cell_meta_cluster'].replace(to_replace=unique_vals, value=list(range(len(unique_vals))))
# cell_table = cell_table.rename(columns={'label': 'segmentation_label'})
# masks = data_utils.generate_cell_cluster_mask(fovs=fovs, seg_dir='/Volumes/Noah/segmentation_masks/',
#                                              cell_data=cell_table, cell_cluster_col='unique_vals')
#
#
# data_utils.save_fov_images(
#         fovs,
#         '/Volumes/Noah/cell_masks',
#         masks,
#         sub_dir=None,
#         name_suffix='_cell_mask'
#     )
#
# mantis_df = cell_table.loc[:, ['cell_meta_cluster', 'unique_vals']]
# mantis_df = mantis_df.drop_duplicates()
# mantis_df = mantis_df.rename({'unique_vals': 'metacluster', 'cell_meta_cluster': 'mc_name'}, axis=1)
#
# create_mantis_project(mantis_project_path='/Volumes/Noah/mantis_cell_first20',
#                       img_data_path='blank',
#                       mask_output_dir='/Volumes/Noah/cell_masks',
#                       mask_suffix='_cell_mask',
#                       map_df=mantis_df,
#                       seg_dir='/Volumes/Noah/segmentation_masks')
#
# # save updated cell table
# cell_table = pd.read_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/combined_cell_table_normalized_cell_labels_updated.csv')
#
#
# # find functional marker threshold cutoffs
# identify_thresholds(cell_table=cell_table, top_populations=['CD8T', 'CD163', 'CD4T'],
#                     bottom_populations=['tumor_ecad', 'FAP'],
#                     marker='TIM3', threshold=0.001, percentile=0.99)
#
# threshold_list = [['Ki67', 0.002], ['CD38', 0.002], ['CD45RB', 0.001], ['CD45RO', 0.002],
#                   ['CD57', 0.002], ['CD69', 0.002], ['GLUT1', 0.002], ['IDO', 0.001],
#                   ['PD1', 0.0005], ['PDL1', 0.0005, "tumors could use either threshold", 0.001],
#                   ['HLA1', 0.001], ['HLADR', 0.001], ['TBET', 0.0015], ['TCF1', 0.001],
#                   ['TIM3', 0.001]]
#
# marker, threshold = threshold_list[0]
# col_name = marker + '_threshold'
# cell_table[col_name] = cell_table['cell_meta_cluster']
# pos_idx = cell_table[marker] > threshold
#
# cell_table[col_name].values[pos_idx] = [marker + '_pos_'] + cell_table['cell_meta_cluster'].values[pos_idx]
# cell_table[col_name].values[~pos_idx] = [marker + '_neg_'] + cell_table['cell_meta_cluster'].values[~pos_idx]
#
#
# # create project
# unique_vals = cell_table[col_name].unique()
# cell_table['unique_vals'] = cell_table[col_name].replace(to_replace=unique_vals, value=list(range(1, len(unique_vals) + 1)))
# cell_table = cell_table.rename(columns={'label': 'segmentation_label'})
# masks = data_utils.generate_cell_cluster_mask(fovs=fovs, seg_dir='/Volumes/Noah/segmentation_masks/',
#                                              cell_data=cell_table, cell_cluster_col='unique_vals')
#
#
# data_utils.save_fov_images(
#         fovs,
#         '/Volumes/Noah/cell_masks',
#         masks,
#         sub_dir=None,
#         name_suffix='_cell_mask'
#     )
#
# mantis_df = cell_table.loc[:, [col_name, 'unique_vals']]
# mantis_df = mantis_df.drop_duplicates()
# mantis_df = mantis_df.rename({'unique_vals': 'metacluster', col_name: 'mc_name'}, axis=1)
#
# create_mantis_project(mantis_project_path='/Volumes/Noah/mantis_cell_first20',
#                       img_data_path='blank',
#                       mask_output_dir='/Volumes/Noah/cell_masks',
#                       mask_suffix='_cell_mask',
#                       map_df=mantis_df,
#                       seg_dir='/Volumes/Noah/segmentation_masks')
#
# cell_table.loc[np.logical_and(cell_table['segmentation_label'] == 234, cell_table['fov'] == 'TONIC_TMA18_R2C2'), ['cell_meta_cluster']]
#
# # export features for Mantis
# mantis_feature_df = cell_table.loc[np.isin(cell_table['fov'], fovs),
#                                    ['fov', 'segmentation_label', 'HLADR']]
#
# 'IDO', 'IDO', 'Ki67', 'LAG3', 'PD1', 'PDL1', 'TBET', 'TCF1', 'TIM3',
# mantis_feature_df.to_csv('/Volumes/Noah/mantis_cell_first20/HLADR_feature_df.csv', index=False)
#
