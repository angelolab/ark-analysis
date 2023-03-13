import os
import warnings

import feather
import pandas as pd
from alpineer import io_utils, misc_utils

from ark.phenotyping import cluster_helpers


def compute_cell_som_cluster_cols_avg(cell_cluster_data, cell_som_cluster_cols,
                                      cell_cluster_col, keep_count=False):
    """For each cell SOM cluster, compute the average expression of all `cell_som_cluster_cols`

    Args:
        cell_cluster_data (pandas.DataFrame):
            The cell data with SOM and/or meta labels, created by `cluster_cells` or
            `cell_consensus_cluster`
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_cluster_col (str):
            Name of the cell cluster column to group by,
            should be `'cell_som_cluster'` or `'cell_meta_cluster'`
        keep_count (bool):
            Whether to include the cell counts or not,
            should only be set to `True` for visualization support

    Returns:
        pandas.DataFrame:
            Contains the average values for each column across cell SOM clusters
    """

    # verify the cell cluster col prefix specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=cell_cluster_col,
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # verify that the cluster columns are valid
    misc_utils.verify_in_list(
        provided_cluster_col=cell_som_cluster_cols,
        cluster_data_valid_cols=cell_cluster_data.columns.values
    )

    # subset the data by columns used for SOM training, as well as the cell SOM assignments
    cell_cluster_data_subset = cell_cluster_data.loc[
        :, list(cell_som_cluster_cols) + [cell_cluster_col]
    ]

    # average each column grouped by the cell cluster column
    mean_count_totals = cell_cluster_data_subset.groupby(cell_cluster_col).mean().reset_index()

    # if keep_count is included, add the count column to the cell table
    if keep_count:
        cell_cluster_totals = cell_cluster_data_subset.groupby(
            cell_cluster_col
        ).size().to_frame('count')
        cell_cluster_totals = cell_cluster_totals.reset_index(drop=True)
        mean_count_totals['count'] = cell_cluster_totals['count']

    return mean_count_totals


def create_c2pc_data(fovs, pixel_data_path, cell_table_path,
                     pixel_cluster_col='pixel_meta_cluster_rename'):
    """Create a matrix with each fov-cell label pair and their SOM pixel/meta cluster counts

    Args:
        fovs (list):
            The list of fovs to subset on
        pixel_data_path (str):
            Path to directory with the pixel data with SOM and meta labels attached.
            Created by `pixel_consensus_cluster`.
        cell_table_path (str):
            Path to the cell table, needs to be created with `Segment_Image_Data.ipynb`
        pixel_cluster_col (str):
            The name of the pixel cluster column to count per cell
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`

    Returns:
        tuple:

        - `pandas.DataFrame`: cell x cluster counts of each pixel SOM/meta cluster per each cell
        - `pandas.DataFrame`: same as above, but normalized by `cell_size`
    """

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # read the cell table data
    cell_table = pd.read_csv(cell_table_path)

    # verify that the user has specified fov, label, and cell_size columns in their cell table
    misc_utils.verify_in_list(
        required_cell_table_cols=['fov', 'label', 'cell_size'],
        provided_cell_table_cols=cell_table.columns.values
    )

    # subset on fov, label, and cell size
    cell_table = cell_table[['fov', 'label', 'cell_size']]

    # convert labels to int type
    cell_table['label'] = cell_table['label'].astype(int)

    # rename cell_table label as segmentation_label for joining purposes
    cell_table = cell_table.rename(columns={'label': 'segmentation_label'})

    # subset on only the fovs the user has specified
    cell_table = cell_table[cell_table['fov'].isin(fovs)]

    # define cell_table columns to subset on for merging
    cell_table_cols = ['fov', 'segmentation_label', 'cell_size']

    for fov in fovs:
        # read in the pixel dataset for the fov
        fov_pixel_data = feather.read_dataframe(
            os.path.join(pixel_data_path, fov + '.feather')
        )

        # create a groupby object that aggregates the segmentation_label and the pixel_cluster_col
        # intermediate step for creating a pivot table, makes it easier
        group_by_cluster_col = fov_pixel_data.groupby(
            ['segmentation_label', pixel_cluster_col]
        ).size().reset_index(name='count')

        # if cluster labels end up as float (can happen with numeric types), convert to int
        if group_by_cluster_col[pixel_cluster_col].dtype == float:
            group_by_cluster_col[pixel_cluster_col] = group_by_cluster_col[
                pixel_cluster_col
            ].astype(int)

        # counts number of pixel SOM/meta clusters per cell
        num_cluster_per_seg_label = group_by_cluster_col.pivot(
            index='segmentation_label', columns=pixel_cluster_col, values='count'
        ).fillna(0).astype(int)

        # renames the columns to have 'pixel_som_cluster_' or 'pixel_meta_cluster_rename_' prefix
        new_columns = [
            '%s_' % pixel_cluster_col + str(c) for c in num_cluster_per_seg_label.columns
        ]
        num_cluster_per_seg_label.columns = new_columns

        # get intersection of the segmentation labels between cell_table_indices
        # and num_cluster_per_seg_label
        cell_table_labels = list(cell_table[cell_table['fov'] == fov]['segmentation_label'])
        cluster_labels = list(num_cluster_per_seg_label.index.values)
        label_intersection = list(set(cell_table_labels).intersection(cluster_labels))

        # subset on the label intersection
        num_cluster_per_seg_label = num_cluster_per_seg_label.loc[label_intersection]
        cell_table_indices = pd.Index(
            cell_table[
                (cell_table['fov'] == fov) &
                (cell_table['segmentation_label'].isin(label_intersection))
            ].index.values
        )

        # combine the data of num_cluster_per_seg_label into cell_table_indices
        num_cluster_per_seg_label = num_cluster_per_seg_label.set_index(cell_table_indices)
        cell_table = cell_table.combine_first(num_cluster_per_seg_label)

    # NaN means the cluster wasn't found in the specified fov-cell pair
    cell_table = cell_table.fillna(0)

    # also produce a cell table with counts normalized by cell_size
    cell_table_norm = cell_table.copy()

    count_cols = [c for c in cell_table_norm.columns if '%s_' % pixel_cluster_col in c]
    cell_table_norm[count_cols] = cell_table_norm[count_cols].div(cell_table_norm['cell_size'],
                                                                  axis=0)

    # reset the indices of cell_table and cell_table_norm to make things consistent
    cell_table = cell_table.reset_index(drop=True)
    cell_table_norm = cell_table_norm.reset_index(drop=True)

    # find columns that are set to all 0
    cell_zero_cols = list(cell_table_norm[count_cols].columns[
        (cell_table_norm[count_cols] == 0).all()
    ].values)

    # filter out these columns (they will cause normalization to fail)
    if len(cell_zero_cols) > 0:
        warnings.warn('Pixel clusters %s do not appear in any cells, removed from analysis' %
                      ','.join(cell_zero_cols))
        cell_table = cell_table.drop(columns=cell_zero_cols)
        cell_table_norm = cell_table_norm.drop(columns=cell_zero_cols)

    return cell_table, cell_table_norm


def add_consensus_labels_cell_table(base_dir, cell_table_path, cell_som_input_data):
    """Adds the consensus cluster labels to the cell table,
    then resaves data to `{cell_table_path}_cell_labels.csv`

    Args:
        base_dir (str):
            The path to the data directory
        cell_table_path (str):
            Path of the cell table, needs to be created with `Segment_Image_Data.ipynb`
        cell_som_input_data (pandas.DataFrame):
            The input data used for SOM training
    """

    # file path validation
    io_utils.validate_paths([cell_table_path])

    # read in the data, ensure sorted by FOV column just in case
    cell_table = pd.read_csv(cell_table_path)

    # for a simpler merge, rename segmentation_label to label in consensus_data
    cell_som_results = cell_som_input_data.rename(
        {'segmentation_label': 'label'}, axis=1
    )

    # merge the cell table with the consensus data to retrieve the meta clusters
    cell_table_merged = cell_table.merge(
        cell_som_results, how='left', on=['fov', 'label']
    )

    # adjust column names and drop consensus data-specific columns
    # NOTE: non-pixel cluster inputs will not have the cell size attribute for normalization
    if 'cell_size_y' in cell_table_merged.columns.values:
        cell_table_merged = cell_table_merged.drop(columns=['cell_size_y'])
        cell_table_merged = cell_table_merged.rename(
            {'cell_size_x': 'cell_size'}, axis=1
        )

    # subset on just the cell table columns plus the meta cluster rename column
    # NOTE: rename cell_meta_cluster_rename to just cell_meta_cluster for simplicity
    cell_table_merged = cell_table_merged[
        list(cell_table.columns.values) + ['cell_meta_cluster_rename']
    ]
    cell_table_merged = cell_table_merged.rename(
        {'cell_meta_cluster_rename': 'cell_meta_cluster'}, axis=1
    )

    # fill any N/A cell_meta_cluster values with 'Unassigned'
    # NOTE: this happens when a cell is so small no pixel clusters are detected inside of them
    cell_table_merged['cell_meta_cluster'] = cell_table_merged['cell_meta_cluster'].fillna(
        'Unassigned'
    )

    # resave cell table with new meta cluster column
    new_cell_table_path = os.path.splitext(cell_table_path)[0] + '_cell_labels.csv'
    cell_table_merged.to_csv(new_cell_table_path, index=False)
