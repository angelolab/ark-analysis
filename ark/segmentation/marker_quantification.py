import copy

import numpy as np
import pandas as pd

import xarray as xr

from skimage.measure import regionprops_table

from ark.utils import io_utils, load_utils, misc_utils, segmentation_utils
from ark.segmentation import signal_extraction

import ark.settings as settings


def compute_marker_counts(input_images, segmentation_labels, nuclear_counts=False,
                          regionprops_features=None, split_large_nuclei=False):
    """Extract single cell protein expression data from channel TIFs for a single fov

    Args:
        input_images (xarray.DataArray):
            rows x columns x channels matrix of imaging data
        segmentation_labels (numpy.ndarray):
            rows x columns x compartment matrix of masks
        nuclear_counts (bool):
            boolean flag to determine whether nuclear counts are returned
        regionprops_features (list):
            morphology features for regionprops to extract for each cell
        split_large_nuclei (bool):
            controls whether nuclei which have portions outside of the cell will get relabeled

    Returns:
        xarray.DataArray:
            xarray containing segmented data of cells x markers
    """

    if regionprops_features is None:
        regionprops_features = ['label', 'area', 'eccentricity', 'major_axis_length',
                                'minor_axis_length', 'perimeter', 'centroid']

    if 'coords' not in regionprops_features:
        regionprops_features.append('coords')

    # labels are required
    if 'label' not in regionprops_features:
        regionprops_features.append('label')

    # enforce post channel column is present and first
    if regionprops_features[0] != settings.POST_CHANNEL_COL:
        if settings.POST_CHANNEL_COL in regionprops_features:
            regionprops_features.remove(settings.POST_CHANNEL_COL)
        regionprops_features.insert(0, settings.POST_CHANNEL_COL)

    # create variable to hold names of returned columns only
    regionprops_names = copy.copy(regionprops_features)
    regionprops_names.remove('coords')

    # centroid returns two columns, need to modify names
    if np.isin('centroid', regionprops_names):
        regionprops_names.remove('centroid')
        regionprops_names += ['centroid-0', 'centroid-1']

    unique_cell_ids = np.unique(segmentation_labels[..., 0].values)
    unique_cell_ids = unique_cell_ids[np.nonzero(unique_cell_ids)]

    # create labels for array holding channel counts and morphology metrics
    feature_names = np.concatenate((np.array(settings.PRE_CHANNEL_COL), input_images.channels,
                                    regionprops_names), axis=None)

    # create np.array to hold compartment x cell x feature info
    marker_counts_array = np.zeros((len(segmentation_labels.compartments), len(unique_cell_ids),
                                    len(feature_names)))

    marker_counts = xr.DataArray(copy.copy(marker_counts_array),
                                 coords=[segmentation_labels.compartments,
                                         unique_cell_ids.astype('int'),
                                         feature_names],
                                 dims=['compartments', 'cell_id', 'features'])

    # get regionprops for each cell
    cell_props = pd.DataFrame(regionprops_table(segmentation_labels.loc[:, :, 'whole_cell'].values,
                                                properties=regionprops_features))

    if nuclear_counts:
        nuc_labels = segmentation_labels.loc[:, :, 'nuclear'].values

        if split_large_nuclei:
            cell_labels = segmentation_labels.loc[:, :, 'whole_cell'].values
            nuc_labels = \
                segmentation_utils.split_large_nuclei(cell_segmentation_labels=cell_labels,
                                                      nuc_segmentation_labels=nuc_labels,
                                                      cell_ids=unique_cell_ids)

        nuc_props = pd.DataFrame(regionprops_table(nuc_labels, properties=regionprops_features))

    # TODO: There's some repeated code here, maybe worth refactoring? Maybe not
    # loop through each cell in mask
    for cell_id in cell_props['label']:
        # get coords corresponding to current cell.
        cell_coords = cell_props.loc[cell_props['label'] == cell_id, 'coords'].values[0]

        # calculate the total signal intensity within cell
        cell_counts = signal_extraction.default_extraction(cell_coords, input_images)

        # get morphology metrics
        current_cell_props = cell_props.loc[cell_props['label'] == cell_id, regionprops_names]

        # combine marker counts and morphology metrics together
        cell_features = np.concatenate((cell_counts, current_cell_props), axis=None)

        # add counts of each marker to appropriate column
        marker_counts.loc['whole_cell', cell_id, marker_counts.features[1]:] = cell_features

        # add cell size to first column
        marker_counts.loc['whole_cell', cell_id, marker_counts.features[0]] = cell_coords.shape[0]

        if nuclear_counts:
            # get id of corresponding nucleus
            nuc_id = segmentation_utils.find_nuclear_label_id(nuc_segmentation_labels=nuc_labels,
                                                              cell_coords=cell_coords)

            if nuc_id is not None:
                # get coordinates of corresponding nucleus
                nuc_coords = nuc_props.loc[nuc_props['label'] == nuc_id, 'coords'].values[0]

                # extract nuclear signal
                nuc_counts = signal_extraction.default_extraction(nuc_coords, input_images)

                # get morphology metrics
                current_nuc_props = nuc_props.loc[
                    nuc_props['label'] == nuc_id, regionprops_names]

                # combine marker counts and morphology metrics together
                nuc_features = np.concatenate((nuc_counts, current_nuc_props), axis=None)

                # add counts of each marker to appropriate column
                marker_counts.loc['nuclear', cell_id, marker_counts.features[1]:] = nuc_features

                # add cell size to first column
                marker_counts.loc['nuclear', cell_id, marker_counts.features[0]] = \
                    nuc_coords.shape[0]

    return marker_counts


def create_marker_count_matrices(segmentation_labels, image_data, nuclear_counts=False,
                                 split_large_nuclei=False):
    """Create a matrix of cells by channels with the total counts of each marker in each cell.

    Args:
        segmentation_labels (xarray.DataArray):
            xarray of shape [fovs, rows, cols, compartment] containing segmentation masks for each
            fov, potentially across multiple cell compartments
        image_data (xarray.DataArray):
            xarray containing all of the channel data across all FOVs
        nuclear_counts (bool):
            boolean flag to determine whether nuclear counts are returned, note that if
            set to True, the compartments coordinate in segmentation_labels must contain 'nuclear'
        split_large_nuclei (bool):
            boolean flag to determine whether nuclei which are larger than their assigned cell
            will get split into two different nuclear objects

    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame):
            - marker counts per cell normalized by cell size
            - arcsinh transformation of the above
    """
    if type(segmentation_labels) is not xr.DataArray:
        raise ValueError("Incorrect data type for segmentation_labels, expecting xarray")

    if type(image_data) is not xr.DataArray:
        raise ValueError("Incorrect data type for image_data, expecting xarray")

    if nuclear_counts:
        misc_utils.verify_in_list(
            nuclear_label='nuclear',
            compartment_names=segmentation_labels.compartments.values
        )

    misc_utils.verify_same_elements(segmentation_labels_fovs=segmentation_labels.fovs.values,
                                    img_data_fovs=image_data.fovs.values)

    # initialize data frames
    normalized_data = pd.DataFrame()
    arcsinh_data = pd.DataFrame()

    # loop over each fov in the dataset
    for fov in segmentation_labels.fovs.values:
        print("extracting data from {}".format(fov))

        # current mask
        segmentation_label = segmentation_labels.loc[fov, :, :, :]

        # extract the counts per cell for each marker
        marker_counts = compute_marker_counts(image_data.loc[fov, :, :, :], segmentation_label,
                                              nuclear_counts=nuclear_counts,
                                              split_large_nuclei=split_large_nuclei)

        # normalize counts by cell size
        marker_counts_norm = segmentation_utils.transform_expression_matrix(marker_counts,
                                                                            transform='size_norm')

        # arcsinh transform the data
        marker_counts_arcsinh = segmentation_utils.transform_expression_matrix(marker_counts_norm,
                                                                               transform='arcsinh')

        # add data from each fov to array
        normalized = pd.DataFrame(data=marker_counts_norm.loc['whole_cell', :, :].values,
                                  columns=marker_counts_norm.features)

        arcsinh = pd.DataFrame(data=marker_counts_arcsinh.values[0, :, :],
                               columns=marker_counts_arcsinh.features)

        if nuclear_counts:
            # append nuclear counts pandas array with modified column name
            nuc_column_names = [feature + '_nuclear' for feature in marker_counts.features.values]

            # add nuclear counts to size normalized data
            normalized_nuc = pd.DataFrame(data=marker_counts_norm.loc['nuclear', :, :].values,
                                          columns=nuc_column_names)
            normalized = pd.concat((normalized, normalized_nuc), axis=1)

            # add nuclear counts to arcsinh transformed data
            arcsinh_nuc = pd.DataFrame(data=marker_counts_arcsinh.loc['nuclear', :, :].values,
                                       columns=nuc_column_names)
            arcsinh = pd.concat((arcsinh, arcsinh_nuc), axis=1)

        # add column for current fov
        normalized['fov'] = fov
        normalized_data = normalized_data.append(normalized)

        arcsinh['fov'] = fov
        arcsinh_data = arcsinh_data.append(arcsinh)

    return normalized_data, arcsinh_data


def generate_cell_table(segmentation_labels, tiff_dir, img_sub_folder,
                        is_mibitiff=False, fovs=None, batch_size=5, dtype="int16"):
    """
    This function takes the segmented data and computes the expression matrices batch-wise
    while also validating inputs

    Args:
        segmentation_labels (xarray.DataArray):
            an xarray with the segmented data
        tiff_dir (str):
            the name of the directory which contains the single_channel_inputs
        img_sub_folder (str):
            the name of the folder where the TIF images are located
        fovs (list):
            a list of fovs we wish to analyze, if None will default to all fovs
        is_mibitiff (bool):
            a flag to indicate whether or not the base images are MIBItiffs
        batch_size (int):
            how large we want each of the batches of fovs to be when computing, adjust as
            necessary for speed and memory considerations
        dtype (str/type):
            data type of base images
    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame):

        - size normalized data
        - arcsinh transformed data
    """

    # if no fovs are specified, then load all the fovs
    if fovs is None:
        # handle mibitiffs with an assumed file structure
        if is_mibitiff:
            filenames = io_utils.list_files(tiff_dir, substrs=['.tif'])
            fovs = io_utils.extract_delimited_names(filenames, delimiter=None)
        # otherwise assume the tree-like directory as defined for tree loading
        else:
            filenames = io_utils.list_folders(tiff_dir)
            fovs = filenames

    # check segmentation_labels for given fovs (img loaders will fail otherwise)
    misc_utils.verify_in_list(fovs=fovs,
                              segmentation_labels_fovs=segmentation_labels['fovs'].values)

    # get full filenames from given fovs
    filenames = io_utils.list_files(tiff_dir, substrs=fovs)

    # sort the fovs
    fovs.sort()
    filenames.sort()

    # defined some vars for batch processing
    cohort_len = len(fovs)

    # create the final dfs to store the processed data
    combined_cell_table_size_normalized = pd.DataFrame()
    combined_cell_table_arcsinh_transformed = pd.DataFrame()

    # iterate over all the batches
    for batch_names, batch_files in zip(
        [fovs[i:i + batch_size] for i in range(0, cohort_len, batch_size)],
        [filenames[i:i + batch_size] for i in range(0, cohort_len, batch_size)]
    ):
        # and extract the image data for each batch
        if is_mibitiff:
            image_data = load_utils.load_imgs_from_mibitiff(data_dir=tiff_dir,
                                                            mibitiff_files=batch_files,
                                                            dtype=dtype)
        else:
            image_data = load_utils.load_imgs_from_tree(data_dir=tiff_dir,
                                                        img_sub_folder=img_sub_folder,
                                                        fovs=batch_names,
                                                        dtype=dtype)

        # as well as the labels corresponding to each of them
        current_labels = segmentation_labels.loc[batch_names, :, :, :]

        # segment the imaging data
        cell_table_size_normalized, cell_table_arcsinh_transformed = create_marker_count_matrices(
            segmentation_labels=current_labels,
            image_data=image_data
        )

        # now append to the final dfs to return
        combined_cell_table_size_normalized = combined_cell_table_size_normalized.append(
            cell_table_size_normalized
        )
        combined_cell_table_arcsinh_transformed = combined_cell_table_arcsinh_transformed.append(
            cell_table_arcsinh_transformed
        )

    return combined_cell_table_size_normalized, combined_cell_table_arcsinh_transformed
