import copy
import warnings
from typing import List
import re

import numpy as np
import pandas as pd
import xarray as xr
from alpineer import io_utils, load_utils, misc_utils
from skimage.measure import regionprops, regionprops_table

import ark.settings as settings
from ark.segmentation import segmentation_utils
from ark.segmentation.regionprops_extraction import REGIONPROPS_FUNCTION
from ark.segmentation.signal_extraction import EXTRACTION_FUNCTION


def get_single_compartment_props(segmentation_labels, regionprops_base,
                                 regionprops_single_comp, **kwargs):
    """Gets regionprops features from the provided segmentation labels for a fov

    Based on segmentation labels from a single compartment

    Args:
        segmentation_labels (numpy.ndarray):
            rows x columns matrix of masks
        regionprops_base (list):
            base morphology features directly computed by regionprops to extract for each cell
        regionprops_single_comp (list):
            list of single compartment extra properties derived from regionprops to compute
        **kwargs:
            Arbitrary keyword arguments for compute_extra_props

    Returns:
        pandas.DataFrame:
            Contains the regionprops info (base and derived) for each labeled cell
    """

    # verify that all the regionprops single compartment featues provided actually exist
    # NOTE: in case fast_extraction set to True in calling function, bypass
    if len(regionprops_single_comp) > 0:
        misc_utils.verify_in_list(
            extras_props=regionprops_single_comp,
            props_options=list(REGIONPROPS_FUNCTION.keys())
        )

    # if image is just background, return empty df
    if len(np.unique(segmentation_labels)) < 2:
        output_list = regionprops_base + regionprops_single_comp
        blank_df = pd.DataFrame(columns=output_list)
        return blank_df

    # get the base features
    cell_props = pd.DataFrame(regionprops_table(segmentation_labels,
                                                properties=regionprops_base))

    # define an empty list for each regionprop feature
    cell_props_single_comp = {re: [] for re in regionprops_single_comp}

    # get regionprop info needed for single compartment computations
    props = regionprops(segmentation_labels)

    # generate the required data for each cell
    for prop in props:
        for re in regionprops_single_comp:
            cell_props_single_comp[re].append(REGIONPROPS_FUNCTION[re](prop, **kwargs))

    # convert the dictionary to a DataFrame
    cell_props_single_comp = pd.DataFrame.from_dict(cell_props_single_comp)

    # append the single compartment derived props info to the cell_props DataFrame
    cell_props = pd.concat([cell_props, cell_props_single_comp], axis=1)

    return cell_props


def assign_single_compartment_features(marker_counts, compartment, cell_props, cell_coords,
                                       cell_id, label_id, input_images, regionprops_names,
                                       extraction, **kwargs):
    """Assign computed regionprops features and signal intensity to cell_id in marker_counts

    Args:
        marker_counts (xarray.DataArray):
            xarray containing segmentaed data of cells x markers
        compartment (str):
            either 'whole_cell' or 'nuclear'
        cell_props (pandas.DataFrame):
            regionprops information for each cell
        cell_coords (numpy.ndarray):
            values representing pixels within one cell
        cell_id (int):
            id of the cell
        label_id (int):
            id used to index into cell_props
        input_images (xarray.DataArray):
            rows x columns x channels matrix of imaging data
        regionprops_names (list):
            all of the regionprops features (including derived, except nuclear-specific)
        extraction (str):
            the extraction method to use for signal intensity calculation
        **kwargs:
            arbitrary keyword arguments

    Returns:
        xarray.DataArray:
            the updated marker_counts matrix with data for the specified cell_id and compartment
    """

    # get centroid corresponding to current cell
    kwargs['centroid'] = np.array((
        cell_props.loc[cell_props['label'] == label_id, 'centroid-0'].values,
        cell_props.loc[cell_props['label'] == label_id, 'centroid-1'].values
    )).T

    cell_counts = EXTRACTION_FUNCTION[extraction](cell_coords, input_images, **kwargs)

    # get morphology metrics
    # Filter regionprops_names to only those in cell_props.columns
    filtered_regionprops_names = [rp_name for rp_name in regionprops_names if rp_name in
                                  cell_props.columns]
    current_cell_props = cell_props.loc[cell_props['label'] == label_id,
                                        filtered_regionprops_names]

    # combine marker counts and morphology metrics together
    cell_features = np.concatenate((cell_counts, current_cell_props), axis=None)

    # add counts of each marker to appropriate column
    # Only include the marker_count features up to the last filtered feature.
    marker_counts.loc[
        compartment, cell_id, marker_counts.features[1]:filtered_regionprops_names[-1]
    ] = cell_features

    # add cell size to first column
    marker_counts.loc[compartment, cell_id, marker_counts.features[0]] = cell_coords.shape[0]

    return marker_counts


def assign_multi_compartment_features(marker_counts, regionprops_multi_comp, **kwargs):
    """Assigns features to marker_counts that depend on multiple compartments

    Args:
        marker_counts (xarray.DataArray):
            xarray containing segmentaed data of cells x markers
        regionprops_multi_comp (list):
            list of multi-compartment properties derived from regionprops to compute,
            each value should correspond to a value in REGIONPROPS_FUNCTION
        **kwargs:
            arbitrary keyword arguments

    Returns:
        xarray.DataArray:
            the updated marker_counts matrix with data for the specified cell_id and compartment
    """

    # if no multi regionprops features set, just return the marker counts array as is
    # NOTE: this often happens when fast_extraction set to True in calling function
    if len(regionprops_multi_comp) == 0:
        return marker_counts

    misc_utils.verify_in_list(
        nuclear_props=regionprops_multi_comp,
        props_options=list(REGIONPROPS_FUNCTION.keys())
    )

    for rn in regionprops_multi_comp:
        # if rn is not a feature, then we add a new dimension to hold this regionprop info
        if rn not in marker_counts.features.values:
            rn_fill = np.zeros((marker_counts.shape[0], marker_counts.shape[1], 1))
            rn_arr = xr.DataArray(
                rn_fill,
                coords=[marker_counts.compartments.values, marker_counts.cell_id.values, [rn]],
                dims=['compartments', 'cell_id', 'features']
            )

            # append new dimension to marker_counts
            marker_counts = xr.concat([marker_counts, rn_arr], dim='features')

        # compute the multi-compatment regionprop info
        marker_counts = REGIONPROPS_FUNCTION[rn](marker_counts, **kwargs)

    return marker_counts


def compute_marker_counts(input_images, segmentation_labels, nuclear_counts=False,
                          regionprops_base=copy.deepcopy(settings.REGIONPROPS_BASE),
                          regionprops_single_comp=copy.deepcopy(settings.REGIONPROPS_SINGLE_COMP),
                          regionprops_multi_comp=copy.deepcopy(settings.REGIONPROPS_MULTI_COMP),
                          split_large_nuclei=False, extraction='total_intensity',
                          fast_extraction=False, **kwargs):
    """Extract single cell protein expression data from channel TIFs for a single fov

    Args:
        input_images (xarray.DataArray):
            rows x columns x channels matrix of imaging data
        segmentation_labels (numpy.ndarray):
            rows x columns x compartment matrix of masks
        nuclear_counts (bool):
            boolean flag to determine whether nuclear counts are returned
        regionprops_base (list):
            base morphology features directly computed by regionprops to extract for each cell
        regionprops_single_comp (list):
            list of single compartment extra properties derived from regionprops to compute
        regionprops_multi_comp (list):
            list of multi compartment extra properties derived from regionprops to compute
        split_large_nuclei (bool):
            controls whether nuclei which have portions outside of the cell will get relabeled
        extraction (str):
            extraction function used to compute marker counts
        fast_extraction (bool):
            if set, skips custom regionprops and expensive base regionprops extraction steps
            regardless of other params set
        **kwargs:
            arbitrary keyword arguments for get_cell_props

    Returns:
        xarray.DataArray:
            xarray containing segmented data of cells x markers
    """

    misc_utils.verify_in_list(
        extraction=extraction,
        extraction_options=list(EXTRACTION_FUNCTION.keys())
    )

    if 'coords' not in regionprops_base:
        regionprops_base.append('coords')

    # labels are required
    if 'label' not in regionprops_base:
        regionprops_base.append('label')

    # centroid is required
    if not any(['centroid' in rpf for rpf in regionprops_base]):
        regionprops_base.append('centroid')

    # set regionprops_base to just POST_CHANNEL_COL, coords, and centroid if skip extraction set
    # no additional base regionprops names or custom regionprops properties will be extracted
    if fast_extraction:
        regionprops_base = [settings.POST_CHANNEL_COL, 'coords', 'centroid']
        regionprops_single_comp = []
        regionprops_multi_comp = []

    # enforce post channel column is present and first
    if regionprops_base[0] != settings.POST_CHANNEL_COL:
        if settings.POST_CHANNEL_COL in regionprops_base:
            regionprops_base.remove(settings.POST_CHANNEL_COL)
        regionprops_base.insert(0, settings.POST_CHANNEL_COL)

    # create variable to hold names of returned columns only
    regionprops_names = copy.copy(regionprops_base)
    regionprops_names.remove('coords')

    # centroid returns two columns, need to modify names
    if np.isin('centroid', regionprops_names):
        regionprops_names.remove('centroid')
        regionprops_names += ['centroid-0', 'centroid-1']

    # add the single compartment features to regionprops_names
    regionprops_names.extend(regionprops_single_comp)

    # get all the cell ids
    unique_cell_ids = np.unique(segmentation_labels[..., 0].values)
    unique_cell_ids = unique_cell_ids[np.nonzero(unique_cell_ids)]

    # set the channel features
    channel_features = input_images.channels

    # create labels for array holding channel counts and morphology metrics
    feature_names = np.concatenate((np.array(settings.PRE_CHANNEL_COL), channel_features,
                                    regionprops_names), axis=None)

    # create np.array to hold compartment x cell x feature info
    marker_counts_array = np.zeros((len(segmentation_labels.compartments), len(unique_cell_ids),
                                    len(feature_names)))

    marker_counts = xr.DataArray(copy.copy(marker_counts_array),
                                 coords=[segmentation_labels.compartments,
                                         unique_cell_ids.astype('int'),
                                         feature_names],
                                 dims=['compartments', 'cell_id', 'features'])

    # get the regionprops kwargs
    reg_props = kwargs.get('regionprops_kwargs', {})

    # get regionprops for each cell
    cell_props = get_single_compartment_props(segmentation_labels.loc[:, :, 'whole_cell'].values,
                                              regionprops_base, regionprops_single_comp,
                                              **reg_props)

    if len(unique_cell_ids) == 0:
        fov_name = str(segmentation_labels.fovs.values)
        warnings.warn("No cells found in the following image: {}".format(fov_name))

    if nuclear_counts:
        nuc_labels = segmentation_labels.loc[:, :, 'nuclear'].values

        if split_large_nuclei:
            cell_labels = segmentation_labels.loc[:, :, 'whole_cell'].values
            nuc_labels = \
                segmentation_utils.split_large_nuclei(cell_segmentation_labels=cell_labels,
                                                      nuc_segmentation_labels=nuc_labels,
                                                      cell_ids=unique_cell_ids)

        nuc_props = get_single_compartment_props(nuc_labels,
                                                 regionprops_base, regionprops_single_comp,
                                                 **reg_props)
        if len(nuc_props) == 0:
            fov_name = str(segmentation_labels.fovs.values)
            warnings.warn("No nuclei found in the following image: {}".format(fov_name))

    # get the signal kwargs
    sig_kwargs = kwargs.get('signal_kwargs', {})

    # loop through each cell in mask
    for cell_id in cell_props['label']:
        # get coords corresponding to current cell.
        cell_coords = cell_props.loc[cell_props['label'] == cell_id, 'coords'].values[0]

        # assign properties for whole cell compartment
        marker_counts = assign_single_compartment_features(
            marker_counts, 'whole_cell', cell_props, cell_coords, cell_id, cell_id,
            input_images, regionprops_names, extraction, **sig_kwargs
        )

        if nuclear_counts:
            # get id of corresponding nucleus
            nuc_id = segmentation_utils.find_nuclear_label_id(nuc_segmentation_labels=nuc_labels,
                                                              cell_coords=cell_coords)

            if nuc_id is not None:
                # get the coords of the corresponding nucleus
                nuc_coords = nuc_props.loc[nuc_props['label'] == nuc_id, 'coords'].values[0]

                # assign properties for nuclear compartment
                marker_counts = assign_single_compartment_features(
                    marker_counts, 'nuclear', nuc_props, nuc_coords, cell_id, nuc_id,
                    input_images, regionprops_names, extraction, **sig_kwargs
                )

                # generate properties which involve multiple compartments
                marker_counts = assign_multi_compartment_features(
                    marker_counts, regionprops_multi_comp
                )

                # if regionprops names does not contain multi_comp props then add them
                if not set(regionprops_multi_comp).issubset(regionprops_names):
                    regionprops_names.extend(regionprops_multi_comp)

    return marker_counts


def create_marker_count_matrices(segmentation_labels, image_data, nuclear_counts=False,
                                 split_large_nuclei=False, extraction='total_intensity',
                                 fast_extraction=False, **kwargs):
    """Create a matrix of cells by channels with the total counts of each marker in each cell.

    Args:
        segmentation_labels (xarray.DataArray):
            xarray of shape [fovs, rows, cols, compartment] containing segmentation masks for one
            FOV, potentially across multiple cell compartments
        image_data (xarray.DataArray):
            xarray containing all of the channel data across one FOV
        nuclear_counts (bool):
            boolean flag to determine whether nuclear counts are returned, note that if
            set to True, the compartments coordinate in segmentation_labels must contain 'nuclear'
        split_large_nuclei (bool):
            boolean flag to determine whether nuclei which are larger than their assigned cell
            will get split into two different nuclear objects
        extraction (str):
            extraction function used to compute marker counts.
        fast_extraction (bool):
            if set, skips the custom regionprops and expensive base regionprops extraction steps
        **kwargs:
            arbitrary keyword args for compute_marker_counts

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

    misc_utils.verify_in_list(
        extraction=extraction,
        extraction_options=list(EXTRACTION_FUNCTION.keys())
    )

    misc_utils.verify_same_elements(segmentation_labels_fovs=segmentation_labels.fovs.values,
                                    img_data_fovs=image_data.fovs.values)

    # define the FOV associated with this segmentation label
    fov = segmentation_labels.fovs.values[0]

    # current mask
    label = segmentation_labels.loc[fov, :, :, :]

    # extract the counts per cell for each marker
    marker_counts = compute_marker_counts(image_data.loc[fov, :, :, :], label,
                                          nuclear_counts=nuclear_counts,
                                          split_large_nuclei=split_large_nuclei,
                                          extraction=extraction,
                                          fast_extraction=fast_extraction, **kwargs)

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

    normalized[settings.CELL_LABEL] = normalized[settings.CELL_LABEL].astype(np.int32)
    arcsinh[settings.CELL_LABEL] = arcsinh[settings.CELL_LABEL].astype(np.int32)

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
    arcsinh['fov'] = fov

    return normalized, arcsinh


def generate_cell_table(segmentation_dir, tiff_dir, img_sub_folder="TIFs",
                        is_mibitiff=False, fovs=None, extraction='total_intensity',
                        nuclear_counts=False, split_large_nuclei=False,
                        fast_extraction=False, mask_types=['whole_cell'],
                        add_underscore=True, **kwargs):
    """This function takes the segmented data and computes the expression matrices batch-wise
    while also validating inputs

    Args:
        segmentation_dir (str):
            the path to the directory containing the segmentation labels generated by Mesmer
        tiff_dir (str):
            the name of the directory which contains the single_channel_inputs
        img_sub_folder (str):
            the name of the folder where the TIF images are located
            ignored if is_mibitiff is True
        fovs (list):
            a list of fovs we wish to analyze, if None will default to all fovs
        is_mibitiff (bool):
            a flag to indicate whether or not the base images are MIBItiffs
        extraction (str):
            extraction function used to compute marker counts
        nuclear_counts (bool):
            boolean flag to determine whether nuclear counts are returned, note that if
            set to True, the compartments coordinate in segmentation_labels must contain 'nuclear'
        split_large_nuclei (bool):
            boolean flag to determine whether nuclei which are larger than their assigned cell
            will get split into two different nuclear objects
        fast_extraction (bool):
            if set, skips the custom regionprops and expensive base regionprops extraction steps
        mask_types (list):
            list of masks to extract data for, defaults to ['whole_cell']
        add_underscore (str):
            whether to add '_' before the mask type suffix in file names, defaults to True
        **kwargs:
            arbitrary keyword arguments for signal and regionprops extraction

    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame):
        - size normalized data
        - arcsinh transformed data
    """

    # if no fovs are specified, then load all the fovs
    if fovs is None:
        if is_mibitiff:
            fovs = io_utils.list_files(tiff_dir, substrs=[".tif", ".tiff"])
        else:
            fovs = io_utils.list_folders(tiff_dir)

    # drop file extensions
    fovs = io_utils.remove_file_extensions(fovs)

    misc_utils.verify_in_list(
        extraction=extraction,
        extraction_options=list(EXTRACTION_FUNCTION.keys())
    )

    # get full filenames from given fovs
    # TODO: deprecate filenames support as part of MIBItiff phasing out
    filenames = io_utils.list_files(tiff_dir, substrs=fovs, exact_match=True)

    # sort the fovs
    fovs.sort()
    filenames.sort()

    # define number of FOVs for batch processing
    cohort_len = len(fovs)

    # create the final dfs to store the processed data
    normalized_tables = []
    arcsinh_tables = []

    for fov_index, fov_name in enumerate(fovs):
        if is_mibitiff:
            image_data = load_utils.load_imgs_from_mibitiff(data_dir=tiff_dir,
                                                            mibitiff_files=[filenames[fov_index]])
        else:
            image_data = load_utils.load_imgs_from_tree(data_dir=tiff_dir,
                                                        img_sub_folder=img_sub_folder,
                                                        fovs=[fov_name])

        for mask_type in mask_types:
            if mask_type is None:
                mask_type, mask_suff = 'cell_mask', None
            else:
                mask_suff = '_' + mask_type if add_underscore else mask_type
            # load the segmentation labels in
            fov_mask_name = fov_name + mask_suff + ".tiff" if mask_suff else fov_name + ".tiff"
            current_labels_cell = load_utils.load_imgs_from_dir(data_dir=segmentation_dir,
                                                                files=[fov_mask_name],
                                                                xr_dim_name='compartments',
                                                                xr_channel_names=[mask_type],
                                                                trim_suffix=mask_suff)

            compartments = ['whole_cell']
            segmentation_labels = current_labels_cell.values

            if nuclear_counts and mask_type == 'whole_cell':
                nuclear_file = fov_name + '_nuclear.tiff'
                current_labels_nuc = load_utils.load_imgs_from_dir(data_dir=segmentation_dir,
                                                                   files=[nuclear_file],
                                                                   xr_dim_name='compartments',
                                                                   xr_channel_names=['nuclear'],
                                                                   trim_suffix='_nuclear')
                compartments = ['whole_cell', 'nuclear']
                segmentation_labels = np.concatenate((current_labels_cell.values,
                                                      current_labels_nuc.values), axis=-1)

            current_labels = xr.DataArray(segmentation_labels,
                                          coords=[current_labels_cell.fovs,
                                                  current_labels_cell.rows,
                                                  current_labels_cell.cols,
                                                  compartments],
                                          dims=current_labels_cell.dims)

            # segment the imaging data
            cell_table_size_normalized, cell_table_arcsinh_transformed = create_marker_count_matrices(
                segmentation_labels=current_labels,
                image_data=image_data,
                extraction=extraction,
                nuclear_counts=nuclear_counts,
                split_large_nuclei=split_large_nuclei,
                fast_extraction=fast_extraction,
                **kwargs
            )

            # add mask type column to the data frame
            if mask_type == "final_cells_remaining":
                mask_type_str = "whole_cell"
            else:
                mask_type_str = mask_type
            cell_table_size_normalized['mask_type'] = mask_type_str
            cell_table_arcsinh_transformed['mask_type'] = mask_type_str

            # add to larger dataframe
            normalized_tables.append(cell_table_size_normalized)
            arcsinh_tables.append(cell_table_arcsinh_transformed)

    # now append to the final dfs to return
    combined_cell_table_size_normalized = pd.concat(normalized_tables)
    combined_cell_table_arcsinh_transformed = pd.concat(arcsinh_tables)

    return combined_cell_table_size_normalized, combined_cell_table_arcsinh_transformed


def get_existing_mask_types(fov_names: List[str], mask_names: List[str]) -> List[str]:
    """ Function to strip prefixes from list: fov_names, strip '.tiff' suffix from list: mask names,
    and remove underscore prefixes, returning unique mask values (i.e. categories of masks).

    Args:
        fov_names (List[str]): list of fov names. Matching fov names in mask names will be returned without fov prefix.
        mask_names (List[str]): list of mask names. Mask names will be returned without tiff suffix.
        
    Returns:
            List[str]: Unique mask names (i.e. categories of masks)
    """
    stripped_mask_names = io_utils.remove_file_extensions(mask_names)

    # break fov names into tokens, compare against mask names and return only those mask names that also contain the fov
    result = []
    for prefix in fov_names:
        prefix_tokens = list(filter(bool, re.split("[^a-zA-Z0-9]", prefix)))
        for itemB in stripped_mask_names:
            itemB_tokens = list(filter(bool, re.split("[^a-zA-Z0-9]", itemB)))
            if set(prefix_tokens).issubset(itemB_tokens):
                result.append(itemB[len(prefix):])

    # Remove underscore prefixes and return unique values
    cleaned_result = [item.lstrip('_') for item in result]
    unique_result = list(set(cleaned_result))
    return unique_result