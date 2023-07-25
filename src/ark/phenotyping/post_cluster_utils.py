import os
import pathlib
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alpineer import load_utils, misc_utils
from alpineer.settings import EXTENSION_TYPES
from ark import settings

from ark.utils import data_utils, plot_utils


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

    # Make populations a list if it is a string
    populations: List[str] = misc_utils.make_iterable(populations, ignore_str=True)

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
    fig, axes = plt.subplots(pop_num, 1, figsize=[6.4, 2.2 * pop_num], squeeze=False)
    for ax, pop in zip(axes.flat, populations):
        plot_vals = cell_table.loc[cell_table[pop_col] == pop, marker].values
        ax.hist(plot_vals, 50, density=True, facecolor='g', alpha=0.75, range=(0, x_max))
        ax.set_title("Distribution of {} in {}".format(marker, pop))

        if threshold:
            ax.axvline(x=threshold)

    plt.tight_layout()


def create_mantis_project(
    cell_table: pd.DataFrame,
    fovs: List[str],
    seg_dir: Union[pathlib.Path, str],
    mask_dir: Union[pathlib.Path, str],
    image_dir: Union[pathlib.Path, str],
    mantis_dir: Union[pathlib.Path, str],
    pop_col: str = settings.CELL_TYPE,
    fov_col: str = settings.FOV_ID,
    label_col: str = settings.CELL_LABEL,
    seg_suffix_name: str = "_whole_cell.tiff",
    cluster_type: str = "pixel",
) -> None:
    """Creates a complete Mantis Project for viewing cell labels.

    Args:
        cell_table (pd.DataFrame):
            DataFrame of extracted cell features and subtypes.
        fovs (List[str]):
            A list of FOVs to use for creating the project.
        seg_dir (Union[pathlib.Path, str]):
            The path to the directory containing the segmentation images.
        mask_dir (Union[pathlib.Path, str]):
            The path to the directory where the masks will be stored.
        image_dir (Union[pathlib.Path, str]):
            The path to the directory containing the raw image data.
        mantis_dir (Union[pathlib.Path, str]):
            The path to the directory where the mantis project will be created.
        pop_col (str, optional):
            The column name containing the distinct cell populations. Defaults to
            `settings.CELL_TYPE` (`"cell_meta_cluster"`)
        fov_col (str, optional):
            The column name containing the FOV IDs. Defaults to `settings.FOV_ID` (`"fov"`).
        label_col (str, optional):
            The column name containing the cell label. Defaults to `settings.CELL_LABEL`
            (`"label"`).
        seg_suffix_name (str, optional):
            The suffix of the segmentation file and it's file extension. Defaults to
            `"_whole_cell.tiff"`.
        cluster_type (str, optional):
            The type of clustering being done. Defaults to `"pixel"`.
    """

    # verify the type of clustering provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type], valid_cluster_types=["pixel", "cell"]
    )

    # Validate image extension input.
    seg_suffix_ext: str = seg_suffix_name.split(".")[-1]
    misc_utils.verify_in_list(
        seg_suffix_ext=seg_suffix_ext,
        supported_image_extensions=EXTENSION_TYPES["IMAGE"],
    )

    # split the file extension from the suffix name
    seg_suffix_name_no_ext: str = seg_suffix_name.split(".")[0]

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    small_table: pd.DataFrame = cell_table.loc[:, [pop_col, "label", "fov"]]
    # create small df compatible with FOV function

    # generate unique numeric value for each population
    small_table["pop_vals"] = pd.factorize(small_table[pop_col].tolist())[0] + 1

    ccmd_pop = data_utils.CellClusterMaskData(
        data=small_table,
        fov_col=fov_col,
        label_col=label_col,
        cluster_col="pop_vals",
    )

    # label and save the cell mask for each FOV
    for fov in fovs:
        whole_cell_files = [fov + seg_suffix_name]

        # load the segmentation labels in for the FOVs
        label_map = load_utils.load_imgs_from_dir(
            data_dir=seg_dir,
            files=whole_cell_files,
            xr_dim_name="compartments",
            xr_channel_names=[seg_suffix_name_no_ext],
            trim_suffix=seg_suffix_name_no_ext,
        ).loc[fov, ...]

        # use label_cells_by_cluster to create cell masks
        mask_data = data_utils.label_cells_by_cluster(
            fov=fov,
            ccmd=ccmd_pop,
            label_map=label_map,
        )
        # save the cell mask for each FOV -- (saves with ".tiff" extension)
        data_utils.save_fov_mask(
            fov, mask_dir, mask_data, sub_dir=None, name_suffix="_cell_mask"
        )

    # rename the columns of small_table
    mantis_df: pd.DataFrame = small_table.rename(
        {
            "pop_vals": f"{cluster_type}_meta_cluster",
            pop_col: f"{cluster_type}_meta_cluster_rename",
        },
        axis=1,
    )

    # create the mantis project
    plot_utils.create_mantis_dir(
        fovs=fovs,
        mantis_project_path=mantis_dir,
        img_data_path=image_dir,
        mask_output_dir=mask_dir,
        mask_suffix="_cell_mask",
        mapping=mantis_df,
        seg_dir=seg_dir,
        cluster_type=cluster_type,
        img_sub_folder="",
        seg_suffix_name=seg_suffix_name,
    )
