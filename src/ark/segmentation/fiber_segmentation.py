import itertools
import os
import pathlib
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
from matplotlib import colormaps
import natsort as ns
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.io as io
from alpineer import image_utils, io_utils, load_utils, misc_utils
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.exposure import equalize_adapthist
from skimage.filters import frangi, sobel, threshold_multiotsu
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from tqdm.auto import tqdm

from ark import settings
from ark.utils.plot_utils import set_minimum_color_for_colormap


def plot_fiber_segmentation_steps(data_dir, fov_name, fiber_channel, img_sub_folder=None, blur=2,
                                  contrast_scaling_divisor=128, fiber_widths=range(1, 10, 2),
                                  ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                                  img_cmap="bone", labels_cmap="cool"):
    """Plots output from each fiber segmentation step for single FoV

    Args:
        data_dir (str | PathLike):
            Folder containing dataset
        fov_name (str):
            Name of test FoV
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen
        img_sub_folder (str | NoneType):
            Whether to expect image subfolder in `data_dir`.  If no subfolder, set to None.
        blur (float):
            Preprocessing gaussian blur radius
        contrast_scaling_divisor (int):
            Roughly speaking, the average side length of a fibers bounding box.  This argument
            controls the local contrast enhancement operation, which helps differentiate dim
            fibers from only slightly more dim backgrounds.  This should always be a power of two.
        fiber_widths (Iterable):
            Widths of fibers to filter for.  Be aware that adding larger fiber widths can join
            close, narrow branches into one thicker fiber.
        ridge_cutoff (float):
            Threshold for ridge inclusion post-frangi filtering.
        sobel_blur (float):
            Gaussian blur radius for sobel driven elevation map creation
        min_fiber_size (int):
            Minimum area of fiber object
        img_cmap (matplotlib.Colormap):
            Matplotlib colormap to use for (non-labeled) images
        labels_cmap (matplotlib.Colormap):
            Base matplotlib colormap to use for labeled images.  This will only be applied to the
            non-zero labels, with the zero-region being colored black.
    """
    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    io_utils.validate_paths(data_dir)
    misc_utils.verify_in_list(fiber_channel=[fiber_channel],
                              all_channels=io_utils.remove_file_extensions(
                                  io_utils.list_files(
                                      os.path.join(data_dir, fov_name, img_sub_folder)
                                  )
    )
    )

    data_xr = load_utils.load_imgs_from_tree(
        data_dir, img_sub_folder, fovs=[fov_name], channels=[fiber_channel]
    )

    channel_data = data_xr.loc[fov_name, :, :, fiber_channel].values

    _, axes = plt.subplots(3, 3)
    img_cmap = colormaps[img_cmap]
    labels_cmap = colormaps[labels_cmap]

    axes[0, 0].imshow(channel_data, cmap=img_cmap)
    axes[0, 0].set_title(f"{fov_name} {fiber_channel} raw image")

    blurred = ndi.gaussian_filter(channel_data.astype('float'), sigma=blur)
    axes[0, 1].imshow(blurred, cmap=img_cmap)
    axes[0, 1].set_title(f"Gaussian Blur, sigma={blur}")

    contrast_adjusted = equalize_adapthist(
        blurred / np.max(blurred),
        kernel_size=channel_data.shape[0] / contrast_scaling_divisor
    )
    axes[0, 2].imshow(contrast_adjusted, cmap=img_cmap)
    axes[0, 2].set_title(f"Contrast Adjuisted, CSD={contrast_scaling_divisor}")

    ridges = frangi(contrast_adjusted, sigmas=fiber_widths, black_ridges=False)*10000

    axes[1, 0].imshow(ridges, cmap=img_cmap)
    axes[1, 0].set_title("Frangi Filter")

    distance_transformed = ndi.gaussian_filter(
        distance_transform_edt(ridges > ridge_cutoff),
        sigma=1
    )
    axes[1, 1].imshow(distance_transformed, cmap=img_cmap)
    axes[1, 1].set_title(f"Ridges Filtered, ridge_cutoff={ridge_cutoff}")

    # watershed setup
    threshed = np.zeros_like(distance_transformed)
    thresholds = threshold_multiotsu(distance_transformed, classes=3)

    threshed[distance_transformed < thresholds[0]] = 1
    threshed[distance_transformed > thresholds[1]] = 2
    axes[1, 2].imshow(threshed, cmap=img_cmap)
    axes[1, 2].set_title("Watershed thresholding")

    elevation_map = sobel(
        ndi.gaussian_filter(distance_transformed, sigma=sobel_blur)
    )
    axes[2, 0].imshow(elevation_map, cmap=img_cmap)
    axes[2, 0].set_title(f"Sobel elevation map, sobel_blur={sobel_blur}")

    # build label color map
    transparent_cmap = set_minimum_color_for_colormap(labels_cmap)

    segmentation = watershed(elevation_map.astype(np.int32), threshed.astype(np.int32)) - 1

    labeled, _ = ndi.label(segmentation)
    axes[2, 1].imshow(labeled, cmap=transparent_cmap)
    axes[2, 1].set_title("Unfiltered segmentation")

    labeled_filtered = remove_small_objects(labeled, min_size=min_fiber_size) * segmentation
    axes[2, 2].imshow(labeled_filtered, cmap=transparent_cmap)
    axes[2, 2].set_title(f"Filtered segmentation, min_fiber_size={min_fiber_size}")

    for ax in axes.reshape(-1):
        ax.axis('off')


def run_fiber_segmentation(data_dir, fiber_channel, out_dir, img_sub_folder=None,
                           csv_compression: Optional[Dict[str, str]] = None, **kwargs):
    """Segments fibers one FOV at a time

    Args:
        data_dir (str | PathLike):
            Folder containing dataset
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen.
        out_dir (str | PathLike):
            Directory to save fiber object labels and table.
        img_sub_folder (str | NoneType):
            Image subfolder name in `data_dir`. If there is not subfolder, set this to None.
        csv_compression (Optional[Dict[str, str]]): Dictionary of compression arguments to pass
            when saving csvs. See :meth:`to_csv <pandas.DataFrame.to_csv>` for details.
        **kwargs:
            Keyword arguments for `segment_fibers`

    Returns:
        pd.DataFrame:
         - Dataframe containing the fiber objects and their properties
    """

    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    io_utils.validate_paths([data_dir, out_dir])

    fovs = ns.natsorted(io_utils.list_folders(data_dir))
    misc_utils.verify_in_list(fiber_channel=[fiber_channel],
                              all_channels=io_utils.remove_file_extensions(
                                  io_utils.list_files(
                                      os.path.join(data_dir, fovs[0], img_sub_folder)
                                  )
    )
    )

    fiber_object_table = []

    with tqdm(total=len(fovs), desc="Fiber Segmentation", unit="FOVs") \
            as fibseg_progress:
        for fov in fovs:
            fibseg_progress.set_postfix(FOV=fov)

            subset_xr = load_utils.load_imgs_from_tree(
                data_dir, img_sub_folder, fovs=fov, channels=[fiber_channel]
            )
            # run fiber segmentation on the FOV
            subtable = segment_fibers(subset_xr, fiber_channel, out_dir, fov, save_csv=False,
                                      **kwargs)
            fiber_object_table.append(subtable)

            # update progress bar
            fibseg_progress.update(1)

    fiber_object_table = pd.concat(fiber_object_table)

    # append fiber knn alignment and save table to csv
    if len(fiber_object_table) > 0:
        fiber_object_table = calculate_fiber_alignment(fiber_object_table)
    fiber_object_table.to_csv(os.path.join(out_dir, 'fiber_object_table.csv'), index=False,
                              compression=csv_compression)

    return fiber_object_table


def calculate_fiber_alignment(fiber_object_table, k=4, axis_thresh=2):
    """ Calculates an alignment score for each fiber in an image. Based on the angle difference of
    the fiber compared to it's k nearest neighbors.

    Args:
        fiber_object_table (pd.DataFrame):
            dataframe containing the fiber objects and their properties (fov, label, alignment,
             centroid-0, centroid-1, major_axis_length, minor_axis_length)
        k (int):
            number of neighbors to check alignment difference for
        axis_thresh (int):
            threshold for how much longer the length of the fiber must be compared to the width

    Returns:
        pd.DataFrame:
         - Dataframe with the alignment scores appended
    """

    fovs = np.unique(fiber_object_table.fov)
    fov_data = []

    # process one fov at a time
    for fov in fovs:
        fov_fiber_table = fiber_object_table[fiber_object_table.fov == fov]

        # only grab fibers of specified length to width ratio
        filtered_lengths = fov_fiber_table[(fov_fiber_table['major_axis_length'].values /
                                            fov_fiber_table['minor_axis_length'].values)
                                           >= axis_thresh]
        filtered_lengths = filtered_lengths.reset_index()

        # create a distance matrix between fiber centroids
        centroids = np.vstack((filtered_lengths['centroid-0'].values,
                               filtered_lengths['centroid-1'].values)).T
        fiber_dist_mat = cdist(centroids, centroids)

        # compute alignment scores for each individual fiber
        fiber_scores = []
        for indx, angle in enumerate(filtered_lengths.orientation):
            # find index for smallest distances, excluding itself
            indy = fiber_dist_mat[indx, :].argsort()[1:1+k]
            neighbor_angles = filtered_lengths.orientation[indy]
            fiber_scores.append((np.sqrt(np.sum((neighbor_angles - angle) ** 2)) / k))

        fov_alignments = pd.DataFrame(
            zip([fov] * len(fiber_scores), filtered_lengths.label, fiber_scores),
            columns=['fov', 'label', 'alignment_score'])
        fov_data.append(fov_alignments)

    # append alignment score to fiber object table
    alignment_data = pd.concat(fov_data)
    fiber_object_table_adj = fiber_object_table.merge(alignment_data, 'left')

    return fiber_object_table_adj


def segment_fibers(data_xr, fiber_channel, out_dir, fov, blur=2, contrast_scaling_divisor=128,
                   fiber_widths=range(1, 10, 2), ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                   object_properties=settings.FIBER_OBJECT_PROPS, save_csv=True, debug=False):
    """ Segments fiber objects from image data

    Args:
        data_xr (xr.DataArray):
            Multiplexed image data in (fov, x, y, channel) format
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen.
        out_dir (str | PathLike):
            Directory to save fiber object labels and table.
        fov (str):
            name of the fov being processed
        blur (float):
            Preprocessing gaussian blur radius
        contrast_scaling_divisor (int):
            Roughly speaking, the average side length of a fibers bounding box.  This argument
            controls the local contrast enhancement operation, which helps differentiate dim
            fibers from only slightly more dim backgrounds.  This should always be a power of two.
        fiber_widths (Iterable):
            Widths of fibers to filter for.  Be aware that adding larger fiber widths can join
            close, narrow branches into one thicker fiber.
        ridge_cutoff (float):
            Threshold for ridge inclusion post-frangi filtering.
        sobel_blur (float):
            Gaussian blur radius for sobel driven elevation map creation
        min_fiber_size (int):
            Minimum area of fiber object
        object_properties (Iterable[str]):
            Properties to compute, any keyword for region props may be used.  Defaults are:
             - major_axis_length
             - minor_axis_length
             - orientation
             - centroid
             - label
             - eccentricity
             - euler_number
        save_csv (bool):
            Whether or not to save csv of fiber objects
        debug (bool):
            Save intermediate preprocessing steps

    Returns:
        pd.DataFrame:
         - Dataframe containing the fiber objects and their properties
    """
    channel_xr = data_xr.loc[:, :, :, fiber_channel]
    fov_len = channel_xr.shape[1]

    if debug:
        debug_path = os.path.join(out_dir, '_debug')
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)

    fiber_channel_data = channel_xr.loc[fov, :, :].values.astype('float')

    blurred = ndi.gaussian_filter(fiber_channel_data, sigma=blur)

    # local contrast enhancement
    contrast_adjusted = equalize_adapthist(
        blurred / np.max(blurred),
        kernel_size=fov_len / contrast_scaling_divisor
    )

    # frangi filtering
    ridges = frangi(contrast_adjusted, sigmas=fiber_widths, black_ridges=False)*10000

    # remove image intensity influence for watershed setup
    distance_transformed = ndi.gaussian_filter(
        distance_transform_edt(ridges > ridge_cutoff),
        sigma=1
    )

    # watershed setup
    threshed = np.zeros_like(distance_transformed)
    thresholds = threshold_multiotsu(distance_transformed, classes=3)

    threshed[distance_transformed < thresholds[0]] = 1
    threshed[distance_transformed > thresholds[1]] = 2

    elevation_map = sobel(
        ndi.gaussian_filter(distance_transformed, sigma=sobel_blur)
    )

    segmentation = watershed(elevation_map.astype(np.int32), threshed.astype(np.int32)) - 1

    labeled, _ = ndi.label(segmentation)

    labeled_filtered = remove_small_objects(labeled, min_size=min_fiber_size) * segmentation

    if debug:
        image_utils.save_image(os.path.join(debug_path, f'{fov}_thresholded.tiff'),
                               threshed)
        image_utils.save_image(os.path.join(debug_path, f'{fov}_ridges_thresholded.tiff'),
                               distance_transformed)
        image_utils.save_image(os.path.join(debug_path, f'{fov}_frangi_filter.tiff'),
                               ridges)
        image_utils.save_image(os.path.join(debug_path, f'{fov}_contrast_adjusted.tiff'),
                               contrast_adjusted)

    image_utils.save_image(os.path.join(out_dir, f'{fov}_fiber_labels.tiff'), labeled_filtered)

    fiber_object_table = regionprops_table(labeled_filtered, properties=object_properties)

    fiber_object_table = pd.DataFrame(fiber_object_table)
    fiber_object_table.insert(0, settings.FOV_ID, fov)

    if save_csv:
        fiber_object_table.to_csv(os.path.join(out_dir, 'fiber_object_table.csv'))

    return fiber_object_table


def calculate_density(fov_fiber_table, total_pixels):
    """ Calculates both pixel area and fiber number based densities.
    pixel based = fiber pixel area / total image area
    fiber number based = number of fibers / total image area

    Args:
        fov_fiber_table (pd.DataFrame):
            the array representation of the fiber segmented mask for an image
        total_pixels (int):
            area of the image

    Returns:
        tuple (float, float):
         - returns the both densities scaled up by 100
    """

    fiber_num = len(np.unique(fov_fiber_table.label))
    fiber_density = fiber_num / total_pixels

    pixel_sum = np.sum(fov_fiber_table['area'].values)
    pixel_density = pixel_sum / total_pixels

    return pixel_density * 100, fiber_density * 100


def generate_tile_stats(fov_table, fov_fiber_img, fov_length, tile_length, min_fiber_num,
                        save_dir, save_tiles):
    """ Calculates the tile level statistics for alignment, length, and density.

        Args:
            fov_table (pd.DataFrame):
                dataframe containing the fiber objects and their properties (fov, label, alignment,
                centroid-0, centroid-1, major_axis_length, minor_axis_length)
            fov_fiber_img (np.array):
                represents the fiber mask
            fov_length (int):
                length of the image
            tile_length (int):
                length of tile size, must be a factor of the total image size (default 512)
            min_fiber_num (int):
                the amount of fibers to get tile statistics calculated, if not then NaN (default 5)
            save_dir (str):
                directory where to save tiled image folder to
            save_tiles (bool):
                whether to save cropped images (default to False)

        Returns:
            pd.DataFrame:
             - a dataframe specifying each tile in the image and its calculated stats
        """

    fov_table = fov_table.reset_index(drop=True)
    fov = fov_table.fov[0]
    alignment, pixel_density, fiber_density, tile_stats = [], [], [], []
    fov_list, tile_x, tile_y = [], [], []

    # other tile stats
    properties = ["major_axis_length", "minor_axis_length", "orientation", "area",
                  "eccentricity", "euler_number"]

    # create tiles based on provided tile_length
    for i, j in itertools.product(
            range(int(fov_length / tile_length)), range(int(fov_length / tile_length))):
        y_range = (i * tile_length, (i + 1) * tile_length)
        x_range = (j * tile_length, (j + 1) * tile_length)

        fov_list.append(fov)
        tile_x.append(x_range[0])
        tile_y.append(y_range[0])

        if save_tiles:
            tile_fiber_img = fov_fiber_img[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            tile_fiber_img[tile_fiber_img > 0] = 1
            if not os.path.exists(os.path.join(save_dir, fov)):
                os.makedirs(os.path.join(save_dir, fov))
            io.imsave(os.path.join(save_dir, fov, f'tile_{y_range[0]},{x_range[0]}.tiff'),
                      tile_fiber_img, check_contrast=False)

        # subset table for only fibers within the tile coords
        tile_table = fov_table[np.logical_and(
            fov_table['centroid-0'] >= y_range[0], fov_table['centroid-0'] < y_range[1])]
        tile_table = tile_table[np.logical_and(
            tile_table['centroid-1'] >= x_range[0], tile_table['centroid-1'] < x_range[1])]

        # tile must have a certain number of fibers to receive values, otherwise NaN
        avg_alignment, p_density, f_density = [np.nan]*3
        tile_avgs = np.array([np.nan]*len(properties))

        if len(tile_table) >= min_fiber_num:
            # alignment
            align_scores = tile_table['alignment_score'].values
            align_scores = align_scores[~np.isnan(align_scores)]
            avg_alignment = np.mean(align_scores) if len(align_scores) >= min_fiber_num else np.nan

            # take the average of the properties
            tile_avgs = tile_table[properties].mean().array

            # density
            p_density, f_density = calculate_density(tile_table, tile_length ** 2)

        alignment.append(avg_alignment)
        pixel_density.append(p_density)
        fiber_density.append(f_density)
        tile_stats.append(tile_avgs)

    tile_stats = np.vstack(tile_stats)

    fov_tile_stats = pd.DataFrame(zip(
        fov_list, tile_y, tile_x, pixel_density, fiber_density, alignment),
        columns=['fov', 'tile_y', 'tile_x', 'pixel_density', 'fiber_density',
                 'avg_alignment_score'])

    for i, metric in enumerate(properties):
        fov_tile_stats[f"avg_{metric}"] = tile_stats.T[i]

    return fov_tile_stats


def generate_summary_stats(fiber_object_table, fibseg_dir, tile_length=512, min_fiber_num=5,
                           save_tiles=False):
    """ Calculates the fov level and tile level statistics for alignment, length, and density.
    Saves them to separate csvs.

    Args:
        fiber_object_table (pd.DataFrame):
            dataframe containing the fiber objects and their properties (fov, label, alignment,
            centroid-0, centroid-1, major_axis_length, minor_axis_length)
        fibseg_dir (string):
            path to directory containing the fiber segmentation masks
        tile_length (int):
            length of tile size, must be a factor of the total image size (default 512)
        min_fiber_num (int):
            the amount of fibers to get tile statistics calculated, if not then NaN (default 5)
        save_tiles (bool):
            whether to save cropped images (default to False)

    Returns:
        tuple (pd.DataFrame, pd.DataFrame):
         - returns the both fov and tile stats
    """

    io_utils.validate_paths(fibseg_dir)
    # this makes sure tile length is a factor of 1024 and 2048
    if 1024 % tile_length != 0:
        raise ValueError("Tile length must be a factor of the minimum image size.")

    save_dir = os.path.join(fibseg_dir, f'tile_stats_{tile_length}')
    fovs = np.unique(fiber_object_table.fov)
    fov_stats, tile_stats = [], []
    fov_alignment, fov_pixel_density, fov_fiber_density, fov_avg_stats = [], [], [], []

    # stat list
    properties = ["major_axis_length", "minor_axis_length", "orientation", "area",
                  "eccentricity", "euler_number", "alignment_score"]

    # get fov level and tile level stats for each image
    for fov in fovs:
        fov_fiber_img = io.imread(os.path.join(fibseg_dir, fov + '_fiber_labels.tiff'))
        fov_length = fov_fiber_img.shape[0]
        fov_table = fiber_object_table[fiber_object_table.fov == fov]

        # take the average of the fov level properties
        avg_stats = fov_table[properties].mean().array

        # density
        fov_p_density, fov_f_density = calculate_density(fov_table, fov_length**2)
        fov_pixel_density.append(fov_p_density)
        fov_fiber_density.append(fov_f_density)

        # tile level stats
        fov_tile_stats = generate_tile_stats(fov_table, fov_fiber_img, fov_length, tile_length,
                                             min_fiber_num, save_dir, save_tiles)

        fov_avg_stats.append(avg_stats)
        tile_stats.append(fov_tile_stats)

    fov_stats = pd.DataFrame({
        'fov': fovs,
        'pixel_density': fov_pixel_density,
        'fiber_density': fov_fiber_density
    })

    fov_prop_stats = np.vstack(fov_avg_stats)
    for i, metric in enumerate(properties):
        fov_stats[f"avg_{metric}"] = fov_prop_stats.T[i]

    fov_stats.to_csv(os.path.join(fibseg_dir, f'fiber_stats_table.csv'), index=False)

    tile_stats = pd.concat(tile_stats)
    tile_stats.to_csv(os.path.join(save_dir, f'fiber_stats_table-tile_{tile_length}.csv'),
                      index=False)

    return fov_stats, tile_stats
