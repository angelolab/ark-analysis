from typing import Generator, List, Union
from skimage.io import imread
from alpineer.image_utils import save_image
from alpineer import io_utils
import os
import re
import shutil
from tqdm.auto import tqdm
import numpy as np
import pathlib
import pandas as pd


def find_and_copy_files(mask_names: List[str], source_folder: Union[str, pathlib.Path],
                        destination_folder: Union[str, pathlib.Path]):
    """
    Creates a new directory of masks for relabeling and cell table generation. Useful if more than
    one mask type is needed for cell table generation. E.g. merged cells and proteopathy objects.

    Args:
        mask_names (List[str]):
            List of mask names to be merged. Can be partial names.
        source_folder (Union[str, pathlib.Path]):
            The parent segmentation folder all masks are found in.
        destination_folder (Union[str, pathlib.Path]):
            New dir where final masks will be copied to.
    """
    # Ensure the destination folder exists, create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through each name in the list
    for mn in mask_names:
        # Compile a regex pattern to match files containing the name anywhere in the file name
        pattern = re.compile(f".*{re.escape(mn)}.*", re.IGNORECASE)

        # Search for files associated with the current name in the source folder using regex
        files_to_copy = []
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if pattern.match(file) and str(destination_folder) not in str(root):
                    files_to_copy.append(os.path.join(root, file))

        # Copy the found files to the destination folder
        for file_path in files_to_copy:
            shutil.copy(file_path, os.path.join(destination_folder, os.path.basename(file_path)))


def renumber_masks(
    mask_dir: Union[pathlib.Path, str]
):
    """
    Relabels all masks in mask tiffs so each label is unique across all mask images
    in entire dataset.

    Args:
        mask_dir (Union[pathlib.Path, str]): Directory that points to parent directory of all
            segmentation masks to be relabeled.
    """
    mask_dir_path = pathlib.Path(mask_dir)
    io_utils.validate_paths(mask_dir_path)

    all_images: Generator[pathlib.Path, None, None] = mask_dir_path.rglob("*.tiff")

    global_unique_labels = 1

    # First pass - get total number of unique masks
    for image in all_images:
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        non_zero_labels: np.ndarray = unique_labels[unique_labels != 0]
        global_unique_labels += len(non_zero_labels)

    all_images: Generator[pathlib.Path, None, None] = mask_dir_path.rglob("*.tiff")

    # Second pass - relabel all masks starting at unique num of masks +1
    for image in all_images:
        img: np.ndarray = imread(image)
        unique_labels: np.ndarray = np.unique(img)
        for label in unique_labels:
            if label != 0:
                img[img == label] = global_unique_labels
                global_unique_labels += 1
        save_image(fname=image, data=img)
    print("Relabeling Complete.")


def create_mantis_project(
    fovs: Union[str, List[str]],
    image_data_dir: Union[str, pathlib.Path],
    segmentation_dir: Union[str, pathlib.Path],
    mantis_dir: Union[str, pathlib.Path],
) -> None:
    """
    Creates a folder for viewing FOVs in Mantis.

    Args:
        fovs (Union[str, List[str]]):
            A list of FOVs to use for creating the mantis project
        image_data_dir (Union[str, pathlib.Path]):
            The path to the directory containing the raw image data.
        segmentation_dir (Union[str, pathlib.Path]):
            The path to the directory containing masks.
        mantis_dir:
            The path to the directory containing housing the ez_seg specific mantis project.
    """
    for fov in tqdm(io_utils.list_folders(image_data_dir, substrs=fovs)):
        shutil.copytree(os.path.join(image_data_dir, fov), dst=os.path.join(mantis_dir, fov))

        for seg_type in io_utils.list_folders(segmentation_dir):
            for mask in io_utils.list_files(os.path.join(segmentation_dir, seg_type), substrs=fov):
                shutil.copy(os.path.join(segmentation_dir, seg_type, mask),
                            dst=os.path.join(mantis_dir, fov)
                            )


def log_creator(variables_to_log: dict, base_dir: str, log_name: str = "config_values.txt"):
    """Logs the variables in `variables_to_log` to the file at `base_dir/log_name`

    Args:
        variables_to_log (dict):
            The name of each variable along with their associated value
        base_dir (str):
            Where the log will be written to
        log_name (str):
            The name of the log file to write the variables to
    """
    # Define the filename for the text file
    output_file = os.path.join(base_dir, log_name)

    # Open the file in write mode and write the variable values
    with open(output_file, "w") as file:
        for variable_name, variable_value in variables_to_log.items():
            file.write(f"{variable_name}: {variable_value}\n")

    print(f"Values saved to {output_file}")


def filter_csvs_by_mask(csv_path_name: Union[str, pathlib.Path], csv_substr_replace: str,
                        column_to_filter: str = "mask_type") -> None:
    """Function to take in and separate a single cell table into multiple
    based on the mask_type parameter.

    Args:
        csv_path_name (Union[str, pathlib.Path]):
            The path to the directory containing the cell table CSVs.
        csv_substr_replace (str):
            The substring in the CSV file name to replace in favor of the mask name
        column_to_filter (str):
            The name of the column to split on, defaults to `"mask_type"`
    """
    # Load the CSV file as a DataFrame (replace 'input.csv' with your CSV file)
    csv_files = io_utils.list_files(csv_path_name, substrs=".csv")
    for item in csv_files:
        if csv_substr_replace not in item:
            continue

        input_csv_file = os.path.join(csv_path_name, item)
        df = pd.read_csv(input_csv_file)

        # Get unique values from the specified column
        filter_values = df[column_to_filter].unique()

        # Create a dictionary to store filtered DataFrames
        filtered_dfs = {}

        # Filter the DataFrame for each unique value and save as separate CSV files
        for filter_value in filter_values:
            filtered_df = df[df[column_to_filter] == filter_value]

            # Define the output CSV file name based on the filtered value
            table_type_str = item.replace(csv_substr_replace, '')
            output_csv_file = os.path.join(
                csv_path_name, ''.join([f'filtered_{filter_value}', table_type_str])
            )

            # Save the filtered DataFrame to a new CSV file
            filtered_df.to_csv(output_csv_file, index=False)

            # Store the filtered DataFrame in the dictionary
            filtered_dfs[filter_value] = filtered_df

    # Print msg
    print("Filtering of csv's complete.")
