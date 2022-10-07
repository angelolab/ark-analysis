import pathlib
import shutil
from typing import Union

import datasets


class ExampleDataset():
    def __init__(self, dataset: str, overwrite_existing: bool = True, cache_dir: str = None,
                 revision: str = None) -> None:
        """
        Constructs a utility class for downloading and moving the dataset with respect to it's
        various partitions on Hugging Face: https://huggingface.co/datasets/angelolab/ark_example.

        Args:
            dataset (str): The name of the dataset to download. Can be one of `nb1`, `nb2`,
                `nb3`, `nb4`.
            overwrite_existing (bool): A flag to overwrite existing data. Defaults to `True`.
            cache_dir (str, optional): The directory to save the cache dir. Defaults to `None`,
                which internally in Hugging Face defaults to `~/.cache/huggingface/datasets`.
            revision (str, optional): The commit ID from Hugging Face for the dataset. Used for
                internal development only. Allows the user to fetch a commit from a particular
                `revision` (Hugging Face's terminology for branch). Defaults to `None`. This
                defaults to the latest version in the `main` branch.
                (https://huggingface.co/datasets/angelolab/ark_example/tree/main).
        """
        self.dataset = dataset
        self.overwrite_existing = overwrite_existing
        self.cache_dir = cache_dir
        self.revision = revision

        self.path_suffixes = {
            "image_data": "image_data",
            "cell_table": "segmentation/cell_table",
            "deepcell_output": "segmentation/deepcell_output"
        }
        """
        Path suffixes for mapping each downloaded dataset partition to it's appropriate
        relative save directory.
        """

    def download_example_dataset(self):
        """Downloads the example dataset from Hugging Face Hub.
        The following is a link to the dataset used:
        https://huggingface.co/datasets/angelolab/ark_example

        The dataset will be downloaded to the Hugging Face default cache
        `~/.cache/huggingface/datasets`.
        """
        self.dataset_paths = datasets.load_dataset(path="angelolab/ark_example",
                                                   revision=self.revision,
                                                   name=self.dataset,
                                                   cache_dir=self.cache_dir,
                                                   use_auth_token=False)

    def check_downloaded(self, dst_path: pathlib.Path) -> bool:
        """
        Checks to see if the folder for a dataset config already exists in the `save_dir`
        (i.e. `dst_path` is the specific folder for the config.). If the folder exists, and it
        there are no contents, then it'll return True, False otherwise.

        Args:
            dst_path (pathlib.Path): _description_

        Returns:
            bool: _description_
        """
        dst_files = list(dst_path.rglob("*"))

        if len(dst_files) == 0:
            return False
        else:
            return True

    def move_example_dataset(self, move_dir: Union[str, pathlib.Path]):
        """
        Moves the downloaded example data from the `cache_dir` to the `save_dir`.

        Args:
            save_dir (Union[str, pathlib.Path]): The path to save the dataset files in.
        """
        if type(move_dir) is not pathlib.Path:
            move_dir = pathlib.Path(move_dir)

        dataset_names = list(self.dataset_paths[self.dataset].features.keys())

        for ds_n in dataset_names:
            ds_n_suffix = self.path_suffixes[ds_n]

            # The path where the dataset is saved in the Hugging Face Cache post-download,
            # Necessary to copy + move the data from the cache to the user specified `move_dir`.
            dataset_cache_path = pathlib.Path(self.dataset_paths[self.dataset][ds_n][0])
            src_path = dataset_cache_path / ds_n
            dst_path = move_dir / ds_n_suffix

            # Overwrite the existing dataset if specified, or if there is no dataset.
            if self.overwrite_existing or self.check_downloaded(dst_path=dst_path):

                shutil.copytree(src_path, dst_path, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("._*"))


def get_example_dataset(dataset: str, save_dir: Union[str, pathlib.Path],
                        overwrite_existing: bool = True):
    """
    A user facing wrapper function which downloads a specified dataset from Hugging Face,
    and moves it to the specified save directory `save_dir`.
    The dataset may be found here: https://huggingface.co/datasets/angelolab/ark_example


    Args:
        dataset (str): The dataset to download for a particular notebook.
        save_dir (Union[str, pathlib.Path]): The path to save the dataset files in.
        overwrite_existing (bool): The option to overwrite existing configs of the `dataset`
            downloaded. Defaults to True.
    """

    valid_datasets = ["nb1", "nb2", "nb3", "nb4"]

    # Check the appropriate dataset name
    if dataset not in valid_datasets:
        ValueError(f"The dataset <{dataset}> is not one of the valid datasets available.")

    example_dataset = ExampleDataset(dataset=dataset, overwrite_existing=overwrite_existing,
                                     cache_dir=None,
                                     revision="9fecc0ccbb8f2cf1b33172b827f51dfdcf11c149")

    # Download the dataset
    example_dataset.download_example_dataset()

    # Move the dataset over to the save_dir from the user.
    example_dataset.move_example_dataset(move_dir=save_dir)
