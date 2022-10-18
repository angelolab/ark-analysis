from email.generator import Generator
import pathlib
from typing import Callable, Iterator
import itertools
import pytest
from ark.utils.example_dataset import ExampleDataset, get_example_dataset
from ark.utils import test_utils


# Sets the example dataset path once: Will not cause duplicate downloads
@pytest.fixture(scope="session")
def setup_temp_path_factory(tmp_path_factory) -> Iterator[pathlib.Path]:
    """
    A Fixture which creates the directory where the dataset is saved.
    Downloads the dataset once per session, instead of once per notebook.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Factory for temporary directories under the
            common base temp directory.

    Yields:
        Iterator[pathlib.Path]: The iterable path containing the location of the dataset.
    """
    cache_dir = tmp_path_factory.mktemp("example_dataset")
    yield cache_dir


@pytest.fixture(scope="session", params=["segment_image_data", "cluster_pixels"])
def dataset_download(setup_temp_path_factory, request) -> Iterator[ExampleDataset]:
    """
    A Fixture which instantiates and downloads the dataset with respect to each
    notebook.

    Args:
        setup_temp_path_factory (pytest.Fixture): Factory for temporary directories under the
            common base temp directory.
        request (pytest.FixtureRequest): The parameter, in this case it is the notebook to
            download the dataset for.

    Yields:
        Iterator[ExampleDataset]: The iterable Example Dataset.
    """
    # Set up ExampleDataset class
    example_dataset: ExampleDataset = ExampleDataset(
        dataset=request.param,
        cache_dir=setup_temp_path_factory,
        revision="a3b0db4fa93c194bfcaf5d4daccbe6573c6a6f7c"
    )
    # Download example data for a particular notebook
    example_dataset.download_example_dataset()
    yield example_dataset


class TestExampleDataset:
    @pytest.fixture(autouse=True)
    def _setup(self):
        """
        Sets up necessary information needed for assert statements.
        Sets up dictionary to call the functions which check each dataset that is downloaded.
        """
        self.fov_names = [f"fov{i}" for i in range(11)]
        self.channel_names = ["CD3", "CD4", "CD8", "CD14", "CD20", "CD31", "CD45", "CD68",
                              "CD163", "CK17", "Collagen1", "ECAD", "Fibronectin", "GLUT1",
                              "H3K9ac", "H3K27me3", "HLADR", "IDO", "Ki67", "PD1", "SMA", "Vim"]

        self.cell_table_names = ["cell_table_arcsinh_transformed", "cell_table_size_normalized",
                                 "cell_table_size_normalized_cell_labels"]

        self.deepcell_output_names = [f"fov{i}_feature_{j}" for i in range(11) for j in range(2)]

        self.dataset_test_fns: dict[str, Callable] = {
            "image_data": self._image_data_check,
            "cell_table": self._cell_table_check,
            "deepcell_output": self._deepcell_output_check,
        }

        # Mapping the datasets to their respective test functions.
        self.move_path_suffixes = {
            "image_data": "image_data",
            "cell_table": "segmentation/cell_table",
            "deepcell_output": "segmentation/deepcell_output",
        }

    def test_download_example_dataset(self, dataset_download: ExampleDataset):
        """
        Tests to make sure the proper files are downloaded from Hugging Face.

        Args:
            dataset_download (ExampleDataset): Fixture for the dataset, respective to each
            partition (`segment_image_data`, `cluster_pixels`, `cluster_cells`,
            `post_clustering`).
        """
        dataset_names = list(
            dataset_download.dataset_paths[dataset_download.dataset].features.keys())

        for ds_n in dataset_names:
            dataset_cache_path = pathlib.Path(
                dataset_download.dataset_paths[dataset_download.dataset][ds_n][0])
            self.dataset_test_fns[ds_n](dir_p=dataset_cache_path / ds_n)

    @pytest.mark.parametrize("_overwrite_existing", [True, False])
    def test_move_example_dataset(self, tmp_path_factory, dataset_download: ExampleDataset,
                                  _overwrite_existing: bool):
        """
        Tests to make sure the proper files are moved to the correct directories.

        Args:
            tmp_path_factory (pytest.TempPathFactory): Factory for temporary directories under the
                common base temp directory.
            dataset_download (ExampleDataset): Fixture for the dataset, respective to each
                partition (`segment_image_data`, `cluster_pixels`, `cluster_cells`,
                `post_clustering`).
            _overwrite_existing (bool): If `True` the dataset will be overwritten. If `False` it
                will not be.
        """
        dataset_download.overwrite_existing = _overwrite_existing

        # Move data if _overwrite_existing is `True`
        if _overwrite_existing:

            # Case 1: Move Path is empty
            tmp_dir_c1 = tmp_path_factory.mktemp("move_example_data_c1")
            move_dir_c1 = tmp_dir_c1 / "example_dataset"
            dataset_download.move_example_dataset(move_dir=move_dir_c1)

            for dir_p, ds_n in self._suffix_paths(dataset_download, parent_dir=move_dir_c1):
                self.dataset_test_fns[ds_n](dir_p)

            # Case 2: Move Path contains files
            tmp_dir_c2 = tmp_path_factory.mktemp("move_example_data_c2")
            move_dir_c2 = tmp_dir_c2 / "example_dataset"

            # Add files for each config to test moving with files
            for dir_p, ds_n in self._suffix_paths(dataset_download, parent_dir=move_dir_c2):
                # make directory
                dir_p.mkdir(parents=True, exist_ok=False)
                # make blank file
                test_utils._make_blank_file(dir_p, "data_test.txt")

            # Move files to directory which has existing files
            # Make sure warning is raised
            with pytest.warns(UserWarning):
                dataset_download.move_example_dataset(move_dir=move_dir_c2)
                for dir_p, ds_n in self._suffix_paths(dataset_download, parent_dir=move_dir_c2):
                    self.dataset_test_fns[ds_n](dir_p)

        # Move data if _overwrite_existing is `False`
        else:
            # Case 1: Move Path is empty
            tmp_dir_c1 = tmp_path_factory.mktemp("move_example_data_c1")
            move_dir_c1 = tmp_dir_c1 / "example_dataset"

            # Check that the files were moved to the empty directory
            # Make sure warning is raised
            with pytest.warns(UserWarning):
                dataset_download.move_example_dataset(move_dir=move_dir_c1)

                for dir_p, ds_n in self._suffix_paths(dataset_download, parent_dir=move_dir_c1):
                    self.dataset_test_fns[ds_n](dir_p)

            # Case 2: Move Path contains files
            tmp_dir_c2 = tmp_path_factory.mktemp("move_example_data_c2")
            move_dir_c2 = tmp_dir_c2 / "example_dataset"

            # Add files for each config to test moving with files
            for dir_p, ds_n in self._suffix_paths(dataset_download, parent_dir=move_dir_c2):
                # make directory
                dir_p.mkdir(parents=True, exist_ok=False)
                # make blank file
                test_utils._make_blank_file(dir_p, "data_test.txt")

            # Do not move files to directory containing files
            # Make sure warning is raised.
            with pytest.warns(UserWarning):
                dataset_download.move_example_dataset(move_dir=move_dir_c2)
                for dir_p, ds_n in self._suffix_paths(dataset_download, parent_dir=move_dir_c2):
                    assert len(list(dir_p.rglob("*"))) == 1

    # Will cause duplicate downloads
    def test_get_example_dataset(self, tmp_path_factory):
        """
        Tests to make sure that if an incorrect `dataset` is provided, the function
        errors out with the appropriate error message for the user.
        """

        with pytest.raises(ValueError):
            get_example_dataset("incorrect_dataset", save_dir=tmp_path_factory)

    def test_check_empty_dst(self, tmp_path):
        """
        Tests to make sure that `ExampleDataset.check_empty_dst
        ()` accurately
        reports if a directory contains files or not.
        """

        example_dataset = ExampleDataset(None)
        empty_data_dir: pathlib.Path = tmp_path / "empty_dst_dir"
        packed_data_dir: pathlib.Path = tmp_path / "packed_dst_dir"
        empty_data_dir.mkdir(parents=True)
        packed_data_dir.mkdir(parents=True)

        # Empty directory has no files
        assert example_dataset.check_empty_dst(empty_data_dir) is True

        # Directory has files
        test_utils._make_blank_file(packed_data_dir, "data_test.txt")
        assert example_dataset.check_empty_dst(packed_data_dir) is False

    def _image_data_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that all the FOVs exist.

        Args:
            dir (pathlib.Path): The directory to check.
        """
        # Check to make sure all the FOVs exist
        downloaded_fovs = list(dir_p.glob("*"))
        downloaded_fov_names = [f.stem for f in downloaded_fovs]
        assert set(self.fov_names) == set(downloaded_fov_names)

        # Check to make sure all 22 channels exist
        for fov in downloaded_fovs:
            c_names = [c.stem for c in fov.rglob("*")]
            assert set(self.channel_names) == set(c_names)

    def _cell_table_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that the following cell tables exist:
            * `cell_table_arcsinh_transformed.csv`
            * `cell_table_size_normalized.csv`

        Args:
            dir_p (pathlib.Path): The directory to check.
        """

        downloaded_cell_tables = list(dir_p.glob("*.csv"))
        downloaded_cell_table_names = [f.stem for f in downloaded_cell_tables]
        assert set(self.cell_table_names) == set(downloaded_cell_table_names)

    def _deepcell_output_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that all cell nucleus (feature 0) and cell membrane masks (feature 1)
        exist from deepcell output.

        Args:
            dir_p (pathlib.Path): The directory to check.
        """
        downloaded_deepcell_output = list(dir_p.glob("*.tif"))
        downloaded_deepcell_output_names = [f.stem for f in downloaded_deepcell_output]
        assert set(self.deepcell_output_names) == set(downloaded_deepcell_output_names)

    def _suffix_paths(self, dataset_download: ExampleDataset,
                      parent_dir: pathlib.Path) -> Generator:
        """
        Creates a generator where each element is a tuple of the data directory
        and the dataset name.

        Args:
            dataset_download (ExampleDataset): Fixture for the dataset, respective to each
            partition (`segment_image_data`, `cluster_pixels`, `cluster_cells`,
            `post_clustering`).
            parent_dir (pathlib.Path): The path where the example dataset will be saved.

        Yields:
            Generator: Yields the data directory for the files to be moved, and the dataset name.
        """
        dataset_names = list(
            dataset_download.dataset_paths[dataset_download.dataset].features.keys()
        )

        ds_n_suffixes = [self.move_path_suffixes[ds_n] for ds_n in dataset_names]
        for ds_n_suffix, ds_n in zip(ds_n_suffixes, dataset_names):
            yield (parent_dir / ds_n_suffix, ds_n)
