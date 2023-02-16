import os
import pathlib
from typing import Callable, Dict, Generator, Iterator, List

import pytest
from alpineer import test_utils

from ark.settings import EXAMPLE_DATASET_REVISION
from ark.utils.example_dataset import ExampleDataset, get_example_dataset


@pytest.fixture(scope="session", params=["segment_image_data",
                                         "cluster_pixels",
                                         "cluster_cells",
                                         "post_clustering",
                                         "fiber_segmentation",
                                         "LDA_preprocessing",
                                         "LDA_training_inference",
                                         "neighborhood_analysis",
                                         "pairwise_spatial_enrichment",
                                         "ome_tiff"])
def dataset_download(request) -> Iterator[ExampleDataset]:
    """
    A Fixture which instantiates and downloads the dataset with respect to each
    notebook.

    Args:
        request (pytest.FixtureRequest): The parameter, in this case it is the notebook to
            download the dataset for.

    Yields:
        Iterator[ExampleDataset]: The iterable Example Dataset.
    """

    # Set up ExampleDataset class
    example_dataset: ExampleDataset = ExampleDataset(
        dataset=request.param,
        cache_dir=None,  # Use the default cache
        revision=EXAMPLE_DATASET_REVISION
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
                                 "cell_table_size_normalized_cell_labels",
                                 "generalized_cell_table_input"]

        self.deepcell_output_names = [f"fov{i}_{j}" for i in range(11)
                                      for j in ['whole_cell', 'nuclear']]

        self._example_pixel_output_dir_names = {
            "root_files": ["cell_clustering_params", "channel_norm", "pixel_thresh",
                           "pixel_channel_avg_meta_cluster", "pixel_channel_avg_som_cluster",
                           "pixel_meta_cluster_mapping", "pixel_som_weights",
                           "channel_norm_post_rowsum"],
            "pixel_mat_data": [f"fov{i}" for i in range(11)],
            "pixel_mat_subset": [f"fov{i}" for i in range(11)],
            "pixel_masks": [f"fov{i}_pixel_mask" for i in range(2)]
        }

        self._example_cell_output_dir_names = {
            "root_files": ["cell_meta_cluster_channel_avg",
                           "cell_meta_cluster_count_avg",
                           "cell_som_cluster_channel_avg",
                           "cell_meta_cluster_mapping",
                           "cell_som_cluster_channel_avg",
                           "cell_som_cluster_count_avg",
                           "cell_som_weights", "cluster_counts",
                           "cluster_counts_size_norm", "weighted_cell_channel"],
            "cell_masks": [f"fov{i}_cell_mask" for i in range(2)]
        }

        self._spatial_analysis_lda_preprocessed_files: List[str] = [
            "difference_mats",
            "featurized_cell_table",
            "formatted_cell_table",
            "fov_stats",
            "topic_eda"]

        self._post_clustering_files = ["cell_table_thresholded",
                                       "marker_thresholds", "updated_cell_table"]

        self._ome_tiff_files: List[str] = ["fov1.ome"]

        self.dataset_test_fns: dict[str, Callable] = {
            "image_data": self._image_data_check,
            "cell_table": self._cell_table_check,
            "deepcell_output": self._deepcell_output_check,
            "example_pixel_output_dir": self._example_pixel_output_dir_check,
            "example_cell_output_dir": self._example_cell_output_dir_check,
            "spatial_lda": self._spatial_lda_output_dir_check,
            "post_clustering": self._post_clustering_output_dir_check,
            "ome_tiff": self._ome_tiff_check
        }

        # Mapping the datasets to their respective test functions.
        # Should be the same as `example_dataset.ExampleDataset.path_suffixes`
        self.move_path_suffixes = {
            "image_data": "image_data",
            "cell_table": "segmentation/cell_table",
            "deepcell_output": "segmentation/deepcell_output",
            "example_pixel_output_dir": "pixie/example_pixel_output_dir",
            "example_cell_output_dir": "pixie/example_cell_output_dir",
            "spatial_lda": "spatial_analysis/spatial_lda",
            "post_clustering": "post_clustering",
            "ome_tiff": "ome_tiff",
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
        errors out with an appropriate error message for the user.
        """

        with pytest.raises(ValueError):
            get_example_dataset("incorrect_dataset", save_dir=tmp_path_factory)

    def test_check_empty_dst(self, tmp_path):
        """
        Tests to make sure that `ExampleDataset.check_empty_dst()` accurately
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
        Checks to make sure that all cell nucleus (nuclear) and cell membrane masks (whole_cell)
        exist from deepcell output.

        Args:
            dir_p (pathlib.Path): The directory to check.
        """
        downloaded_deepcell_output = list(dir_p.glob("*.tiff"))
        downloaded_deepcell_output_names = [f.stem for f in downloaded_deepcell_output]
        assert set(self.deepcell_output_names) == set(downloaded_deepcell_output_names)

    def _example_pixel_output_dir_check(self, dir_p: pathlib.Path):
        """
        `example_pixel_output_dir`.
            ├── cell_clustering_params.json
            ├── channel_norm.feather
            ├── pixel_thresh.feather
            ├── pixel_channel_avg_meta_cluster.csv
            ├── pixel_channel_avg_som_cluster.csv
            ├── pixel_masks/
            │  ├── fov0_pixel_mask.tiff
            │  └── fov1_pixel_mask.tiff
            ├── pixel_mat_data/
            │  ├── fov0.feather
            │  ├── fov1.feather
            │  ├── ...
            │  └── fov10.feather
            ├── pixel_mat_subset/
            │  ├── fov0.feather
            │  ├── fov1.feather
            │  ├── ...
            │  └── fov10.feather
            ├── pixel_meta_cluster_mapping.csv
            ├── pixel_som_weights.feather
            └── channel_norm_post_rowsum.feather
        ```
        Args:
            dir_p (pathlib.Path): The directory to check.
        """
        # Root Files
        root_files = list(dir_p.glob("*.json")) + \
            list(dir_p.glob("*feather")) + \
            list(dir_p.glob("*csv"))
        root_file_names = [f.stem for f in root_files]
        assert set(self._example_pixel_output_dir_names["root_files"]) == set(root_file_names)

        # Pixel Mat Data
        pixel_mat_files = list((dir_p / "pixel_mat_data").glob("*.feather"))
        pixel_mat_files_names = [f.stem for f in pixel_mat_files]
        assert set(self._example_pixel_output_dir_names["pixel_mat_data"]) \
            == set(pixel_mat_files_names)

        # Pixel Mat Subset
        pixel_mat_subset_files = list((dir_p / "pixel_mat_subset").glob("*.feather"))
        pixel_mat_subset_names = [f.stem for f in pixel_mat_subset_files]
        assert set(self._example_pixel_output_dir_names["pixel_mat_subset"]) \
            == set(pixel_mat_subset_names)

        # Pixel Masks
        pixel_mask_files = list((dir_p / "pixel_masks").glob("*.tiff"))
        pixel_mask_names = [f.stem for f in pixel_mask_files]
        assert set(self._example_pixel_output_dir_names["pixel_masks"]) \
            == set(pixel_mask_names)

    def _example_cell_output_dir_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that the following files exist w.r.t the
        `example_cell_output_dir`.

        ```
        example_cell_output_dir/
        ├── cell_masks/
        │  ├── fov0_cell_mask.tiff
        │  └── fov1_cell_mask.tiff
        ├── cell_meta_cluster_channel_avg.csv
        ├── cell_meta_cluster_count_avgs.csv
        ├── cell_meta_cluster_mapping.csv
        ├── cell_som_cluster_channel_avg.csv
        ├── cell_som_cluster_count_avgs.csv
        ├── cell_som_weights.feather
        ├── cluster_counts.feather
        ├── cluster_counts_size_norm.feather
        └── weighted_cell_channel.csv
        ```

        Args:
            dir_p (pathlib.Path): The directory to check.
        """

        # Root Files
        root_files = list(dir_p.glob("*.feather")) + list(dir_p.glob("*.csv"))
        root_file_names = [f.stem for f in root_files]
        assert set(self._example_cell_output_dir_names["root_files"]) == set(root_file_names)

        # Cell Masks
        cell_mask_files = list((dir_p / "cell_masks").glob("*.tiff"))
        cell_mask_names = [f.stem for f in cell_mask_files]
        assert set(self._example_cell_output_dir_names["cell_masks"]) \
            == set(cell_mask_names)

    def _spatial_lda_output_dir_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that the correct files exist w.r.t the `spatial_lda` output dir
        `spatial_analysis/spatial_lda/preprocessed`.

        Args:
            dir_p (pathlib.Path): The directory to check.
        """
        downloaded_lda_preprocessed = list((dir_p / "preprocessed").glob("*.pkl"))
        downloaded_lda_preprocessed_names = [f.stem for f in downloaded_lda_preprocessed]
        assert set(self._spatial_analysis_lda_preprocessed_files) == set(
            downloaded_lda_preprocessed_names)

    def _post_clustering_output_dir_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that the correct files exist w.r.t the `post_clustering` output dir

        Args:
            dir_p (pathlib.Path): The directory to check.
        """
        downloaded_post_cluster = list(dir_p.glob("*.csv"))
        downloaded_post_cluster_names = [f.stem for f in downloaded_post_cluster]
        assert set(self._post_clustering_files) == set(downloaded_post_cluster_names)

    def _ome_tiff_check(self, dir_p: pathlib.Path):
        """
        Checks to make sure that the correct files exist w.r.t the `ome_tiff` output dir

        Args:
            dir_p (pathlib.Path): The directory to check.
        """
        downloaded_ome_tiff = list(dir_p.glob("*.ome.tiff"))
        downloaded_ome_tiff_names = [f.stem for f in downloaded_ome_tiff]
        assert set(self._ome_tiff_files) == set(downloaded_ome_tiff_names)

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
