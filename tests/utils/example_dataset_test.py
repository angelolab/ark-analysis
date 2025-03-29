"""Tests for the example dataset functionality."""

from pathlib import Path
import shutil
from typing import Callable, Generator, Iterator, List, Dict, Any
import warnings

import pytest
from alpineer import test_utils

from ark.utils.example_dataset import (
    ExampleDataset,
    get_example_dataset,
    DatasetFile,
    _check_empty_dir,
)


# Test Data Constants
# ==================

# FOV and channel names
FOV_NAMES = [f"fov{i}" for i in range(11)]
CHANNEL_NAMES = [
    "CD3",
    "CD4",
    "CD8",
    "CD14",
    "CD20",
    "CD31",
    "CD45",
    "CD68",
    "CD163",
    "CK17",
    "Collagen1",
    "ECAD",
    "Fibronectin",
    "GLUT1",
    "H3K9ac",
    "H3K27me3",
    "HLADR",
    "IDO",
    "Ki67",
    "PD1",
    "SMA",
    "Vim",
]

# Cell table names
CELL_TABLE_NAMES = [
    "cell_table_arcsinh_transformed",
    "cell_table_size_normalized",
    "cell_table_size_normalized_cell_labels",
    "generalized_cell_table_input",
    "noisy_groundtruth",
]

# Deepcell output names
DEEPCELL_OUTPUT_NAMES = [
    f"fov{i}_{j}" for i in range(11) for j in ["whole_cell", "nuclear"]
]

# Example pixel output directory structure
EXAMPLE_PIXEL_OUTPUT_DIR_NAMES = {
    "root_files": [
        "cell_clustering_params",
        "channel_norm_pre_rownorm",
        "pixel_thresh",
        "pixel_channel_avg_meta_cluster",
        "pixel_channel_avg_som_cluster",
        "pixel_meta_cluster_mapping",
        "pixel_som_weights",
        "channel_norm_post_rownorm",
    ],
    "pixel_mat_data": [f"fov{i}" for i in range(11)],
    "pixel_mat_subset": [f"fov{i}" for i in range(11)],
    "pixel_masks": [f"fov{i}_pixel_mask" for i in range(2)],
}

# Example cell output directory structure
EXAMPLE_CELL_OUTPUT_DIR_NAMES = {
    "root_files": [
        "cell_meta_cluster_channel_avg",
        "cell_meta_cluster_count_avg",
        "cell_som_cluster_channel_avg",
        "cell_meta_cluster_mapping",
        "cell_som_cluster_channel_avg",
        "cell_som_cluster_count_avg",
        "cell_som_weights",
        "cluster_counts",
        "cluster_counts_size_norm",
        "weighted_cell_channel",
    ],
    "cell_masks": [f"fov{i}_cell_mask" for i in range(2)],
}

# Spatial LDA preprocessed files
SPATIAL_LDA_PREPROCESSED_FILES = [
    "difference_mats",
    "featurized_cell_table",
    "formatted_cell_table",
    "fov_stats",
    "topic_eda",
]

# Post clustering files
POST_CLUSTERING_FILES = [
    "cell_table_thresholded",
    "marker_thresholds",
    "updated_cell_table",
]

# OME TIFF files
OME_TIFF_FILES = ["fov1.ome"]

# EZ segmentation files
EZ_SEG_FILES = {
    "fov_names": [f"fov{i}" for i in range(10)],
    "channel_names": [
        "Ca40",
        "GFAP",
        "Synaptophysin",
        "PanAmyloidbeta1724",
        "Na23",
        "Reelin",
        "Presenilin1NTF",
        "Iba1",
        "CD105",
        "C12",
        "EEA1",
        "VGLUT1",
        "PolyubiK63",
        "Ta181",
        "Au197",
        "Si28",
        "PanGAD6567",
        "CD33Lyo",
        "MAP2",
        "Calretinin",
        "PolyubiK48",
        "MAG",
        "TotalTau",
        "Amyloidbeta140",
        "Background",
        "CD45",
        "8OHGuano",
        "pTDP43",
        "ApoE4",
        "PSD95",
        "TH",
        "HistoneH3Lyo",
        "CD47",
        "Parvalbumin",
        "Amyloidbeta142",
        "Calbindin",
        "PanApoE2E3E4",
        "empty139",
        "CD31",
        "MCT1",
        "MBP",
        "SERT",
        "PHF1Tau",
        "VGAT",
        "VGLUT2",
        "CD56Lyo",
        "MFN2",
    ],
    "composite_names": ["amyloid", "microglia-composite"],
    "ez_mask_suffixes": ["microglia-projections", "plaques"],
    "merged_mask_suffixes": [
        "final_whole_cell_remaining",
        "microglia-projections_merged",
    ],
    "final_mask_suffixes": [
        "final_whole_cell_remaining",
        "microglia-projections_merged",
        "plaques",
    ],
    "cell_table_names": [
        "cell_and_objects_table_arcsinh_transformed",
        "cell_and_objects_table_size_normalized",
        "filtered_final_whole_cell_remaining_table_arcsinh_transformed",
        "filtered_final_whole_cell_remaining_table_size_normalized",
        "filtered_microglia-projections_merged_table_arcsinh_transformed",
        "filtered_microglia-projections_merged_table_size_normalized",
        "filtered_plaques_table_arcsinh_transformed",
        "filtered_plaques_table_size_normalized",
    ],
    "log_names": [
        "amyloid_composite_log",
        "mask_merge_log",
        "microglia-composite_composite_log",
        "microglia-projections_segmentation_log",
        "plaques_segmentation_log",
    ],
}


class BaseDirectoryChecker:
    """Base class for directory checking functionality."""

    def check_files_exist(
        self, dir_path: Path, expected_files: List[str], pattern: str = "*"
    ) -> None:
        """Check if all expected files exist in the directory.

        Args:
            dir_path: Path to directory to check
            expected_files: List of expected file names (without extensions)
            pattern: Glob pattern to use for finding files
        """
        found_files = list(dir_path.glob(pattern))
        found_names = {f.stem for f in found_files}
        expected_names = set(expected_files)
        assert (
            found_names == expected_names
        ), f"Expected {expected_names}, found {found_names}"

    def check_subdirectories(self, dir_path: Path, expected_subdirs: List[str]) -> None:
        """Check if all expected subdirectories exist.

        Args:
            dir_path: Path to parent directory
            expected_subdirs: List of expected subdirectory names
        """
        found_dirs = {d.name for d in dir_path.iterdir() if d.is_dir()}
        expected_dirs = set(expected_subdirs)
        assert (
            found_dirs == expected_dirs
        ), f"Expected {expected_dirs}, found {found_dirs}"

    def find_directory_with_files(
        self,
        base_path: Path,
        expected_files: List[str],
        pattern: str = "*",
        subdirs_to_check: List[str] = None,
        min_matches: int = 3,
    ) -> Path:
        """Find a directory containing the expected files.

        Args:
            base_path: Base directory to start searching from
            expected_files: List of expected file names (without extensions)
            pattern: Glob pattern to use for finding files
            subdirs_to_check: Optional list of subdirectories to check
            min_matches: Minimum number of files to match for a positive result

        Returns:
            Path to the directory containing the expected files
        """
        # Check if the base path has enough of the expected files
        found_files = list(base_path.glob(pattern))
        found_names = {f.stem for f in found_files}
        expected_names = set(expected_files)

        if len(found_names.intersection(expected_names)) >= min_matches:
            return base_path

        # If subdirectories to check are specified, check them
        if subdirs_to_check:
            for subdir in subdirs_to_check:
                subdir_path = base_path / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    found_files = list(subdir_path.glob(pattern))
                    found_names = {f.stem for f in found_files}
                    if len(found_names.intersection(expected_names)) >= min_matches:
                        return subdir_path

        # Check all immediate subdirectories
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                found_files = list(subdir.glob(pattern))
                found_names = {f.stem for f in found_files}
                if len(found_names.intersection(expected_names)) >= min_matches:
                    return subdir

        # If no matching directory found, return the base path
        warnings.warn(f"Could not find directory with expected files in {base_path}")
        return base_path


class ImageDataChecker(BaseDirectoryChecker):
    """Checker for image data directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check image data directory structure."""
        # Check FOVs directly in the directory
        self.check_files_exist(dir_path, FOV_NAMES)

        # Check channels in each FOV
        for fov in dir_path.iterdir():
            if fov.is_dir():
                self.check_files_exist(fov, CHANNEL_NAMES)


class CellTableChecker(BaseDirectoryChecker):
    """Checker for cell table directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check cell table directory structure."""
        # The cell tables are directly in the directory
        self.check_files_exist(dir_path, CELL_TABLE_NAMES, "*.csv")


class DeepcellOutputChecker(BaseDirectoryChecker):
    """Checker for deepcell output directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check deepcell output directory structure."""
        # The deepcell output files are directly in the directory
        self.check_files_exist(dir_path, DEEPCELL_OUTPUT_NAMES, "*.tiff")


class PixelOutputChecker(BaseDirectoryChecker):
    """Checker for pixel output directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check pixel output directory structure."""
        # Find the directory containing pixel output files
        pixel_dir = self.find_directory_with_files(
            dir_path, EXAMPLE_PIXEL_OUTPUT_DIR_NAMES["root_files"], pattern="*.json"
        )

        # Check root files
        root_files = (
            list(pixel_dir.glob("*.json"))
            + list(pixel_dir.glob("*feather"))
            + list(pixel_dir.glob("*.csv"))
        )
        root_names = {f.stem for f in root_files}
        assert root_names == set(EXAMPLE_PIXEL_OUTPUT_DIR_NAMES["root_files"])

        # Check subdirectories
        for subdir, expected_files in EXAMPLE_PIXEL_OUTPUT_DIR_NAMES.items():
            if subdir != "root_files":
                subdir_path = pixel_dir / subdir
                self.check_files_exist(subdir_path, expected_files)


class CellOutputChecker(BaseDirectoryChecker):
    """Checker for cell output directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check cell output directory structure."""
        # Find the directory containing cell output files
        cell_dir = self.find_directory_with_files(
            dir_path, EXAMPLE_CELL_OUTPUT_DIR_NAMES["root_files"], pattern="*.feather"
        )

        # Check root files
        root_files = list(cell_dir.glob("*.feather")) + list(cell_dir.glob("*.csv"))
        root_names = {f.stem for f in root_files}
        assert root_names == set(EXAMPLE_CELL_OUTPUT_DIR_NAMES["root_files"])

        # Check cell masks
        self.check_files_exist(
            cell_dir / "cell_masks",
            EXAMPLE_CELL_OUTPUT_DIR_NAMES["cell_masks"],
            "*.tiff",
        )


class SpatialLDAChecker(BaseDirectoryChecker):
    """Checker for spatial LDA directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check spatial LDA directory structure."""
        # Find the spatial_lda directory
        spatial_lda_path = self.find_directory_with_files(
            dir_path,
            [],  # We're not looking for specific files at the top level
            pattern="*",
            subdirs_to_check=["spatial_lda", "preprocessed"],
            min_matches=1,
        )

        # If we've found the spatial_lda directory but not the preprocessed subdirectory
        if spatial_lda_path.name == "spatial_lda":
            spatial_lda_dir = spatial_lda_path / "preprocessed"
        # If we've found the preprocessed directory
        elif spatial_lda_path.name == "preprocessed":
            spatial_lda_dir = spatial_lda_path
        # Otherwise, try the expected directory structure
        else:
            spatial_lda_dir = dir_path / "spatial_lda" / "preprocessed"

        self.check_files_exist(spatial_lda_dir, SPATIAL_LDA_PREPROCESSED_FILES, "*.pkl")


class PostClusteringChecker(BaseDirectoryChecker):
    """Checker for post clustering directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check post clustering directory structure."""
        # Find the directory containing post clustering files
        post_clustering_dir = self.find_directory_with_files(
            dir_path,
            POST_CLUSTERING_FILES,
            pattern="*.csv",
            subdirs_to_check=["post_clustering"],
            min_matches=1,
        )

        self.check_files_exist(post_clustering_dir, POST_CLUSTERING_FILES, "*.csv")


class OMETiffChecker(BaseDirectoryChecker):
    """Checker for OME TIFF directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check OME TIFF directory structure."""
        # The OME TIFF files are directly in the directory
        self.check_files_exist(dir_path, OME_TIFF_FILES, "*.ome.tiff")


class EZSegChecker(BaseDirectoryChecker):
    """Checker for EZ segmentation directory structure."""

    def check(self, dir_path: Path) -> None:
        """Check EZ segmentation directory structure."""
        # Find the ez_seg_data directory
        ez_seg_dir = self.find_directory_with_files(
            dir_path,
            [],  # We're not looking for specific files at the top level
            pattern="*",
            subdirs_to_check=["ez_seg_data"],
            min_matches=1,
        )

        if ez_seg_dir.name != "ez_seg_data":
            ez_seg_dir = ez_seg_dir / "ez_seg_data"

        # Check image data - FOVs are directly in the image_data directory
        image_data = ez_seg_dir / "image_data"
        # First check if the directory exists
        assert image_data.exists(), f"Directory {image_data} does not exist"

        # Get all subdirectories that start with 'fov'
        fov_dirs = [
            d for d in image_data.iterdir() if d.is_dir() and d.name.startswith("fov")
        ]
        found_fov_names = {d.name for d in fov_dirs}
        expected_fov_names = set(EZ_SEG_FILES["fov_names"])
        msg = f"Expected FOVs {expected_fov_names}, " f"found {found_fov_names}"
        assert found_fov_names == expected_fov_names, msg

        # Check channels in each FOV
        for fov in fov_dirs:
            # Get all files in the FOV directory that have a channel name
            channel_files = [f for f in fov.iterdir() if f.is_file()]
            found_channels = {f.stem for f in channel_files}
            expected_channels = set(EZ_SEG_FILES["channel_names"])
            msg = (
                f"In {fov.name}, "
                f"expected channels {expected_channels}, "
                f"found {found_channels}"
            )
            assert found_channels == expected_channels, msg

        # Check composites
        composites = ez_seg_dir / "composites"
        assert composites.exists(), f"Directory {composites} does not exist"

        # Get all subdirectories that start with 'fov'
        composite_fov_dirs = [
            d for d in composites.iterdir() if d.is_dir() and d.name.startswith("fov")
        ]
        found_composite_fovs = {d.name for d in composite_fov_dirs}
        msg = (
            f"Expected composite FOVs {expected_fov_names}, "
            f"found {found_composite_fovs}"
        )
        assert found_composite_fovs == expected_fov_names, msg

        # Check composite files in each FOV
        for fov in composite_fov_dirs:
            composite_files = [f for f in fov.iterdir() if f.is_file()]
            found_composites = {f.stem for f in composite_files}
            expected_composites = set(EZ_SEG_FILES["composite_names"])
            msg = (
                f"In {fov.name}, "
                f"expected composites {expected_composites}, "
                f"found {found_composites}"
            )
            assert found_composites == expected_composites, msg

        # Check cell tables
        cell_table_dir = ez_seg_dir / "cell_table"
        assert cell_table_dir.exists(), f"Directory {cell_table_dir} does not exist"
        self.check_files_exist(
            cell_table_dir, EZ_SEG_FILES["cell_table_names"], "*.csv"
        )

        # All segmentation-related directories are under segmentation/
        segmentation_dir = ez_seg_dir / "segmentation"
        assert segmentation_dir.exists(), f"Directory {segmentation_dir} does not exist"

        # Check deepcell output
        deepcell_output = segmentation_dir / "deepcell_output"
        assert deepcell_output.exists(), f"Directory {deepcell_output} does not exist"
        whole_cell_names = [f"{fov}_whole_cell" for fov in EZ_SEG_FILES["fov_names"]]
        self.check_files_exist(deepcell_output, whole_cell_names, "*.tiff")

        # Check ez masks
        ez_masks = segmentation_dir / "ez_masks"
        assert ez_masks.exists(), f"Directory {ez_masks} does not exist"
        ez_names = [
            f"{fov}_{suffix}"
            for fov in EZ_SEG_FILES["fov_names"]
            for suffix in EZ_SEG_FILES["ez_mask_suffixes"]
        ]
        self.check_files_exist(ez_masks, ez_names, "*.tiff")

        # Check merged masks
        merged_masks = segmentation_dir / "merged_masks_dir"
        assert merged_masks.exists(), f"Directory {merged_masks} does not exist"
        merged_names = [
            f"{fov}_{suffix}"
            for fov in EZ_SEG_FILES["fov_names"]
            for suffix in EZ_SEG_FILES["merged_mask_suffixes"]
        ]
        self.check_files_exist(merged_masks, merged_names, "*.tiff")

        # Check final masks
        final_masks = segmentation_dir / "final_mask_dir"
        final_names = [
            f"{fov}_{suffix}"
            for fov in EZ_SEG_FILES["fov_names"]
            for suffix in EZ_SEG_FILES["final_mask_suffixes"]
        ]
        self.check_files_exist(final_masks, final_names, "*.tiff")

        # Check logs
        logs_dir = ez_seg_dir / "logs"
        assert logs_dir.exists(), f"Directory {logs_dir} does not exist"
        self.check_files_exist(logs_dir, EZ_SEG_FILES["log_names"], "*.txt")


# Test Fixtures
# ============


@pytest.fixture(
    scope="class",
    params=[
        "segment_image_data",
        "cluster_pixels",
        "cluster_cells",
        "post_clustering",
        "fiber_segmentation",
        "LDA_preprocessing",
        "LDA_training_inference",
        "neighborhood_analysis",
        "pairwise_spatial_enrichment",
        "ome_tiff",
        "ez_seg_data",
    ],
)
def dataset_download(request, dataset_cache_dir) -> Iterator[ExampleDataset]:
    """Fixture which instantiates and downloads the dataset for each notebook."""
    example_dataset = ExampleDataset(
        dataset=request.param, cache_dir=dataset_cache_dir, revision="main"
    )
    example_dataset.download_example_dataset()
    yield example_dataset


@pytest.fixture(scope="function")
def cleanable_tmp_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """Fixture for creating and cleaning up temporary directories."""
    data_path = tmp_path_factory.mktemp("data")
    yield data_path
    shutil.rmtree(data_path)


# Test Cases
# ==========


class TestExampleDataset:
    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up test data and checkers."""
        # Mapping the datasets to their respective test functions
        self.dataset_test_fns: Dict[DatasetFile, Callable] = {
            DatasetFile(name="image_data.zip"): ImageDataChecker().check,
            DatasetFile(
                name="cell_table.zip", parent_dir="segmentation"
            ): CellTableChecker().check,
            DatasetFile(
                name="deepcell_output.zip", parent_dir="segmentation"
            ): DeepcellOutputChecker().check,
            DatasetFile(
                name="example_pixel_output_dir.zip", parent_dir="pixie"
            ): PixelOutputChecker().check,
            DatasetFile(
                name="example_cell_output_dir.zip", parent_dir="pixie"
            ): CellOutputChecker().check,
            DatasetFile(
                name="spatial_lda.zip", parent_dir="spatial_analysis"
            ): SpatialLDAChecker().check,
            DatasetFile(name="post_clustering.zip"): PostClusteringChecker().check,
            DatasetFile(name="ome_tiff.zip"): OMETiffChecker().check,
            DatasetFile(name="ez_seg_data.zip"): EZSegChecker().check,
        }

    def _get_checker_for_dataset_file(self, ds_file: DatasetFile) -> Callable:
        """Get the appropriate checker function for a dataset file.

        Args:
            ds_file: The dataset file to match

        Returns:
            The checker function for the dataset file
        """
        for key, value in self.dataset_test_fns.items():
            if key.name == ds_file.name and key.parent_dir == ds_file.parent_dir:
                return value

        raise ValueError(f"No checker found for dataset file {ds_file}")

    def test_download_example_dataset(self, dataset_download: ExampleDataset):
        """Test that files are downloaded correctly from Hugging Face."""
        dataset_files = list(
            dataset_download.dataset_paths[dataset_download.dataset].keys()
        )
        for ds_file in dataset_files:
            dataset_cache_path = Path(
                dataset_download.dataset_paths[dataset_download.dataset][ds_file]
            )

            # Get the inner data path (where the actual data is stored)
            inner_path = ds_file.get_inner_data_path(dataset_cache_path)

            # If inner path doesn't exist, try the unzipped directory directly
            if not inner_path.exists():
                inner_path = dataset_cache_path

            # Get the appropriate checker for this dataset file
            checker = self._get_checker_for_dataset_file(ds_file)
            checker(dir_path=inner_path)

    @pytest.mark.parametrize("_overwrite_existing", [True, False])
    def test_move_example_dataset(
        self,
        cleanable_tmp_path,
        dataset_download: ExampleDataset,
        _overwrite_existing: bool,
    ):
        """Test moving files to correct directories with overwrite options."""
        dataset_download.overwrite_existing = _overwrite_existing

        # Move data if _overwrite_existing is `True`
        if _overwrite_existing:
            # Case 1: Move Path is empty
            tmp_dir_c1: Path = cleanable_tmp_path / "move_example_data_c1"
            tmp_dir_c1.mkdir(parents=True, exist_ok=False)

            move_dir_c1: Path = tmp_dir_c1 / "example_dataset"
            move_dir_c1.mkdir(parents=True, exist_ok=False)

            dataset_download.move_example_dataset(move_dir=move_dir_c1)

            for dir_p, ds_file in self._suffix_paths(
                dataset_download, parent_dir=move_dir_c1
            ):
                checker = self._get_checker_for_dataset_file(ds_file)
                checker(dir_path=dir_p)

            # Case 2: Move Path contains files
            tmp_dir_c2: Path = cleanable_tmp_path / "move_example_data_c2"
            tmp_dir_c2.mkdir(parents=True, exist_ok=False)

            move_dir_c2: Path = tmp_dir_c2 / "example_dataset"
            move_dir_c2.mkdir(parents=True, exist_ok=False)

            # Add files for each config to test moving with files
            for dir_p, ds_file in self._suffix_paths(
                dataset_download, parent_dir=move_dir_c2
            ):
                # make directory
                dir_p.mkdir(parents=True, exist_ok=False)
                # make blank file
                test_utils._make_blank_file(dir_p, "data_test.txt")

            # Move files to directory which has existing files
            # Make sure warning is raised
            with pytest.warns(UserWarning):
                dataset_download.move_example_dataset(move_dir=move_dir_c2)
                for dir_p, ds_file in self._suffix_paths(
                    dataset_download, parent_dir=move_dir_c2
                ):
                    checker = self._get_checker_for_dataset_file(ds_file)
                    checker(dir_path=dir_p)

        # Move data if _overwrite_existing is `False`
        else:
            # Case 1: Move Path is empty
            tmp_dir_c1: Path = cleanable_tmp_path / "move_example_data_c1"
            tmp_dir_c1.mkdir(parents=True, exist_ok=False)
            move_dir_c1 = tmp_dir_c1 / "example_dataset"
            move_dir_c1.mkdir(parents=True, exist_ok=False)

            # Check that the files were moved to the empty directory
            # Make sure warning is raised
            with pytest.warns(UserWarning):
                dataset_download.move_example_dataset(move_dir=move_dir_c1)

                for dir_p, ds_file in self._suffix_paths(
                    dataset_download, parent_dir=move_dir_c1
                ):
                    checker = self._get_checker_for_dataset_file(ds_file)
                    checker(dir_path=dir_p)

            # Case 2: Move Path contains files
            tmp_dir_c2 = cleanable_tmp_path / "move_example_data_c2"
            tmp_dir_c2.mkdir(parents=True, exist_ok=False)
            move_dir_c2 = tmp_dir_c2 / "example_dataset"
            move_dir_c2.mkdir(parents=True, exist_ok=False)

            # Add files for each config to test moving with files
            for dir_p, ds_file in self._suffix_paths(
                dataset_download, parent_dir=move_dir_c2
            ):
                # make directory
                dir_p.mkdir(parents=True, exist_ok=False)
                # make blank file
                test_utils._make_blank_file(dir_p, "data_test.txt")

            # Do not move files to directory containing files
            # Make sure warning is raised.
            with pytest.warns(UserWarning):
                dataset_download.move_example_dataset(move_dir=move_dir_c2)
                for dir_p, ds_file in self._suffix_paths(
                    dataset_download, parent_dir=move_dir_c2
                ):
                    assert len(list(dir_p.rglob("*"))) == 1

    def test_get_example_dataset(self, cleanable_tmp_path):
        """Test error handling for incorrect dataset names."""
        with pytest.raises(ValueError):
            get_example_dataset("incorrect_dataset", save_dir=cleanable_tmp_path)

    def test_check_empty_dir(self, tmp_path):
        """Test directory emptiness checking functionality."""
        empty_data_dir: Path = tmp_path / "empty_dst_dir"
        packed_data_dir: Path = tmp_path / "packed_dst_dir"
        empty_data_dir.mkdir(parents=True)
        packed_data_dir.mkdir(parents=True)

        # Empty directory has no files
        assert _check_empty_dir(empty_data_dir) is True

        # Directory has files
        test_utils._make_blank_file(packed_data_dir, "data_test.txt")
        assert _check_empty_dir(packed_data_dir) is False

    def _suffix_paths(
        self, dataset_download: ExampleDataset, parent_dir: Path
    ) -> Generator:
        """Create generator for dataset paths and names.

        Args:
            dataset_download: The example dataset fixture
            parent_dir: The parent directory for the dataset

        Yields:
            Tuples of (directory path, dataset file)
        """
        dataset_files = list(
            dataset_download.dataset_paths[dataset_download.dataset].keys()
        )

        for ds_file in dataset_files:
            # Use the get_destination_path method to get the path where the data will be stored
            dst_path = ds_file.get_destination_path(parent_dir)
            yield (dst_path, ds_file)
