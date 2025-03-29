import warnings
from typing import Union, List, Optional, Dict
from enum import Enum
from collections.abc import Mapping
import pooch
from alpineer.misc_utils import verify_in_list
from pathlib import Path
from fsspec.implementations.local import LocalFileSystem
from dataclasses import dataclass


class ArkDatasetSettings(str, Enum):
    """Dataset settings"""

    OS_CACHE = "ark-analysis-datasets"
    BASE_URL = "https://huggingface.co/datasets/angelolab/ark_example/resolve/main/data"

    @classmethod
    def get_cache_dir(cls) -> Path:
        """Returns the cache directory path"""
        return pooch.os_cache(cls.OS_CACHE)

    @classmethod
    def get_dataset_url(cls, revision: str = "main") -> str:
        """Returns the full URL for a dataset"""
        base_url = f"https://huggingface.co/datasets/angelolab/ark_example/resolve/{revision}/data"
        return base_url


class DatasetConfig(str, Enum):
    """Available dataset configurations"""

    SEGMENT_IMAGE_DATA = "segment_image_data"
    CLUSTER_PIXELS = "cluster_pixels"
    CLUSTER_CELLS = "cluster_cells"
    POST_CLUSTERING = "post_clustering"
    FIBER_SEGMENTATION = "fiber_segmentation"
    LDA_PREPROCESSING = "LDA_preprocessing"
    LDA_TRAINING_INFERENCE = "LDA_training_inference"
    NEIGHBORHOOD_ANALYSIS = "neighborhood_analysis"
    PAIRWISE_SPATIAL_ENRICHMENT = "pairwise_spatial_enrichment"
    OME_TIFF = "ome_tiff"
    EZ_SEG_DATA = "ez_seg_data"


@dataclass(frozen=True)
class DatasetFile:
    """Represents a dataset file with its path structure"""

    name: str  # The filename (e.g., "image_data.zip")
    parent_dir: Optional[str] = None  # Subdirectory (e.g., "pixie", "segmentation")
    sha256: str = ""  # SHA256 hash for verification

    @property
    def value(self) -> str:
        """Returns the full path to the file including parent directory"""
        if self.parent_dir:
            return f"{self.parent_dir}/{self.name}"
        return self.name

    def get_unzipped_path(self, cache_dir: Path) -> Path:
        """Returns the path to the unzipped directory for this dataset file.

        Args:
            cache_dir: The cache directory where files are downloaded

        Returns:
            Path to the unzipped directory
        """
        # Handle parent directories in the cache structure
        if self.parent_dir:
            return cache_dir / self.parent_dir / f"{self.name}.unzip"
        return cache_dir / f"{self.name}.unzip"

    def get_inner_data_path(self, base_path: Path) -> Path:
        """Returns the path to the inner data directory within the unzipped folder

        Args:
            base_path: The base path to the unzipped directory

        Returns:
            Path to the inner data directory
        """
        # The file name without the extension
        data_dir_name = self.name.replace(".zip", "")
        return base_path / data_dir_name

    def get_destination_path(self, base_dir: Path) -> Path:
        """Calculate the destination path for this dataset file

        Args:
            base_dir: The base destination directory

        Returns:
            Path to where this dataset should be saved
        """
        # Get the data directory name without .zip extension
        data_dir = self.name.replace(".zip", "")

        # Construct the path preserving parent directories if present
        dst_parts = []
        if self.parent_dir:
            dst_parts.extend(self.parent_dir.split("/"))
        dst_parts.append(data_dir)

        return base_dir.joinpath(*dst_parts)


# Define all dataset files with their proper structure
class DatasetFiles:
    """Individual dataset zip files with their structure"""

    IMAGE_DATA = DatasetFile(
        name="image_data.zip",
        sha256="4d1f29e53a40bb162e795f68906095131d88647f70a6e080ca3bac749a3f6174",
    )

    CELL_TABLE = DatasetFile(
        name="cell_table.zip",
        parent_dir="segmentation",
        sha256="4c453a9a6fe6f21d411ecd9e60278e4360d8df4b876f9b382d2d139ae62ade5d",
    )

    DEEPCELL_OUTPUT = DatasetFile(
        name="deepcell_output.zip",
        parent_dir="segmentation",
        sha256="92c1bf023c58187264d31848addb602d8aeb7f7545ef0fa370c73a5f35325a15",
    )

    EXAMPLE_PIXEL_OUTPUT = DatasetFile(
        name="example_pixel_output_dir.zip",
        parent_dir="pixie",
        sha256="1e01263920bb7d98b836204ad3c7e6175487c9d139c7cf504c136391536796ba",
    )

    EXAMPLE_CELL_OUTPUT = DatasetFile(
        name="example_cell_output_dir.zip",
        parent_dir="pixie",
        sha256="a76c024385ae04c9f21229a053c3623429d9fa26241265fc304e92716eb5fe3d",
    )

    SPATIAL_LDA = DatasetFile(
        name="spatial_lda.zip",
        parent_dir="spatial_analysis",
        sha256="eb7bdab9c054d4fff8504c51e9b185e02a8cc13e15287bafdee2fbb7ddcd66b4",
    )

    POST_CLUSTERING = DatasetFile(
        name="post_clustering.zip",
        sha256="d969d298c2a42130f17955b13f5b1b2c1e8687c180e1e27ce9205bc9ea8d33a1",
    )

    OME_TIFF = DatasetFile(
        name="ome_tiff.zip",
        sha256="82d3278f7d3e27e128f3ec3607934b933f68e3a3aab1298ce1e6dc5d0cf402e2",
    )

    EZ_SEG_DATA = DatasetFile(
        name="ez_seg_data.zip",
        sha256="7251a3c98e53ac00858560c1f234208228f14198b915aabd6715cac0c8d5d74c",
    )

    @classmethod
    def get_all_files(cls) -> List[DatasetFile]:
        """Returns all dataset files as a list"""
        return [
            cls.IMAGE_DATA,
            cls.CELL_TABLE,
            cls.DEEPCELL_OUTPUT,
            cls.EXAMPLE_PIXEL_OUTPUT,
            cls.EXAMPLE_CELL_OUTPUT,
            cls.SPATIAL_LDA,
            cls.POST_CLUSTERING,
            cls.OME_TIFF,
            cls.EZ_SEG_DATA,
        ]


# Dataset configurations mapping
DATASET_CONFIGS: Mapping[DatasetConfig, List[DatasetFile]] = {
    DatasetConfig.SEGMENT_IMAGE_DATA: [DatasetFiles.IMAGE_DATA],
    DatasetConfig.CLUSTER_PIXELS: [
        DatasetFiles.IMAGE_DATA,
        DatasetFiles.CELL_TABLE,
        DatasetFiles.DEEPCELL_OUTPUT,
    ],
    DatasetConfig.CLUSTER_CELLS: [
        DatasetFiles.IMAGE_DATA,
        DatasetFiles.CELL_TABLE,
        DatasetFiles.DEEPCELL_OUTPUT,
        DatasetFiles.EXAMPLE_PIXEL_OUTPUT,
    ],
    DatasetConfig.POST_CLUSTERING: [
        DatasetFiles.IMAGE_DATA,
        DatasetFiles.CELL_TABLE,
        DatasetFiles.DEEPCELL_OUTPUT,
        DatasetFiles.EXAMPLE_CELL_OUTPUT,
    ],
    DatasetConfig.FIBER_SEGMENTATION: [DatasetFiles.IMAGE_DATA],
    DatasetConfig.LDA_PREPROCESSING: [DatasetFiles.IMAGE_DATA, DatasetFiles.CELL_TABLE],
    DatasetConfig.LDA_TRAINING_INFERENCE: [
        DatasetFiles.IMAGE_DATA,
        DatasetFiles.CELL_TABLE,
        DatasetFiles.SPATIAL_LDA,
    ],
    DatasetConfig.NEIGHBORHOOD_ANALYSIS: [
        DatasetFiles.IMAGE_DATA,
        DatasetFiles.CELL_TABLE,
        DatasetFiles.DEEPCELL_OUTPUT,
    ],
    DatasetConfig.PAIRWISE_SPATIAL_ENRICHMENT: [
        DatasetFiles.IMAGE_DATA,
        DatasetFiles.CELL_TABLE,
        DatasetFiles.DEEPCELL_OUTPUT,
        DatasetFiles.POST_CLUSTERING,
    ],
    DatasetConfig.OME_TIFF: [DatasetFiles.OME_TIFF],
    DatasetConfig.EZ_SEG_DATA: [DatasetFiles.EZ_SEG_DATA],
}


def create_dataset_registry() -> Dict[str, str]:
    """Creates the dataset registry mapping file paths to SHA256 checksums"""
    registry = {}

    # Create registry entries for each dataset file
    for dataset_file in DatasetFiles.get_all_files():
        registry[dataset_file.value] = dataset_file.sha256

    return registry


def create_pooch_instance(revision: str = "main") -> pooch.Pooch:
    """Creates a Pooch instance for managing datasets"""
    return pooch.create(
        path=ArkDatasetSettings.get_cache_dir(),
        base_url=ArkDatasetSettings.get_dataset_url(revision),
        registry=create_dataset_registry(),
    )


def _check_empty_dir(dir_path: Path) -> bool:
    """
    Checks if a directory is empty (or doesn't exist)

    Args:
        dir_path: Path to the directory to check

    Returns:
        True if the directory is empty or doesn't exist
    """
    if not dir_path.exists():
        return True

    dir_files = list(dir_path.rglob("*"))
    return len(dir_files) == 0


def _copy_directory_contents(
    src_path: Path, dst_path: Path, fs: LocalFileSystem
) -> None:
    """
    Copies all files from source to destination recursively

    Args:
        src_path: Source directory path
        dst_path: Destination directory path
        fs: File system to use for copying
    """
    if not src_path.exists():
        return

    # Create the destination directory and its parents
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy all files from source to destination
    for src_file in src_path.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(src_path)
            dst_file = dst_path / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            fs.copy(str(src_file), str(dst_file))


def _create_warning_message(dst_path: Path, will_overwrite: bool) -> str:
    """
    Creates a consistent warning message about file operations

    Args:
        dst_path: The destination path
        will_overwrite: Whether files will be overwritten

    Returns:
        A formatted warning message
    """
    if will_overwrite:
        return (
            f"Files exist in {dst_path}. "
            f"They will be overwritten by the downloaded example dataset."
        )
    else:
        return f"Files exist in {dst_path}. " f"They will not be overwritten."


def _clean_macosx_dirs(base_dir: Path, fs: LocalFileSystem) -> None:
    """
    Removes __MACOSX directories

    Args:
        base_dir: The directory to clean
        fs: File system to use for removal
    """
    for path in base_dir.rglob("__MACOSX/"):
        fs.rm(str(path), recursive=True)


class ExampleDataset:
    def __init__(
        self,
        dataset: str,
        overwrite_existing: bool = True,
        cache_dir: str = None,
        revision: str = "main",
    ) -> None:
        """
        Constructs a utility class for downloading and moving the dataset.

        Args:
            dataset (str): The name of the dataset to download
            overwrite_existing (bool): A flag to overwrite existing data. Defaults to `True`.
            cache_dir (str, optional): The directory to save the cache dir. Defaults to `None`.
            revision (str): The revision of the dataset to use. Defaults to "main".
        """
        self.dataset = DatasetConfig(dataset)
        self.overwrite_existing = overwrite_existing
        self.cache_dir = (
            Path(cache_dir) if cache_dir else ArkDatasetSettings.get_cache_dir()
        )
        self.dataset_paths = {}
        self.pooch = create_pooch_instance(revision)
        self.fs = LocalFileSystem(auto_mkdir=True)

    def download_example_dataset(self):
        """Downloads all required dataset files for the selected configuration"""
        required_files: List[DatasetFile] = DATASET_CONFIGS[self.dataset]
        extracted_paths = {}

        for file in required_files:
            # Extract directly into the cache directory
            _ = self.pooch.fetch(file.value, processor=pooch.Unzip(), progressbar=True)
            # Store the path to the extracted data directory
            extracted_paths[file] = file.get_unzipped_path(self.cache_dir)
        self.dataset_paths = {self.dataset: extracted_paths}

    def _process_dataset_file(self, ds_file: DatasetFile, move_dir: Path) -> None:
        """
        Process a single dataset file - downloading and moving it to the destination

        Args:
            ds_file: The dataset file to process
            move_dir: The base directory to move files to
        """
        src_path = Path(self.dataset_paths[self.dataset][ds_file])
        dst_path = ds_file.get_destination_path(move_dir)

        # Get the inner data path (where the actual data is stored)
        inner_src_path = ds_file.get_inner_data_path(src_path)

        # If inner path doesn't exist, try the unzipped directory directly
        if not inner_src_path.exists():
            inner_src_path = src_path

        empty_dst_path = _check_empty_dir(dst_path)

        if self.overwrite_existing:
            if not empty_dst_path:
                warnings.warn(UserWarning(_create_warning_message(dst_path, True)))
                # Remove existing content
                if self.fs.exists(str(dst_path)):
                    self.fs.rm(str(dst_path), recursive=True)

            _copy_directory_contents(inner_src_path, dst_path, self.fs)
        else:
            if empty_dst_path:
                warnings.warn(
                    UserWarning(
                        f"Files do not exist in {dst_path}. "
                        f"The example dataset will be added in."
                    )
                )
                _copy_directory_contents(inner_src_path, dst_path, self.fs)
            else:
                warnings.warn(UserWarning(_create_warning_message(dst_path, False)))

    def move_example_dataset(self, move_dir: Union[str, Path]):
        """
        Moves the downloaded example data from the cache to the specified directory.
        """
        move_dir = Path(move_dir).absolute().resolve()
        dataset_files = list(self.dataset_paths[self.dataset].keys())

        for ds_file in dataset_files:
            self._process_dataset_file(ds_file, move_dir)

        # Clean up the __MACOSX directories
        _clean_macosx_dirs(move_dir, self.fs)


def get_example_dataset(
    dataset: str,
    save_dir: Union[str, Path],
    overwrite_existing: bool = True,
    revision: str = "main",
):
    """
    Downloads a specified dataset and moves it to the specified save directory.

    Args:
        dataset (str): The name of the dataset to download. Must be one of:
            * "segment_image_data"
            * "cluster_pixels"
            * "cluster_cells"
            * "post_clustering"
            * "fiber_segmentation"
            * "LDA_preprocessing"
            * "LDA_training_inference"
            * "neighborhood_analysis"
            * "pairwise_spatial_enrichment"
            * "ome_tiff"
            * "ez_seg_data"
        save_dir (Union[str, Path]): The path to save the dataset files in
        overwrite_existing (bool): Whether to overwrite existing files
        revision (str): The revision of the dataset to use. Defaults to "main".
    """
    valid_datasets = [e.value for e in DatasetConfig]

    try:
        verify_in_list(dataset=dataset, valid_datasets=valid_datasets)
    except ValueError:
        err_str = f"""The dataset "{dataset}" is not one of the valid datasets available.
        The following are available: {*valid_datasets,}"""
        raise ValueError(err_str) from None

    example_dataset = ExampleDataset(
        dataset=dataset, overwrite_existing=overwrite_existing, revision=revision
    )

    example_dataset.download_example_dataset()
    example_dataset.move_example_dataset(move_dir=save_dir)
