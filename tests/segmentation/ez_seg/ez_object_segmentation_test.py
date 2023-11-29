import pathlib
from scipy import ndimage
from ark.segmentation.ez_seg import ez_object_segmentation
import pytest
from pytest_cases import param_fixture
import numpy as np
from skimage import draw
import xarray as xr
from skimage.io import imread
from alpineer import image_utils
from skimage.util import img_as_int
from sklearn.preprocessing import minmax_scale


@pytest.fixture(scope="session")
def ez_fov(
    tmp_path_factory: pytest.TempPathFactory, rng: np.random.Generator
) -> pathlib.Path:
    """
    Creates 2 FOVs with 3 channels each, with 60 spots per channel.

    data
    ├── ez_seg_masks
    └── image_data
            ├── fov_0
            │   ├── chan_0.tiff
            │   ├── chan_1.tiff
            │   └── chan_2.tiff
            └── fov_1
                ├── chan_0.tiff
                ├── chan_1.tiff
                └── chan_2.tiff

    Yields:
        pathlib.Path: The path to the temporary directory containing the image_data
        and the masks directory.
    """
    fov_count: int = 2
    channel_count: int = 3
    image_size: int = 1024
    spot_count: int = 60
    spot_radius: int = 40
    cloud_noise_size: int = 4

    image: np.ndarray = rng.normal(
        loc=0.25, scale=0.25, size=(channel_count, image_size, image_size)
    )
    output_image: np.ndarray = np.zeros_like(a=image, dtype=np.int64)

    # Make temporary path
    tmp_path: pathlib.Path = tmp_path_factory.mktemp("data")

    # Make the temporary directory for the image data
    tmp_image_dir = tmp_path / "image_data"
    tmp_image_dir.mkdir(parents=True, exist_ok=True)

    # Make the temporary directory for the masks dir
    tmp_masks_dir = tmp_path / "ez_seg_masks"
    tmp_masks_dir.mkdir(parents=True, exist_ok=True)

    # Make the temporary directory for the log dir
    tmp_log_dir = tmp_path / "ez_logs"
    tmp_log_dir.mkdir(parents=True, exist_ok=True)

    for fov_idx in range(fov_count):
        for channel_idx in range(channel_count):
            channel: np.ndarray = image[channel_idx]
            for _ in range(spot_count):
                rr, cc = draw.disk(
                    center=(
                        rng.integers(channel.shape[0]),
                        rng.integers(channel.shape[1]),
                    ),
                    radius=spot_radius,
                    shape=channel.shape,
                )
                channel[rr, cc] = 1

            channel *= rng.normal(loc=1.0, scale=0.1, size=channel.shape)

            channel *= ndimage.zoom(
                rng.normal(
                    loc=1.0, scale=0.5, size=(cloud_noise_size, cloud_noise_size)
                ),
                image_size / cloud_noise_size,
            )

            int_channel = img_as_int(
                minmax_scale(
                    ndimage.gaussian_filter(
                        channel, sigma=2.0
                    ),
                    feature_range=(-1., 0.999)
                )
            ).astype(np.int64)

            output_image[channel_idx]: np.ndarray = int_channel

        fov_dir = tmp_image_dir / f"fov_{fov_idx}"
        fov_dir.mkdir(parents=True, exist_ok=True)

        for idx, output_channel in enumerate(output_image):
            image_utils.save_image(
                fname=fov_dir / f"chan_{idx}.tiff", data=output_channel
            )

    yield tmp_path


@pytest.mark.parametrize(
    "_min_object_area, _max_object_area, _object_shape_type, _thresh, _fov_dim",
    [(100, 100000, "blob", None, 400), (200, 2000, "projection", 20, 800)],
)
def test_create_object_masks(
    ez_fov: pathlib.Path,
    _object_shape_type: str,
    _thresh: float,
    _fov_dim: int,
    _min_object_area: int,
    _max_object_area: int,
) -> None:
    _sigma = 1
    _hole_size = 10

    with pytest.raises(ValueError):
        ez_object_segmentation.create_object_masks(
            image_data_dir=ez_fov / "image_data",
            img_sub_folder="wrong_sub_folder",
            fov_list=["fov_0", "fov_1"],
            mask_name="test_mask",
            object_shape_type="wrong_shape",
            channel_to_segment="chan_0",
            masks_dir=ez_fov / "ez_seg_masks",
            log_dir=ez_fov / "ez_logs",
        )
    with pytest.raises(FileNotFoundError):
        ez_object_segmentation.create_object_masks(
            image_data_dir="wrong_path",
            img_sub_folder="wrong_sub_folder",
            fov_list=["fov_0", "fov_1"],
            mask_name="test_mask",
            object_shape_type="blob",
            channel_to_segment="chan_0",
            masks_dir=ez_fov / "ez_seg_masks",
            log_dir=ez_fov / "ez_logs",
        )
    # Test the function (succeeds)
    ez_object_segmentation.create_object_masks(
        image_data_dir=ez_fov / "image_data",
        img_sub_folder=None,
        fov_list=["fov_0", "fov_1"],
        mask_name="test_mask",
        object_shape_type=_object_shape_type,
        channel_to_segment="chan_0",
        masks_dir=ez_fov / "ez_seg_masks",
        log_dir=ez_fov / "ez_logs",
        sigma=_sigma,
        thresh=_thresh,
        hole_size=_hole_size,
        fov_dim=_fov_dim,
        min_object_area=_min_object_area,
        max_object_area=_max_object_area,
    )
    assert (ez_fov / "ez_seg_masks" / "fov_0_test_mask.tiff").exists()
    assert (ez_fov / "ez_seg_masks" / "fov_1_test_mask.tiff").exists()
    assert (ez_fov / "ez_logs" / "test_mask_segmentation_log.txt").exists()
    with open(ez_fov / "ez_logs" / "test_mask_segmentation_log.txt", "r") as f:
        log_contents = f.read()
    assert "fov_0" in log_contents
    assert "fov_1" in log_contents
    assert "test_mask" in log_contents
    assert "chan_0" in log_contents
    assert _object_shape_type in log_contents
    assert str(_hole_size) in log_contents
    assert str(_sigma) in log_contents
    assert str(_thresh) in log_contents
    assert str(_fov_dim) in log_contents
    assert str(_min_object_area) in log_contents
    assert str(_max_object_area) in log_contents


@pytest.mark.parametrize(
    "_min_object_area, _max_object_area, _object_shape_type, _thresh, _fov_dim",
    [(100, 100000, "blob", None, 400), (200, 2000, "projection", 100, 800)],
)
def test_create_object_mask(
    ez_fov: pathlib.Path,
    _object_shape_type: str,
    _thresh: float,
    _fov_dim: int,
    _min_object_area: int,
    _max_object_area: int,
) -> None:
    fov0_chan0: np.ndarray = imread(ez_fov / "image_data" / "fov_0" / "chan_0.tiff")

    object_mask = ez_object_segmentation._create_object_mask(
        input_image=xr.DataArray(fov0_chan0),
        object_shape_type=_object_shape_type,
        sigma=1,
        thresh=_thresh,
        hole_size=10,
        fov_dim=_fov_dim,
        min_object_area=_min_object_area,
        max_object_area=_max_object_area,
    )

    assert object_mask.shape == fov0_chan0.shape
    assert object_mask.dtype == np.int32


@pytest.mark.parametrize(
    "_block_type, _fov_dim, _img_shape, _block_size",
    [("small_holes", 400, 512, 316), ("local_thresh", 800, 1024, 13)]
)
def test_get_block_size(
    _block_type: str,
    _fov_dim: int,
    _img_shape: int,
    _block_size: int,
) -> None:
    assert isinstance(
        ez_object_segmentation.get_block_size(
            block_type=_block_type, fov_dim=_fov_dim, img_shape=_img_shape
        ),
        int,
    )
    # Fails with an invalid `block_type`
    with pytest.raises(ValueError):
        ez_object_segmentation.get_block_size(
            block_type="incorrect_block_type", fov_dim=_fov_dim, img_shape=_img_shape
        )
    assert ez_object_segmentation.get_block_size(
        block_type=_block_type,
        fov_dim=_fov_dim,
        img_shape=_img_shape,
    ) == _block_size
