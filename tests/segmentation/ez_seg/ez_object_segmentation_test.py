import pathlib
from scipy import ndimage
from ark.segmentation.ez_seg import ez_object_segmentation
import pytest
from pytest_cases import param_fixture
import numpy as np
from skimage import draw
import xarray as xr
from alpineer import image_utils


@pytest.fixture(scope="module")
def ez_fov(
    tmpdir_factory: pytest.TempPathFactory, rng: np.random.Generator
) -> pathlib.Path:
    """
    Creates an DataArray with a 1024 by 1024 image with random noise, uneven illumination
    and spots.

    Yields:
        pathlib.Path: The path to the FOV.
    """
    channel_count: int = 3
    image_size: int = 1024
    spot_count: int = 60
    spot_radius: int = 40
    cloud_noise_size: int = 4

    image: np.ndarray = rng.normal(
        loc=0.25, scale=0.25, size=(channel_count, image_size, image_size)
    )
    output_image: np.ndarray = np.zeros_like(a=image)

    for channel_idx in range(channel_count):
        channel: np.ndarray = image[channel_idx]

        for _ in range(spot_count):
            rr, cc = draw.disk(
                center=(rng.integers(channel.shape[0]), rng.integers(channel.shape[1])),
                radius=spot_radius,
                shape=channel.shape,
            )
            channel[rr, cc] = 1

        channel *= rng.normal(loc=1.0, scale=0.1, size=channel.shape)

        channel *= ndimage.zoom(
            rng.normal(loc=1.0, scale=0.5, size=(cloud_noise_size, cloud_noise_size)),
            image_size / cloud_noise_size,
        )

        output_image[channel_idx]: np.ndarray = ndimage.gaussian_filter(
            channel, sigma=2.0
        )

    # Make temporary path
    tmp_path: pathlib.Path = tmpdir_factory.mktemp("data")

    for idx, output_channel in enumerate(output_image):
        image_utils.save_image(fname=tmp_path / f"chan_{idx}.tiff", data=output_channel)

    yield tmp_path


_fov_dim = param_fixture("_fov_dim", [400, 800])


@pytest.mark.skip(reason="WIP")
@pytest.mark.parametrize(
    "_min_object_area, _max_object_area, _object_shape_type, _thresh",
    [(100, 100000, "blob", None), (200, 2000, "projection", 0.1)],
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
    # Test valid inputs
    # output_object_masks: xr.DataArray = ez_object_segmentation.create_object_masks(,
    # assert output_object_masks.shape == (3, 1024, 1024)

    # # Test invalid object_shape_type
    # with pytest.raises(ValueError):
    #     output_object_masks: np.ndarray = ez_object_segmentation.create_object_masks(,


@pytest.mark.parametrize(
    "_block_type, _img_shape", [("small_holes", 512), ("local_thresh", 1024)]
)
def test_get_block_size(
    _block_type: str,
    _fov_dim: int,
    _img_shape: int,
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
