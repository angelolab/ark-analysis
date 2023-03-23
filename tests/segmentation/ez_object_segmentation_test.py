from scipy import ndimage
from ark.segmentation import ez_object_segmentation as ezseg
import pytest
from pytest_cases import param_fixture
from typing import Iterator
import numpy as np
from skimage import draw


@pytest.fixture(scope="module")
def input_image() -> Iterator[np.ndarray]:
    """
    Creates a 1024 by 1024 image with random noise, uneven illumination and spots.

    Yields:
        Iterator[np.ndarray]: The example image.
    """
    image_size: int = 1024
    spot_count: int = 60
    spot_radius: int = 40
    cloud_noise_size: int = 4

    # Initialize a new generator - set seed for reproducibility
    rng: np.random.Generator = np.random.default_rng(12345)
    image: np.ndarray = rng.normal(
        loc=0.25,
        scale=0.25,
        size=(image_size, image_size)
    )

    for _ in range(spot_count):
        rr, cc = draw.disk(
            center=(rng.integers(image.shape[0]),
                    rng.integers(image.shape[1])),
            radius=spot_radius,
            shape=image.shape
        )
        image[rr, cc] = 1

    image *= rng.normal(loc=1.0, scale=0.1, size=image.shape)

    image *= ndimage.zoom(
        rng.normal(
            loc=1.0,
            scale=0.5,
            size=(cloud_noise_size, cloud_noise_size)
        ),
        image_size / cloud_noise_size
    )

    output_image: np.ndarray = ndimage.gaussian_filter(image, sigma=2.0)

    yield output_image


_fov_dim = param_fixture(
    "_fov_dim",
    [400, 800]
)


@pytest.mark.parametrize(
    "_min_object_area, _max_object_area, _object_shape_type",
    [(100, 100000, "blob"),
     (200, 2000, "projection")]
)
def test_create_object_masks(
        input_image: Iterator[np.ndarray],
        _object_shape_type: str,
        _fov_dim: int,
        _min_object_area: int,
        _max_object_area: int,
) -> None:
    _sigma = 1
    _thresh = 0.1
    _hole_size = 10
    # Test valid inputs
    output_object_mask: np.ndarray = ezseg.create_object_masks(
        input_image=input_image, object_shape_type=_object_shape_type, sigma=_sigma,
        thresh=_thresh, hole_size=_hole_size, fov_dim=_fov_dim, min_object_area=_min_object_area,
        max_object_area=_max_object_area)
    assert output_object_mask.shape == input_image.shape
    assert np.issubdtype(output_object_mask.dtype, np.integer)

    # Test invalid object_shape_type
    with pytest.raises(ValueError):
        output_object_mask: np.ndarray = ezseg.create_object_masks(
            input_image=input_image, object_shape_type="incorrect_ost", sigma=_sigma,
            thresh=_thresh, hole_size=_hole_size, fov_dim=_fov_dim,
            min_object_area=_min_object_area, max_object_area=_max_object_area)


@pytest.mark.parametrize(
    "_block_type, _img_shape",
    [("small_holes", 512), ("local_thresh", 1024)]
)
def test_get_block_size(_block_type: str, _fov_dim: int, _img_shape: int,) -> None:

    assert isinstance(ezseg.get_block_size(block_type=_block_type,
                                           fov_dim=_fov_dim,
                                           img_shape=_img_shape), int)
    # Fails with an invalid `block_type`
    with pytest.raises(ValueError):
        ezseg.get_block_size(block_type="incorrect_block_type",
                             fov_dim=_fov_dim,
                             img_shape=_img_shape)
