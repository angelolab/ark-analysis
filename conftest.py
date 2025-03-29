import os
from pathlib import Path
from typing import Generator, Iterator, Union

import numpy as np
import pytest
from ark.utils.example_dataset import ArkDatasetSettings


@pytest.fixture(scope="session")
def dataset_cache_dir() -> Iterator[Union[str, None]]:
    # Change cache directory if running on CI
    if os.environ.get("CI", False):
        cache_dir = (Path(os.environ.get("GITHUB_WORKSPACE")) / "data" / "cache").resolve()
    else:
        cache_dir = ArkDatasetSettings.get_cache_dir().expanduser().resolve()
    yield cache_dir


@pytest.fixture(scope="session")
def rng() -> Generator[np.random.Generator, None, None]:
    """
    Create a new Random Number Generator for tests which require randomized data.

    Yields:
        Generator[np.random.Generator, None, None]: The generator used for creating randomized
        numbers.
    """
    rng: np.random.Generator = np.random.default_rng(12345)
    yield rng
