import os
import pathlib
from typing import Generator, Iterator, Union

import numpy as np
import pytest


@pytest.fixture(scope="session")
def dataset_cache_dir() -> Iterator[Union[str, None]]:
    # Change cache directory if running on CI
    if os.environ.get("CI", None):
        cache_dir = pathlib.Path("./data/cache/")
    else:
        cache_dir = None
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
