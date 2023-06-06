from typing import Generator

import numpy as np
import pytest


@pytest.fixture(scope="module")
def rng() -> Generator[np.random.Generator, None, None]:
    """
    Create a new Random Number Generator for tests which require randomized data.

    Yields:
        Generator[np.random.Generator, None, None]: The generator used for creating randomized
        numbers.
    """
    rng: np.random.Generator = np.random.default_rng(12345)
    yield rng
