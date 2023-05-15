import pytest
from typing import Iterator
import os


@pytest.fixture(scope="function")
def dataset_cache_dir() -> Iterator[str | None]:
    # Change cache directory if running on CI
    if os.environ.get("CI", None):
        cache_dir = "./data/cache/"
    else:
        cache_dir = None
    yield cache_dir
