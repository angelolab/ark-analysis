import pytest
from typing import Iterator, Union
import os


@pytest.fixture(scope="session")
def dataset_cache_dir() -> Iterator[Union[str,None]]:
    # Change cache directory if running on CI
    if os.environ.get("CI", None):
        cache_dir = "./data/cache/"
    else:
        cache_dir = None
    yield cache_dir
