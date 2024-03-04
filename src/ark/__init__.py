from importlib.metadata import version

from . import io, pl, pp, tl
from ._core import SelectionAccessor

__all__ = ["pl", "pp", "tl", "io", "SelectionAccessor"]

__version__ = version("ark")
