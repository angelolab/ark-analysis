import modguard

from .accessor import SpatialDataAccessor, register_spatial_data_accessor
from .indexing import SelectionAccessor

# modguard.boundary()

__all__ =[
    "register_spatial_data_accessor",
    "SpatialDataAccessor",
    "SelectionAccessor"
]
