from collections import OrderedDict
from collections.abc import Callable

import spatialdata as sd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray.core.extensions import _register_accessor


def register_spatial_data_accessor(name: str) -> Callable[[str], sd.SpatialData]:
    """Hijacks xarray _register_accessor to register a SpatialData accessor.

    Used as a decorator to register a SpatialData accessor.

    Source: github/scverse/spatialdata-plot/src/spatialdata_plot/_accessor.py


    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    Callable[[str], sd.SpatialData]
        The accessor.

    """
    return _register_accessor(name, sd.SpatialData)


def _verify_plotting_tree(sdata: sd.SpatialData) -> sd.SpatialData:
    """Verify that the plotting tree exists, and if not, create it."""
    if not hasattr(sdata, "plotting_tree"):
        sdata.plotting_tree = OrderedDict()
    return sdata

class SpatialDataAccessor:
    """Used for creating `SpatialData` accessors.

    Any accessor should inherit this class in order to gain access to the `SpatialData` object.

    """

    @property
    def sdata(self) -> sd.SpatialData:
        """The `SpatialData` object to provide preprocessing functions for."""
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: sd.SpatialData) -> None:
        """The `SpatialData` object.

        Parameters
        ----------
        sdata : sd.SpatialData
            Sets the `SpatialData` object.
        """
        self._sdata = sdata

    def __init__(self, sdata: sd.SpatialData) -> None:
        self.sdata = sdata

    def _copy(
        self,
        images: None | dict[str, SpatialImage | MultiscaleSpatialImage] = None,
        labels: None | dict[str, SpatialImage | MultiscaleSpatialImage] = None,
        points: None | dict[str, DaskDataFrame] = None,
        shapes: None | dict[str, GeoDataFrame] = None,
        tables: None | AnnData = None,
    ) -> sd.SpatialData:
        """
        Copy the references from the original to the new SpatialData object.

        Parameters
        ----------
        images : None | dict[str, SpatialImage  |  MultiscaleSpatialImage], optional
            The images in the `SpatialData` object, by default None
        labels : None | dict[str, SpatialImage  |  MultiscaleSpatialImage], optional
            The labels in the `SpatialData` object, by default None
        points : None | dict[str, DaskDataFrame], optional
            The points in the `SpatialData` object, by default None
        shapes : None | dict[str, GeoDataFrame], optional
            The points in the `SpatialData` object, by default None
        tables : None | dict[str, AnnData], optional
            The Anndata Table in the `SpatialData` object, by default None

        Returns
        -------
        sd.SpatialData
            A new `SpatialData` object with the same references as the original `SpatialData` object.
        """
        sdata = sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            tables=self._sdata.tables if tables is None else tables,
        )
        sdata.plotting_tree = (
            self._sdata.plotting_tree if hasattr(self._sdata, "plotting_tree") else OrderedDict()
        )

        return sdata
