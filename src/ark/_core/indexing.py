from collections.abc import Iterable

import spatialdata as sd
from spatialdata.models import get_table_keys
from spatialdata.transformations import get_transformation

from ark._core.accessor import SpatialDataAccessor, register_spatial_data_accessor


def _get_coordinate_system_mapping(sdata: sd.SpatialData) -> dict[str, list[str]]:
    coordsys_keys = sdata.coordinate_systems
    image_keys = [] if sdata.images is None else sdata.images.keys()
    label_keys = [] if sdata.labels is None else sdata.labels.keys()
    shape_keys = [] if sdata.shapes is None else sdata.shapes.keys()
    point_keys = [] if sdata.points is None else sdata.points.keys()
    mapping: dict[str, list[str]] = {}
    if len(coordsys_keys) < 1:
        raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")
    for key in coordsys_keys:
        mapping[key] = []
        for image_key in image_keys:
            transformations = get_transformation(sdata.images[image_key], get_all=True)
            if key in list(transformations.keys()):
                mapping[key].append(image_key)
        for label_key in label_keys:
            transformations = get_transformation(sdata.labels[label_key], get_all=True)
            if key in list(transformations.keys()):
                mapping[key].append(label_key)
        for shape_key in shape_keys:
            transformations = get_transformation(sdata.shapes[shape_key], get_all=True)
            if key in list(transformations.keys()):
                mapping[key].append(shape_key)
        for point_key in point_keys:
            transformations = get_transformation(sdata.points[point_key], get_all=True)
            if key in list(transformations.keys()):
                mapping[key].append(point_key)
    return mapping



def _get_elements(sdata: sd.SpatialData, elements: str | Iterable[str]) -> sd.SpatialData:
        """
        Get a subset of the spatial data object by specifying elements to keep.

        Parameters
        ----------
        elements :
            A string or a list of strings specifying the elements to keep.
            Valid element types are:

            - 'coordinate_systems'
            - 'images'
            - 'labels'
            - 'shapes'
            - 'tables'
            - 'points'

        Returns
        -------
        sd.SpatialData
            A new spatial data object containing only the specified elements.

        Raises
        ------
        TypeError
            If `elements` is not a string or a list of strings.
            If `elements` is a list of strings but one or more of the strings
            are not valid element types.

        ValueError
            If any of the specified elements is not present in the original
            spatialdata object.

        AssertionError
            If `label_keys` is not an empty list but the spatial data object
            does not have a table or the table does not have 'uns' or 'obs'
            attributes.

        Notes
        -----
        If the original spatialdata object has a table, and `elements`
        includes label keys, the returned spatialdata object will have a
        subset of the original table with only the rows corresponding to the
        specified label keys. The `region` attribute of the returned spatial
        data object's table will be set to the list of specified label keys.

        If the original spatial data object has no table, or if `elements` does
        not include label keys, the returned spatialdata object will have no
        table.
        """
        if not isinstance(elements, str | list):
            raise TypeError("Parameter 'elements' must be a string or a list of strings.")

        if not all(isinstance(e, str) for e in elements):
            raise TypeError("When parameter 'elements' is a list, all elements must be strings.")

        if isinstance(elements, str):
            elements = [elements]

        coord_keys = []
        image_keys = []
        label_keys = []
        shape_keys = []
        point_keys = []

        # prepare list of valid keys to sort elements on
        valid_coord_keys = (
            sdata.coordinate_systems if hasattr(sdata, "coordinate_systems") else None
        )
        valid_image_keys = (
            list(sdata.images.keys()) if hasattr(sdata, "images") else None
        )
        valid_label_keys = (
            list(sdata.labels.keys()) if hasattr(sdata, "labels") else None
        )
        valid_shape_keys = (
            list(sdata.shapes.keys()) if hasattr(sdata, "shapes") else None
        )
        valid_point_keys = (
            list(sdata.points.keys()) if hasattr(sdata, "points") else None
        )
        # first, extract coordinate system keys because they generate implicit keys
        mapping = _get_coordinate_system_mapping(sdata)
        implicit_keys = []
        for e in elements:
            for valid_coord_key in valid_coord_keys:
                if (valid_coord_keys is not None) and (e == valid_coord_key):
                    coord_keys.append(e)
                    implicit_keys += mapping[e]

        for e in elements + implicit_keys:
            found = False

            if valid_coord_keys is not None:
                for valid_coord_key in valid_coord_keys:
                    if e == valid_coord_key:
                        coord_keys.append(e)
                        found = True

            if valid_image_keys is not None:
                for valid_image_key in valid_image_keys:
                    if e == valid_image_key:
                        image_keys.append(e)
                        found = True

            if valid_label_keys is not None:
                for valid_label_key in valid_label_keys:
                    if e == valid_label_key:
                        label_keys.append(e)
                        found = True

            if valid_shape_keys is not None:
                for valid_shape_key in valid_shape_keys:
                    if e == valid_shape_key:
                        shape_keys.append(e)
                        found = True

            if valid_point_keys is not None:
                for valid_point_key in valid_point_keys:
                    if e == valid_point_key:
                        point_keys.append(e)
                        found = True

            if not found:
                msg = f"Element '{e}' not found. Valid choices are:"
                if valid_coord_keys is not None:
                    msg += "\n\ncoordinate_systems\n├ "
                    msg += "\n├ ".join(valid_coord_keys)
                if valid_image_keys is not None:
                    msg += "\n\nimages\n├ "
                    msg += "\n├ ".join(valid_image_keys)
                if valid_label_keys is not None:
                    msg += "\n\nlabels\n├ "
                    msg += "\n├ ".join(valid_label_keys)
                if valid_shape_keys is not None:
                    msg += "\n\nshapes\n├ "
                    msg += "\n├ ".join(valid_shape_keys)
                raise ValueError(msg)

        if (valid_coord_keys is not None) and (len(coord_keys) > 0):
            sdata = sdata.filter_by_coordinate_system(coord_keys, filter_tables=False)

        elif len(coord_keys) == 0:
            if valid_image_keys is not None:
                if len(image_keys) == 0:
                    for valid_image_key in valid_image_keys:
                        del sdata.images[valid_image_key]
                elif len(image_keys) > 0:
                    for valid_image_key in valid_image_keys:
                        if valid_image_key not in image_keys:
                            del sdata.images[valid_image_key]

            if valid_label_keys is not None:
                if len(label_keys) == 0:
                    for valid_label_key in valid_label_keys:
                        del sdata.labels[valid_label_key]
                elif len(label_keys) > 0:
                    for valid_label_key in valid_label_keys:
                        if valid_label_key not in label_keys:
                            del sdata.labels[valid_label_key]

            if valid_shape_keys is not None:
                if len(shape_keys) == 0:
                    for valid_shape_key in valid_shape_keys:
                        del sdata.shapes[valid_shape_key]
                elif len(shape_keys) > 0:
                    for valid_shape_key in valid_shape_keys:
                        if valid_shape_key not in shape_keys:
                            del sdata.shapes[valid_shape_key]

            if valid_point_keys is not None:
                if len(point_keys) == 0:
                    for valid_point_key in valid_point_keys:
                        del sdata.points[valid_point_key]
                elif len(point_keys) > 0:
                    for valid_point_key in valid_point_keys:
                        if valid_point_key not in point_keys:
                            del sdata.points[valid_point_key]

        # subset table if it is present and the region key is a valid column
        if len(sdata.tables) != 0 and len(shape_keys + label_keys + point_keys) > 0:
            for name, table in sdata.tables.items():
                assert hasattr(table, "uns"), "Table in SpatialData object does not have 'uns'."
                assert hasattr(table, "obs"), "Table in SpatialData object does not have 'obs'."

                # create mask of used keys
                _, region_key, _ = get_table_keys(table)
                mask = table.obs[region_key]
                mask = list(mask.str.contains("|".join(shape_keys + label_keys)))

                # create copy and delete original so we can reuse slot
                old_table = table.copy()
                new_table = old_table[mask, :].copy()
                new_table.uns["spatialdata_attrs"]["region"] = list(set(new_table.obs[region_key]))
                sdata.tables[name] = new_table

        else:
            sdata.tables = {}

        return sdata



@register_spatial_data_accessor("sel")
class SelectionAccessor(SpatialDataAccessor):
    # def __init__(self, sdata):
    #     super().__init__(sdata)

    def __call__(self, **kwargs) -> sd.SpatialData:
        # copy that we hard-modify, then get the elements
        return _get_elements(sdata = self._copy(), **kwargs)
