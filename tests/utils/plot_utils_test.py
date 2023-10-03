import itertools
import os
import pathlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List
from matplotlib import cm

import matplotlib.colors as colors
import natsort
import numpy as np
import pandas as pd
import pytest
import skimage.io as io
import xarray as xr
from alpineer import image_utils, test_utils, io_utils
from skimage.draw import disk
from matplotlib import colormaps

from ark.utils import plot_utils


def _generate_segmentation_labels(img_dims, num_cells=20, num_imgs=1):
    if len(img_dims) != 2:
        raise ValueError("must be image data of shape [rows, cols]")

    labels: np.ndarray = np.zeros((num_imgs, *img_dims), dtype=np.uint8)
    radius = 20

    for _img, _cell in itertools.product(range(num_imgs), range(num_cells)):
        r, c = np.random.randint(radius, img_dims[0] - radius, 2)
        rr, cc = disk((r, c), radius)
        labels[_img, rr, cc] = _cell

    # if only one image, return a 2-D array
    if labels.shape[0] == 1:
        labels = labels[0]

    return labels


def _generate_image_data(img_dims):
    if len(img_dims) != 3:
        raise ValueError("must be image data of [rows, cols, channels]")

    return np.random.randint(low=0, high=100, size=img_dims)


@pytest.fixture(scope="module")
def metacluster_colors(n_metaclusters: int) -> Generator[Dict, None, None]:
    """
    Generates a dictionary of metacluster colors and yields the dictionary.

    Yields:
        Generator[dict, None, None]: Yields the dictionary of metacluster colors.
    """
    sample_mapping = {i: tuple(np.random.rand(4)) for i in np.arange(n_metaclusters)}

    yield sample_mapping


def test_create_cmap(rng: np.random.Generator):
    n_clusters = 3
    with pytest.raises(ValueError):
        plot_utils.create_cmap(cmap=["DNE"], n_clusters=n_clusters)

    colors_array = rng.random(size=(n_clusters, 4))

    cmap, norm = plot_utils.create_cmap(cmap=colors_array, n_clusters=n_clusters)
    # Assert that the cmap has the 4 clusters, and the background color, and the unassigned_color
    assert cmap.N == n_clusters + 2


@pytest.mark.parametrize("cbar_visible", [True, False])
@pytest.mark.parametrize(
    "cbar_labels", [None, ["Empty", "Cluster 1", "Cluster 2", "Cluster 3"]]
)
def test_plot_cluster(
        rng: np.random.Generator, cbar_visible: bool, cbar_labels: list[str]
):
    cluster_image: np.ndarray = rng.integers(0, 3, size=(10, 10))
    cmap = cm.get_cmap("tab20")
    norm = colors.BoundaryNorm(np.arange(5), cmap.N)

    fig = plot_utils.plot_cluster(
        image=cluster_image,
        fov="test_fov",
        cmap=cmap,
        norm=norm,
        cbar_visible=cbar_visible,
        cbar_labels=cbar_labels,
        figsize=(5, 5),
    )


def test_plot_neighborhood_cluster_result(tmp_path: pathlib.Path):
    sample_img_data = np.random.randint(0, 5, size=(3, 1024, 1024))
    sample_img_xr = xr.DataArray(
        sample_img_data,
        coords=[["fov1", "fov2", "fov3"], np.arange(1024), np.arange(1024)],
        dims=["fovs", "x", "y"],
    )

    n_clusters = len(np.unique(sample_img_data))

    plot_utils.plot_neighborhood_cluster_result(
        img_xr=sample_img_xr,
        fovs=sample_img_xr.fovs.values,
        k=n_clusters,
        save_dir=tmp_path,
        cmap_name="tab20",
        fov_col="fovs",
        dpi=10,
        figsize=(5, 5),
    )

    for fov in sample_img_xr.fovs.values:
        assert os.path.exists(os.path.join(tmp_path, f"{fov}.png"))

    with pytest.raises(ValueError):
        plot_utils.plot_neighborhood_cluster_result(
            img_xr=sample_img_xr,
            fovs=["bad_fov"],
            k=n_clusters,
            save_dir=tmp_path,
            cmap_name="tab20",
            fov_col="fovs",
            dpi=10,
            figsize=(5, 5),
        )


def test_plot_pixel_cell_cluster(metacluster_colors):
    sample_img_data = np.random.randint(0, 20, size=(3, 1024, 1024))
    sample_img_xr = xr.DataArray(
        sample_img_data,
        coords=[["fov1", "fov2", "fov3"], np.arange(1024), np.arange(1024)],
        dims=["fovs", "x", "y"],
    )

    # invalid FOVs provided
    with pytest.raises(ValueError):
        plot_utils.plot_pixel_cell_cluster(
            sample_img_xr, ["fov3", "fov4"], "sample_mapping_path.csv", {}
        )

    # invalid mapping path provided
    with pytest.raises(FileNotFoundError):
        plot_utils.plot_pixel_cell_cluster(
            sample_img_xr, ["fov1", "fov2"], "bad_sample_mapping_path.csv", {}
        )

    with tempfile.TemporaryDirectory() as td:
        # define the path to the cluster map
        cluster_id_to_name_path = os.path.join(td, "sample_mapping_path.csv")

        # invalid columns provided in mapping
        df = pd.DataFrame.from_dict(
            {
                "bad_cluster_col": np.arange(20),
                "pixel_meta_cluster": np.repeat(np.arange(5), 4),
                "pixel_meta_cluster_rename": [
                    "meta" + str(i) for i in np.repeat(np.arange(5), 4)
                ],
            }
        )
        df.to_csv(cluster_id_to_name_path, index=False)

        with pytest.raises(ValueError):
            plot_utils.plot_pixel_cell_cluster(
                sample_img_xr, ["fov1", "fov2"], cluster_id_to_name_path, {}
            )

        # rename bad_cluster_col to cluster so it passes that test
        df = df.rename({"bad_cluster_col": "pixel_som_cluster"}, axis=1)
        df.to_csv(cluster_id_to_name_path, index=False)

        # invalid sample_mapping dict provided, metaclusters do not match
        # those found in mapping_path
        bad_sample_mapping = {i + 2: (0.0, 0.0, 0.0, 1.0) for i in np.arange(5)}

        with pytest.raises(ValueError):
            plot_utils.plot_pixel_cell_cluster(
                sample_img_xr,
                ["fov1", "fov2"],
                cluster_id_to_name_path,
                bad_sample_mapping,
            )

        # test 1: save_dir not specified
        plot_utils.plot_pixel_cell_cluster(
            sample_img_xr, ["fov1", "fov2"], cluster_id_to_name_path, metacluster_colors
        )

        # assert no files created in temp_dir
        for fov in sample_img_xr.fovs.values:
            assert not os.path.exists(os.path.join(td, fov + ".png"))

        # test 2: save_dir specified
        plot_utils.plot_pixel_cell_cluster(
            sample_img_xr,
            ["fov1", "fov2"],
            cluster_id_to_name_path,
            metacluster_colors,
            save_dir=td,
        )

        # assert files only created for fov1`and fov2, not fov3
        assert os.path.exists(os.path.join(td, "fov1.png"))
        assert os.path.exists(os.path.join(td, "fov2.png"))
        assert not os.path.exists(os.path.join(td, "fov3.png"))


def test_tif_overlay_preprocess():
    example_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 3))

    # 2-D tests
    # dimensions are not the same for 2-D example_images
    with pytest.raises(ValueError):
        plot_utils.tif_overlay_preprocess(
            segmentation_labels=example_labels[:100, :100],
            plotting_tif=example_images[..., 0],
        )

    plotting_tif = plot_utils.tif_overlay_preprocess(
        segmentation_labels=example_labels, plotting_tif=example_images[..., 0]
    )

    # assert the channels all contain the same data
    assert np.all(plotting_tif[:, :, 0] == 0)
    assert np.all(plotting_tif[:, :, 1] == 0)
    assert np.all(plotting_tif[:, :, 2] == example_images[..., 0])

    # 3-D tests
    # test for third dimension == 1
    plotting_tif = plot_utils.tif_overlay_preprocess(
        segmentation_labels=example_labels, plotting_tif=example_images[..., 0:1]
    )

    assert np.all(plotting_tif[..., 0] == 0)
    assert np.all(plotting_tif[..., 1] == 0)
    assert np.all(plotting_tif[..., 2] == example_images[..., 0])

    # test for third dimension == 2
    plotting_tif = plot_utils.tif_overlay_preprocess(
        segmentation_labels=example_labels, plotting_tif=example_images[..., 0:2]
    )

    assert np.all(plotting_tif[..., 0] == 0)
    assert np.all(plotting_tif[..., 1] == example_images[..., 1])
    assert np.all(plotting_tif[..., 2] == example_images[..., 0])

    # test for third dimension == 3
    plotting_tif = plot_utils.tif_overlay_preprocess(
        segmentation_labels=example_labels, plotting_tif=example_images
    )

    assert np.all(plotting_tif[..., 0] == example_images[..., 2])
    assert np.all(plotting_tif[..., 1] == example_images[..., 1])
    assert np.all(plotting_tif[..., 2] == example_images[..., 0])

    # test for third dimension == anything else
    with pytest.raises(ValueError):
        # add another layer to the last dimension
        blank_channel = np.zeros(
            example_images.shape[:2] + (1,), dtype=example_images.dtype
        )
        bad_example_images = np.concatenate((example_images, blank_channel), axis=2)

        plot_utils.tif_overlay_preprocess(
            segmentation_labels=example_labels, plotting_tif=bad_example_images
        )

    # n-D test (n > 3)
    with pytest.raises(ValueError):
        # add a fourth dimension
        plot_utils.tif_overlay_preprocess(
            segmentation_labels=example_labels,
            plotting_tif=np.expand_dims(example_images, axis=0),
        )


def test_create_overlay():
    fov = "fov8"

    example_labels = _generate_segmentation_labels((1024, 1024))
    alternate_labels = _generate_segmentation_labels((1024, 1024))
    example_images = _generate_image_data((1024, 1024, 2))

    with tempfile.TemporaryDirectory() as temp_dir:
        # create the whole cell and nuclear segmentation label compartments
        image_utils.save_image(
            os.path.join(temp_dir, "%s_whole_cell.tiff" % fov), example_labels
        )
        image_utils.save_image(
            os.path.join(temp_dir, "%s_nuclear.tiff" % fov), example_labels
        )

        # save the cell image
        img_dir = os.path.join(temp_dir, "img_dir")
        os.mkdir(img_dir)
        image_utils.save_image(os.path.join(img_dir, "%s.tiff" % fov), example_images)

        # test with both nuclear and membrane specified
        contour_mask = plot_utils.create_overlay(
            fov=fov,
            segmentation_dir=temp_dir,
            data_dir=img_dir,
            img_overlay_chans=["nuclear_channel", "membrane_channel"],
            seg_overlay_comp="whole_cell",
        )

        assert contour_mask.shape == (1024, 1024, 3)

        # test with just nuclear specified
        contour_mask = plot_utils.create_overlay(
            fov=fov,
            segmentation_dir=temp_dir,
            data_dir=img_dir,
            img_overlay_chans=["nuclear_channel"],
            seg_overlay_comp="whole_cell",
        )

        assert contour_mask.shape == (1024, 1024, 3)

        # test with nuclear compartment
        contour_mask = plot_utils.create_overlay(
            fov=fov,
            segmentation_dir=temp_dir,
            data_dir=img_dir,
            img_overlay_chans=["nuclear_channel", "membrane_channel"],
            seg_overlay_comp="nuclear",
        )

        assert contour_mask.shape == (1024, 1024, 3)

        # test with an alternate contour
        contour_mask = plot_utils.create_overlay(
            fov=fov,
            segmentation_dir=temp_dir,
            data_dir=img_dir,
            img_overlay_chans=["nuclear_channel", "membrane_channel"],
            seg_overlay_comp="whole_cell",
            alternate_segmentation=alternate_labels,
        )

        assert contour_mask.shape == (1024, 1024, 3)

        # invalid alternate contour provided
        with pytest.raises(ValueError):
            plot_utils.create_overlay(
                fov=fov,
                segmentation_dir=temp_dir,
                data_dir=img_dir,
                img_overlay_chans=["nuclear_channel", "membrane_channel"],
                seg_overlay_comp="whole_cell",
                alternate_segmentation=alternate_labels[:100, :100],
            )


def test_set_minimum_color_for_colormap():
    cols = ["green", "orange", "gold", "blue"]
    color_map = colors.ListedColormap(cols)

    # check minimum color is defaulted to black
    default_color_map = plot_utils.set_minimum_color_for_colormap(color_map)
    assert default_color_map(0.0) == (0.0, 0.0, 0.0, 1.0)

    # check for specific min color
    new_color_map = plot_utils.set_minimum_color_for_colormap(
        color_map, (0.1, 0.2, 0.5, 0.3)
    )
    assert new_color_map(0.0) == (0.1, 0.2, 0.5, 0.3)


@dataclass
class _mantis:
    data_dir: pathlib.Path
    segmentation_dir: str
    mask_dir: str
    cell_output_dir: str
    fov_path: pathlib.Path
    img_data_path: str
    img_sub_folder: str
    mantis_project_path: pathlib.Path
    mask_output_dir: pathlib.Path
    fovs: List[str]
    mask_suffix: str
    df: pd.DataFrame
    mapping_path: str
    data_xr: xr.DataArray
    example_masks: xr.DataArray


@pytest.fixture(scope="function")
def mantis_data(
    tmp_path, cluster_id_to_name_mapping, cluster_type
) -> Generator[_mantis, None, None]:
    """Generates a mantis folder, saves images and segmentation labels to
    an data folder to simulate moving data from the data folder to the mantis
    folder.

    Args:
        tmp_path (pathlib.Path): The temporary path for the mantis data to
        be stored in.
        cluster_id_to_name_mapping (pathlib.Path): The path to the mapping csv file.

    Yields:
        Generator[_mantis, None, None]: Yields the `_mantis` dataclass which houses
        paths, fovs and data for the mantis data.
    """

    data_dir: pathlib.Path = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Number of FOVs
    fov_count = 4

    # Initial data
    example_labels = xr.DataArray(
        [_generate_segmentation_labels((1024, 1024)) for _ in range(fov_count)],
        coords=[range(fov_count), range(1024), range(1024)],
        dims=["labels", "rows", "cols"],
    )
    example_masks = xr.DataArray(
        _generate_image_data((1024, 1024, fov_count)),
        coords=[range(1024), range(1024), range(fov_count)],
        dims=["rows", "cols", "masks"],
    )

    # Paths used
    segmentation_dir: str = "seg_dir"
    mask_dir: str = "masks"
    cell_output_dir: str = f"{cluster_type}_output"
    img_data_path: str = "img_data"
    img_sub_folder: str = "normalized"
    mantis_project_path: str = "mantis"

    for p in [segmentation_dir, cell_output_dir, img_data_path]:
        (data_dir / p).mkdir(parents=True, exist_ok=True)

    # mask output dir path
    mask_output_dir: pathlib.Path = pathlib.Path(data_dir, cell_output_dir, mask_dir)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    # image data path, create 2 fovs, with 4 channels each
    fovs, channels = test_utils.gen_fov_chan_names(
        num_fovs=fov_count, num_chans=4, use_delimiter=False, return_imgs=False
    )

    fov_path: pathlib.Path = data_dir / img_data_path
    filelocs, data_xr = test_utils.create_paired_xarray_fovs(
        fov_path,
        fovs,
        channels,
        img_shape=(10, 10),
        mode="tiff",
        delimiter=None,
        sub_dir=img_sub_folder,
        fills=True,
        dtype=np.int16,
    )

    # Loop over the xarray, save each fov's channels,
    # segmentation label compartments, and sample masks
    fovs = data_xr.fovs.values

    for idx, fov in enumerate(fovs):
        # Save the segmentation label compartments for each fov
        image_utils.save_image(
            os.path.join(data_dir / segmentation_dir, f"{fov}_whole_cell_test.tiff"),
            example_labels.loc[idx, ...].values,
        )

        # Save the sample masks
        image_utils.save_image(
            os.path.join(mask_output_dir, f"{fov}_mask.tiff"),
            example_masks.loc[..., idx].values,
        )

        # Save each channel per fov
        for idx, chan in enumerate(channels):
            image_utils.save_image(
                filelocs[fov][idx] + ".tiff", data_xr.loc[fov, :, :, chan].values
            )

    # The suffix for finding masks
    mask_suffix = "_mask"

    md = _mantis(
        data_dir=data_dir,
        segmentation_dir=segmentation_dir,
        mask_dir=mask_dir,
        cell_output_dir=cell_output_dir,
        fov_path=fov_path,
        img_data_path=img_data_path,
        img_sub_folder=img_sub_folder,
        mantis_project_path=data_dir / mantis_project_path,
        mask_output_dir=mask_output_dir,
        fovs=fovs,
        mask_suffix=mask_suffix,
        df=pd.read_csv(cluster_id_to_name_mapping),
        mapping_path=cluster_id_to_name_mapping,
        data_xr=data_xr,
        example_masks=example_masks,
    )

    yield md


@pytest.mark.parametrize("_mapping", ["df", "mapping_path"])
@pytest.mark.parametrize("_seg_none", [True, False])
@pytest.mark.parametrize("new_suffix", [True, False])
def test_create_mantis_dir(
        mantis_data: _mantis, _seg_none: bool, _mapping: str, cluster_type: str, new_suffix: bool):
    md = mantis_data

    # Image segmentation full path, and None
    if _seg_none:
        image_segmentation_full_path = None
        seg_suffix_name = None
    else:
        image_segmentation_full_path = os.path.join(md.data_dir, md.segmentation_dir)
        seg_suffix_name = "_whole_cell_test.tiff"

    if _mapping == "df":
        _m = md.df
    else:
        _m = md.mapping_path

    mask_suff = md.mask_suffix
    if new_suffix:
        mask_suff = "_new_mask"

    # Test mapping csv, and df
    for mapping in [md.df, md.mapping_path]:
        plot_utils.create_mantis_dir(
            fovs=md.fovs,
            mantis_project_path=md.mantis_project_path,
            img_data_path=md.fov_path,
            mask_output_dir=md.mask_output_dir,
            mask_suffix=md.mask_suffix,
            mapping=_m,
            seg_dir=image_segmentation_full_path,
            cluster_type=cluster_type,
            seg_suffix_name=seg_suffix_name,
            img_sub_folder=md.img_sub_folder,
            new_mask_suffix=mask_suff
        )

        # Testing file existence and correctness
        for idx, fov in enumerate(md.fovs):
            # output path for testing
            output_path = os.path.join(md.mantis_project_path, fov)

            # 1. Mask tiff tests
            mask_path = os.path.join(output_path, f"population{mask_suff}.tiff")
            original_mask_path: str = os.path.join(md.mask_output_dir, '%s_mask.tiff' % fov)

            # 1.a. Assert that the mask path exists
            assert os.path.exists(mask_path)
            mask_img = io.imread(mask_path)
            original_mask_img = md.example_masks.loc[..., idx].values
            # 1.b. Assert that the mask is the same as the original mask
            np.testing.assert_equal(mask_img, original_mask_img)

            # 2. Cell Segmentation tiff tests
            cell_seg_path = os.path.join(output_path, "cell_segmentation.tiff")
            if _seg_none:
                # 2.a Asser that the segmentation label tiff does not exist in the output dir
                assert not os.path.exists(cell_seg_path)
            else:
                # 2.b.i. Assert that the segmentation label compartments exist in the new directory
                assert os.path.exists(cell_seg_path)

                # 2.b.ii Assert that the `cell_segmentation` file is equal to `fov#_whole_cell`
                cell_seg_img = io.imread(cell_seg_path)
                original_cell_seg_path = os.path.join(
                    md.data_dir, md.segmentation_dir, "%s_whole_cell_test.tiff" % fov
                )
                original_cell_seg_img = io.imread(original_cell_seg_path)
                np.testing.assert_equal(cell_seg_img, original_cell_seg_img)

            # 3. mapping csv tests
            if type(mapping) is pd.DataFrame:
                original_mapping_df = md.df
            else:
                original_mapping_df = pd.read_csv(md.mapping_path)
            new_mapping_df = pd.read_csv(
                os.path.join(output_path, f"population{mask_suff}.csv"))

            # 3.a. Assert that metacluster col equals the region_id col
            metacluster_col = original_mapping_df[[f"{cluster_type}_meta_cluster"]]
            region_id_col = new_mapping_df[["region_id"]]
            metacluster_col.eq(region_id_col)

            # 3.b. Assert that mc_name col equals the region_name col
            mc_name_col = original_mapping_df[[f"{cluster_type}_meta_cluster_rename"]]
            region_name = new_mapping_df[["region_name"]]
            mc_name_col.eq(region_name)

            mantis_fov_channels = natsort.natsorted(
                list(Path(output_path).glob("chan*.tiff"))
            )

            # 4. Test that all fov channels exist and are correct
            for chan_path in mantis_fov_channels:
                new_chan = io.imread(chan_path)

                # get the channel name
                chan, _ = chan_path.name.split(".")
                original_chan = md.data_xr.loc[fov, :, :, chan].values
                np.testing.assert_equal(new_chan, original_chan)


@pytest.fixture(scope="session")
def n_metaclusters() -> Generator[int, None, None]:
    """
    Generates the number of meta clusters for pixel and cell metaclustering visualization tasks.

    Yields:
        Generator[int, None, None]: Yields the number of meta clusters.
    """
    yield 5


@pytest.fixture(scope="module", params=["pixel", "cell"])
def cluster_type(request) -> Generator[str, None, None]:
    """
    Generates the cluster type for pixel and cell metaclustering visualization tasks.

    Yields:
        Generator[str, None, None]: Yields the cluster type.
    """
    yield request.param


@pytest.fixture(scope="function")
def cluster_id_to_name_mapping(
    tmp_path, cluster_type, n_metaclusters
) -> Generator[pathlib.Path, None, None]:
    """
    Generates a mapping csv file and yields the path to the csv file.

    Args:
        tmp_path (pathlib.Path): The temporary path for the mapping csv file to
        be stored in.

    Yields:
        Generator[pathlib.Path, None, None]: Yields the path to the mapping csv file.
    """

    mapping_path = tmp_path / "mapping.csv"

    df = pd.DataFrame.from_dict(
        {
            f"{cluster_type}_som_cluster": np.arange(20),
            f"{cluster_type}_meta_cluster": np.repeat(np.arange(n_metaclusters), 4),
            f"{cluster_type}_meta_cluster_rename": [
                "meta" + str(i) for i in np.repeat(np.arange(n_metaclusters), 4)
            ],
        }
    )
    df.to_csv(mapping_path, index=False)

    yield mapping_path


class TestMetaclusterColormap:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        cluster_type: str,
        cluster_id_to_name_mapping: pathlib.Path,
        metacluster_colors: Dict,
        n_metaclusters: int,
    ):
        self.cluster_type = cluster_type
        self.cluster_id_to_name_mapping = cluster_id_to_name_mapping
        self.metacluster_colors = metacluster_colors
        self.n_metaclusters = n_metaclusters

    def test_metacluster_cmap_generator(self):
        # Test by initilizing the MetaclusterColormap class, then __post_init__ will
        # run `_metacluster_cmp_generator`
        mcc = plot_utils.MetaclusterColormap(
            cluster_type=self.cluster_type,
            cluster_id_to_name_path=self.cluster_id_to_name_mapping,
            metacluster_colors=self.metacluster_colors,
        )

        # Assert that the fields after __post_init__ are correct

        # Assert metacluster_to_id_name is correct, contains "`Unassigned`" and `"No Cluster`
        assert set(
            mcc.metacluster_id_to_name[f"{self.cluster_type}_meta_cluster_rename"]
        ) == set([f"meta{i}" for i in range(5)] + ["Unassigned", "Empty"])

        # Assert mc_colors has the correct shape
        # Add 2 colors to account for an element with no associated cluster, and an element with an
        # unassigned cluster
        assert mcc.mc_colors.shape == (self.n_metaclusters + 2, 4)

        # Assert metacluster_to_index is correct
        assert mcc.metacluster_to_index == {
            i: i + 1 for i in range(self.n_metaclusters + 1)
        }

        # Assert the cmap has the correct number of colors
        assert mcc.cmap.N == self.n_metaclusters + 2

        # Assert the boundary norm has the correct number of colors
        # This would be for `plot_neighborhood_cluster_result`, currently not used by
        # `plot_pixel_cell_cluster`
        assert mcc.norm.N == self.n_metaclusters + 3

    def test_assign_metacluster_cmap(self):
        mcc = plot_utils.MetaclusterColormap(
            cluster_type=self.cluster_type,
            cluster_id_to_name_path=self.cluster_id_to_name_mapping,
            metacluster_colors=self.metacluster_colors,
        )
        fov_img: np.ndarray = _generate_segmentation_labels(
            (1024, 1024), num_cells=self.n_metaclusters
        )
        mc_cmap = mcc.assign_metacluster_cmap(fov_img=fov_img)

        # Assert colored mask shape
        assert mc_cmap.shape == (1024, 1024)

        # Assert colored mask dtype
        assert mc_cmap.dtype == np.uint8

        # Assert the number of unique metacluster_cmap values in the colored mask are correct
        assert np.unique(mc_cmap).shape == (self.n_metaclusters,)

        # Assert the unique metacluster_cmap values are correct
        assert np.all(np.unique(mc_cmap) == np.arange(self.n_metaclusters) + 1)


@pytest.fixture(scope="function")
def create_masks(
    tmp_path: pathlib.Path, n_metaclusters: int, cluster_type: str
) -> Generator[pathlib.Path, None, None]:
    """
    Creates a temporary directory with cluster masks and yields the path to the directory.

    Args:
        tmp_path (pathlib.Path): The temporary path for to save the cluster masks in.
        n_metaclusters (int): The number of unique metaclusters to generate
        cluster_type (str): The type of cluster to generate

    Yields:
        Generator[pathlib.Path, None, None]: Yields the path to the directory with the
            cluster masks.
    """
    num_imgs = 2

    cluster_mask_path: pathlib.Path = tmp_path / "cluster_masks"
    cluster_mask_path.mkdir(parents=True, exist_ok=True)

    seg_labeled_img: np.ndarray = _generate_segmentation_labels(
        (1024, 1024), num_cells=n_metaclusters, num_imgs=num_imgs
    )

    for img, img_idx in zip(seg_labeled_img, range(num_imgs)):
        image_utils.save_image(
            cluster_mask_path / f"fov{img_idx}_{cluster_type}_mask.tiff", img
        )
    yield cluster_mask_path


def test_save_colored_mask(tmp_path: pathlib.Path, rng: np.random.Generator):
    cmap, norm = plot_utils.create_cmap(cmap="viridis", n_clusters=5)

    mask_data = rng.integers(0, 5, size=(256, 256))
    fov_name = "fov0"

    save_dir = tmp_path / "save_dir"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_utils.save_colored_mask(
        fov=fov_name, save_dir=save_dir, suffix=".tiff", data=mask_data, cmap=cmap, norm=norm
    )

    assert (save_dir / f"{fov_name}.tiff").exists()


def test_save_colored_masks(
    tmp_path: pathlib.Path,
    create_masks: pathlib.Path,
    cluster_type: str,
    cluster_id_to_name_mapping: pathlib.Path,
    metacluster_colors: Dict,
):
    save_path: pathlib.Path = tmp_path / "save_path"

    fovs = [f"fov{i}" for i in range(2)]
    plot_utils.save_colored_masks(
        fovs=fovs,
        mask_dir=create_masks,
        save_dir=save_path,
        cluster_id_to_name_path=cluster_id_to_name_mapping,
        metacluster_colors=metacluster_colors,
        cluster_type=cluster_type,
    )

    for fov in fovs:
        # Assert that the file exists and is named appropriatly
        colored_mask_path: pathlib.Path = (
            save_path / f"{fov}_{cluster_type}_mask_colored.tiff"
        )
        assert colored_mask_path.exists()

        # Load the colored mask
        colored_mask: np.ndarray = io.imread(fname=colored_mask_path)
        # Assert that the file is the correct shape  (x,y,color)
        assert colored_mask.shape == (1024, 1024, 4)

        # Get the unique colors in the colored mask
        assert np.max(np.unique(colored_mask, axis=None)) <= 255


@dataclass
class CohortClusterData:
    fov_names: list[str]
    seg_dir: pathlib.Path
    fov_col: str
    label_col: str
    cluster_col: str
    cell_data: pd.DataFrame
    cmap_df: pd.DataFrame
    cmap_name: str
    save_dir: pathlib.Path
    seg_suffix: str


@pytest.fixture(scope="function")
def cohort_cluster_data(tmp_path: pathlib.Path):
    # Fov names
    fov_names = [f"fov{i}" for i in range(2)]

    # Segmentation labels
    seg_labels = _generate_segmentation_labels(
        img_dims=(256, 256), num_cells=5, num_imgs=2
    )

    # Save segmentation labels in the seg dir
    seg_dir = tmp_path / "seg_dir"
    seg_dir.mkdir(parents=True, exist_ok=True)
    for seg_label, fov_name in zip(seg_labels, fov_names):
        image_utils.save_image(seg_dir / f"{fov_name}_whole_cell.tiff", seg_label)

    fov_col = "fov_col"
    label_col = "label_col"
    cluster_col = "cluster_col"

    # Cell data table
    cell_data = pd.DataFrame(
        data={
            fov_col: fov_names * 5,
            label_col: np.repeat(np.arange(1, 6), 2),
            cluster_col: np.repeat(np.arange(1, 3), 5),
        }
    )

    # Assign colors on a per cluster basiss
    cmap_df = pd.DataFrame(
        data={
            cluster_col: np.arange(1, 3),
            "color": ["red", "blue"],
        }
    )

    # Assign colors automatically via a colormap
    cmap_name = "viridis"

    # Saving data
    save_dir = tmp_path / "save_dir"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loading specific masks
    seg_suffix = "_whole_cell.tiff"

    yield CohortClusterData(
        fov_names=fov_names,
        seg_dir=seg_dir,
        fov_col=fov_col,
        label_col=label_col,
        cluster_col=cluster_col,
        cell_data=cell_data,
        cmap_df=cmap_df,
        cmap_name=cmap_name,
        save_dir=save_dir,
        seg_suffix=seg_suffix,
    )


@pytest.mark.parametrize("_cmap ,_file_type",
                         [("cmap_df", "png"), ("cmap_name", "pdf"), ("cmap_df", "svg")])
def test_cohort_cluster_plot(
        cohort_cluster_data: CohortClusterData,
        _cmap: str,
        _file_type: str) -> None:
    plot_utils.cohort_cluster_plot(
        fovs=cohort_cluster_data.fov_names,
        save_dir=cohort_cluster_data.save_dir,
        seg_dir=cohort_cluster_data.seg_dir,
        fov_col=cohort_cluster_data.fov_col,
        label_col=cohort_cluster_data.label_col,
        cluster_col=cohort_cluster_data.cluster_col,
        cell_data=cohort_cluster_data.cell_data,
        cmap=cohort_cluster_data.cmap_name
        if _cmap == "cmap_name"
        else cohort_cluster_data.cmap_df,
        fig_file_type=_file_type
    )

    assert (
        cohort_cluster_data.save_dir
        / "cluster_masks"
        / f"{cohort_cluster_data.fov_names[0]}.tiff"
    ).exists()

    assert (
        cohort_cluster_data.save_dir
        / "cluster_masks_colored"
        / f"{cohort_cluster_data.fov_names[0]}.tiff"
    ).exists()

    assert (
        cohort_cluster_data.save_dir
        / "cluster_plots"
        / f"{cohort_cluster_data.fov_names[0]}.{_file_type}"
    ).exists()


@pytest.mark.parametrize("_cbar_visible", [True, False])
def plot_continuous_variable(rng: np.random.Generator, _cbar_visible: bool):
    image: np.ndarray = rng.random(size=(256, 256))
    name = "test_image"

    norm = colors.Normalize(vmin=np.min(image), vmax=np.max(image))

    cmap = "viridis"

    # just make sure the function runs
    fig = plot_utils.plot_continuous_variable(
        image=image,
        name=name,
        norm=norm,
        cmap=cmap,
        cbar_visible=_cbar_visible,
        dpi=300,
        figsize=(5, 5),
    )


@pytest.mark.parametrize("_file_type", ["png", "pdf", "svg"])
def test_color_segmentation_by_stat(
        cohort_cluster_data: CohortClusterData,
        _file_type: str,
):
    plot_utils.color_segmentation_by_stat(
        fovs=[cohort_cluster_data.fov_names[0]],
        data_table=cohort_cluster_data.cell_data,
        seg_dir=cohort_cluster_data.seg_dir,
        save_dir=cohort_cluster_data.save_dir,
        fov_col=cohort_cluster_data.fov_col,
        label_col=cohort_cluster_data.label_col,
        stat_name=cohort_cluster_data.cluster_col,
        fig_file_type=_file_type,
    )

    assert (
        cohort_cluster_data.save_dir
        / "continuous_plots"
        / f"{cohort_cluster_data.fov_names[0]}.{_file_type}"
    ).exists()

    assert (
        cohort_cluster_data.save_dir
        / "colored"
        / f"{cohort_cluster_data.fov_names[0]}.tiff"
    ).exists()
