import pathlib
from typing import ContextManager, Iterator
import shutil
import pytest
from testbook import testbook

from . import notebooks_test_utils


# Sets a shared notebook testing temporary directory. Saves all notebook related files in a
# temporary directory.
@pytest.fixture(scope="session")
def base_dir_generator(tmp_path_factory) -> Iterator[pathlib.Path]:
    """
    A Fixture which creates the directory where the all notebook test inputs and outputs are
    saved.
    Sets the path once for all notebook related tests.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Factory for temporary directories under the
            common base temp directory.

    Yields:
        Iterator[pathlib.Path]: The iterable path containing the path of the `notebook_tests`
        inputs and outputs.
    """
    notebook_test_output_dir = tmp_path_factory.mktemp("notebook_tests")
    yield notebook_test_output_dir


@pytest.fixture(scope="session")
def templates_dir() -> Iterator[pathlib.Path]:
    """
    A Fixture which gathers the `templates` directory from `ark-analysis`.

    Yields:
        Iterator[pathlib.Path]: The directory of the templates relative to this test file.
    """
    templates_dir: pathlib.Path = (pathlib.Path(__file__).resolve()).parents[2] / "templates"
    yield templates_dir


@pytest.fixture(scope="class")
def nb1_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for notebook 1.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    SEGMENT_IMAGE_DATA_PATH: pathlib.Path = templates_dir / "1_Segment_Image_Data.ipynb"
    with testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "nb1"
    shutil.rmtree(base_dir_generator / "nb1")


@pytest.fixture(scope="class")
def nb2_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for notebook 2.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    CLUSTER_PIXELS: pathlib.Path = templates_dir / "2_Pixie_Cluster_Pixels.ipynb"
    with testbook(CLUSTER_PIXELS, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "nb2"
    shutil.rmtree(base_dir_generator / "nb2")


@pytest.fixture(scope="class")
def nb3_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for notebook 3.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    CLUSTER_CELLS: pathlib.Path = templates_dir / "3_Pixie_Cluster_Cells.ipynb"
    with testbook(CLUSTER_CELLS, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "nb3"
    shutil.rmtree(base_dir_generator / "nb3")


@pytest.fixture(scope="class")
def nb3b_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for notebook 3.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    CLUSTER_CELLS: pathlib.Path = templates_dir / "generic_cell_clustering.ipynb"
    with testbook(CLUSTER_CELLS, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "nb3b"
    shutil.rmtree(base_dir_generator / "nb3b")


@pytest.fixture(scope="class")
def nb4_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for notebook 4.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    POST_CLUSTERING: pathlib.Path = templates_dir / "4_Post_Clustering.ipynb"
    with testbook(POST_CLUSTERING, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "nb4"
    shutil.rmtree(base_dir_generator / "nb4")


@pytest.fixture(scope="class")
def nbfib_seg_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for the fiber segmentation notebook.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    EXAMPLE_FIBER_SEGMENTATION: pathlib.Path = templates_dir / "example_fiber_segmentation.ipynb"
    with testbook(EXAMPLE_FIBER_SEGMENTATION, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "efs"
    shutil.rmtree(base_dir_generator / "efs")


@pytest.fixture(scope="class")
def nbmixing_context(templates_dir, base_dir_generator) -> Iterator[ContextManager]:
    """
    Creates a testbook context manager for the mixing score notebook.

    Args:
        templates_dir (pytest.Fixture): The fixture which yields the directory of the notebook
            templates
        base_dir_generator (pytest.Fixture): The fixture which yields the temporary directory
            to store all notebook input / output.

    Yields:
        Iterator[ContextManager]: The testbook context manager which will get cleaned up
            afterwords.
    """
    EXAMPLE_MIXING: pathlib.Path = templates_dir / "Calculate_Mixing_Scores.ipynb"
    with testbook(EXAMPLE_MIXING, timeout=6000, execute=False) as nb_context_manager:
        yield nb_context_manager, base_dir_generator / "cms"
    shutil.rmtree(base_dir_generator / "cms")


class Test_1_Segment_Image_Data:
    """
    Tests Notebook 1 - Segment Image Data for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nb1_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nb1_context[0]
        self.base_dir: pathlib.Path = nb1_context[1]

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = r"{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_file_path(self):
        self.tb.execute_cell("file_path")

    def test_create_dirs(self):
        self.tb.execute_cell("create_dirs")

    def test_validate_path(self):
        self.tb.execute_cell("validate_path")

    def test_load_fovs(self):
        load_fovs_inject = """
            fovs = ["fov0", "fov1"]
        """
        self.tb.inject(load_fovs_inject, "load_fovs")

    def test_nuc_mem_set(self):
        self.tb.execute_cell("nuc_mem_set")

    def test_gen_input(self):
        self.tb.execute_cell("gen_input")

    def test_seg_scale_set(self):
        self.tb.execute_cell("seg_scale_set")

    def test_create_output(self):
        # Get Deepcell Output Dir from the notebook
        deepcell_output_dir = self.tb.ref("deepcell_output_dir")
        fovs = self.tb.ref("fovs")
        # Generate the sample feature_0, feature_1 tiffs
        # Account for the fact that fov0 is 512 x 512
        for fov, dim in zip(fovs, [512, 1024]):
            notebooks_test_utils.generate_sample_feature_tifs(
                [fov], deepcell_output_dir=deepcell_output_dir, img_shape=(dim, dim))

    def test_overlay_mask(self):
        self.tb.execute_cell("overlay_mask")

    def test_save_mask(self):
        self.tb.execute_cell("save_mask")

    def test_nuc_props_set(self):
        self.tb.execute_cell("nuc_props_set")

    def test_create_exp_mat(self):
        self.tb.execute_cell("create_exp_mat")

    def test_save_exp_mat(self):
        self.tb.execute_cell("save_exp_mat")


class Test_2_Pixel_Clustering:
    """
    Tests Notebook 2 - Cluster Pixels for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nb2_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nb2_context[0]
        self.base_dir: pathlib.Path = nb2_context[1]

        # Variables
        self.pixel_prefix = "test"
        self.channels = ["CD3", "CD4", "CD163_nuc_exclude", "ECAD_smoothed"]

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = r"{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_file_path(self):
        self.tb.execute_cell("file_path")

    def test_load_fovs(self):
        load_fovs_inject = """
            fovs = ["fov0", "fov1"]
        """
        self.tb.inject(load_fovs_inject, "load_fovs")

    def test_set_multi(self):
        self.tb.execute_cell("set_multi")

    def test_pixel_prefix(self):
        pixel_prefix_inject = f"""
            pixel_cluster_prefix = "{self.pixel_prefix}"
        """
        self.tb.inject(pixel_prefix_inject, "pixel_prefix")

    def test_dir_set(self):
        self.tb.execute_cell("dir_set")

    def test_smooth_channels(self):
        self.tb.execute_cell("smooth_channels")

    def test_filter_channels(self):
        self.tb.execute_cell("filter_channels")

    def test_channel_set(self):
        channel_set_inject = f"""
            channels = {self.channels}
            blur_factor = 2
            subset_proportion = 0.1
        """
        self.tb.inject(channel_set_inject, "channel_set")

    def test_gen_pixel_mat(self):
        self.tb.execute_cell("gen_pixel_mat")

    def test_pixel_som_path_set(self):
        self.tb.execute_cell("pixel_som_path_set")

    def test_train_pixel_som(self):
        self.tb.execute_cell("train_pixel_som")

    def test_cluster_pixel_mat(self):
        self.tb.execute_cell("cluster_pixel_mat")

    def test_pixel_consensus_cluster(self):
        self.tb.execute_cell("pixel_consensus_cluster")

    def test_pixel_interactive(self):
        self.tb.execute_cell("pixel_interactive")

    def test_pixel_apply_remap(self):
        # Get pixel paths
        pixel_meta_cluster_remap = self.tb.ref("pixel_meta_cluster_remap_name")

        notebooks_test_utils.create_pixel_remap_files(self.base_dir, pixel_meta_cluster_remap)

        self.tb.execute_cell("pixel_apply_remap")

    def test_pixel_cmap_gen(self):
        self.tb.execute_cell("pixel_cmap_gen")

    def test_pixel_overlay_fovs(self):
        self.tb.execute_cell("pixel_overlay_fovs")

    def test_pixel_mask_gen_save(self):
        self.tb.execute_cell("pixel_mask_gen_save")

    def test_pixel_overlay_gen(self):
        self.tb.execute_cell("pixel_overlay_gen")

    def test_cell_param_save(self):
        self.tb.execute_cell("cell_param_save")

    def test_pixel_mantis_project(self):
        self.tb.execute_cell("pixel_mantis_project")


class Test_3_Cell_Clustering:
    """
    Tests Notebook 3 - Cluster Cells for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nb3_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nb3_context[0]
        self.base_dir: pathlib.Path = nb3_context[1]

        # Variables
        self.cell_prefix = "test"

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = r"{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_dir_set(self):
        self.tb.execute_cell("dir_set")

    def test_param_load(self):
        self.tb.execute_cell("param_load")

    def test_cluster_prefix(self):
        cell_prefix_inject = f"""
        cell_cluster_prefix = "{self.cell_prefix}"
        """
        self.tb.inject(cell_prefix_inject, "cluster_prefix")

    def test_cell_cluster_files(self):
        self.tb.execute_cell("cell_cluster_files")

    def test_pixel_cluster_col(self):
        self.tb.execute_cell("pixel_cluster_col")

    def test_generate_som_input_data(self):
        self.tb.execute_cell("generate_som_input_data")

    def test_generate_weighted_channel_data(self):
        self.tb.execute_cell("generate_weighted_channel_data")

    def test_train_cell_com(self):
        self.tb.execute_cell("train_cell_som")

    def test_cluster_cell_data(self):
        self.tb.execute_cell("cluster_cell_data")

    def test_cell_consensus_cluster(self):
        self.tb.execute_cell("cell_consensus_cluster")

    def test_cell_interactive(self):
        self.tb.execute_cell("cell_interactive")

    def test_cell_apply_remap(self):
        # Get cell paths
        cell_meta_cluster_remap = self.tb.ref("cell_meta_cluster_remap_name")

        notebooks_test_utils.create_cell_remap_files(self.base_dir, cell_meta_cluster_remap)

        self.tb.execute_cell("cell_apply_remap")

    def test_cell_cmap_gen(self):
        self.tb.execute_cell("cell_cmap_gen")

    def test_cell_som_heatmap(self):
        self.tb.execute_cell("cell_som_heatmap")

    def test_cell_meta_heatmap(self):
        self.tb.execute_cell("cell_meta_heatmap")

    def test_cell_overlay_fovs(self):
        self.tb.execute_cell("cell_overlay_fovs")

    def test_cell_mask_gen_save(self):
        self.tb.execute_cell("cell_mask_gen_save")

    def test_cell_overlay_gen(self):
        self.tb.execute_cell("cell_overlay_gen")

    def test_cell_append_meta(self):
        self.tb.execute_cell("cell_append_meta")

    def test_pixie_cell_save(self):
        self.tb.execute_cell("pixie_cell_save")

    def test_cell_mantis_project(self):
        self.tb.execute_cell("cell_mantis_project")


class Test_3b_Generic_Cell_Clustering:
    """
    Tests Notebook 3 - Cluster Cells for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nb3b_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nb3b_context[0]
        self.base_dir: pathlib.Path = nb3b_context[1]

        # Variables
        self.cell_prefix = "test"

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = "{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_input_set(self):
        self.tb.execute_cell("input_set")

    def test_param_load(self):
        self.tb.execute_cell("param_set")

    def test_cluster_prefix(self):
        cell_prefix_inject = f"""
        cell_cluster_prefix = "{self.cell_prefix}"
        """
        self.tb.inject(cell_prefix_inject, "cluster_prefix")

    def test_cell_cluster_files(self):
        self.tb.execute_cell("cell_cluster_files")

    def test_train_cell_som(self):
        self.tb.execute_cell("train_cell_som")

    def test_cluster_cell_data(self):
        self.tb.execute_cell("cluster_cell_data")

    def test_cell_consensus_cluster(self):
        self.tb.execute_cell("cell_consensus_cluster")

    def test_cell_interactive(self):
        self.tb.execute_cell("cell_interactive")

    def test_cell_apply_remap(self):
        # Get cell paths
        cell_meta_cluster_remap = self.tb.ref("cell_meta_cluster_remap_name")

        notebooks_test_utils.create_cell_remap_files(self.base_dir, cell_meta_cluster_remap)

        self.tb.execute_cell("cell_apply_remap")

    def test_cell_cmap_gen(self):
        self.tb.execute_cell("cell_cmap_gen")

    def test_cell_overlay_fovs(self):
        self.tb.execute_cell("cell_overlay_fovs")

    def test_cell_mask_gen_save(self):
        self.tb.execute_cell("cell_mask_gen_save")

    def test_cell_overlay_gen(self):
        self.tb.execute_cell("cell_overlay_gen")

    def test_cell_append_meta(self):
        self.tb.execute_cell("cell_append_meta")

    def test_pixie_cell_save(self):
        self.tb.execute_cell("pixie_cell_save")

    def test_cell_mantis_project(self):
        self.tb.execute_cell("cell_mantis_project")


class Test_4_Post_Clustering:
    """
    Tests Notebook 4 - Post Clustering for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nb4_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nb4_context[0]
        self.base_dir: pathlib.Path = nb4_context[1]

        # Variables
        self.cell_prefix = "test"

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = r"{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_file_path(self):
        self.tb.execute_cell("file_path")

    def test_dir_set(self):
        self.tb.execute_cell("dir_set")

    def test_identify_threshold(self):
        self.tb.execute_cell("identify_threshold")

    def test_split_problematic_clusters(self):
        self.tb.execute_cell("split_problematic_clusters")

    def test_mantis_manual_inspection(self):
        self.tb.execute_cell("mantis_manual_inspection")

    def test_final_cluster_assignment(self):
        self.tb.execute_cell("final_cluster_assignment")

    def test_updated_cell_table(self):
        self.tb.execute_cell("updated_cell_table")

    def test_marker_thresholding_vars(self):
        self.tb.execute_cell("marker_thresholding_vars")

    def test_mantis_marker_counts(self):
        self.tb.execute_cell("mantis_marker_counts")

    def test_marker_threshold_range(self):
        self.tb.execute_cell("marker_threshold_range")

    def test_threshold_list_vars(self):
        self.tb.execute_cell("threshold_list_vars")

    def test_cell_table_threshold(self):
        self.tb.execute_cell("cell_table_threshold")


class Test_Fiber_Segmentation():
    """
    Tests Example Fiber Segmentation for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nbfib_seg_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nbfib_seg_context[0]
        self.base_dir: pathlib.Path = nbfib_seg_context[1]

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = r"{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_file_paths(self):
        self.tb.execute_cell("file_paths")

    def test_param_set(self):
        self.tb.execute_cell("param_set")

    def test_plot_fiber_segmentation(self):
        self.tb.execute_cell("plot_fiber_segmentation")

    def test_run_fiber_segmentation(self):
        self.tb.execute_cell("run_fiber_segmentation")


class Test_Mixing_Score():
    """
    Tests Example Mixing Score for completion.
    NOTE: When modifying the tests, make sure the test are in the
    same order as the tagged cells in the notebook.
    """
    @pytest.fixture(autouse=True, scope="function")
    def _setup(self, nbmixing_context):
        """
        Sets up necessary data and paths to run the notebooks.
        """
        self.tb: testbook = nbmixing_context[0]
        self.base_dir: pathlib.Path = nbmixing_context[1]

    def test_imports(self):
        self.tb.execute_cell("import")

    def test_base_dir(self):
        base_dir_inject = f"""
            base_dir = r"{self.base_dir}"
        """
        self.tb.inject(base_dir_inject, "base_dir")

    def test_ex_data_download(self):
        self.tb.execute_cell("ex_data_download")

    def test_file_paths(self):
        self.tb.execute_cell("file_path")

    def test_create_dirs(self):
        self.tb.execute_cell("create_dirs")

    def test_cell_table(self):
        self.tb.execute_cell("cell_table")

    def test_cell_neighbors(self):
        self.tb.execute_cell("cell_neighbors")

    def test_mixing_type(self):
        self.tb.execute_cell("mixing_type")

    def test_cell_types(self):
        self.tb.execute_cell("cell_types")

    def test_ratio_plots(self):
        self.tb.execute_cell("ratio_plots")

    def test_mixing_args(self):
        self.tb.execute_cell("mixing_args")

    def test_mixing_score(self):
        self.tb.execute_cell("mixing_score")
