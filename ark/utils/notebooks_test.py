import subprocess
import os
import tempfile

from testbook import testbook
from tempfile import TemporaryDirectory as tdir

from ark.utils import notebooks_test_utils


SEGMENT_IMAGE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..', '..', 'templates', 'Segment_Image_Data.ipynb')

FLOWSOM_CLUSTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    '..', '..', 'templates', 'example_flowsom_clustering.ipynb')


def _exec_notebook(nb_filename):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '..', '..', 'templates', nb_filename)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


# test runs with default inputs
def test_segment_image_data():
    _exec_notebook('Segment_Image_Data.ipynb')


def test_example_spatial_analysis():
    _exec_notebook('example_spatial_analysis_script.ipynb')


def test_example_neighborhood_analysis():
    _exec_notebook('example_neighborhood_analysis_script.ipynb')


# test mibitiff segmentation
@testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000)
def test_segment_image_data_mibitiff(tb):
    with tdir() as tiff_dir, tdir() as input_dir, tdir() as output_dir, \
         tdir() as single_cell_dir, tdir() as viz_dir:
        # create input files
        notebooks_test_utils.segment_notebook_setup(tb,
                                                    deepcell_tiff_dir=tiff_dir,
                                                    deepcell_input_dir=input_dir,
                                                    deepcell_output_dir=output_dir,
                                                    single_cell_dir=single_cell_dir,
                                                    viz_dir=viz_dir,
                                                    is_mibitiff=True)

        # hard coded fov setting
        notebooks_test_utils.fov_channel_input_set(
            tb,
            fovs=['fov0_otherinfo-MassCorrected-Filtered.tiff',
                  'fov1-MassCorrected-Filtered.tiff'],
            nucs_list=['chan0'],
            mems_list=['chan1', 'chan2'])

        # generate _feature_0 tif files that would normally be handled by create_deepcell_output
        notebooks_test_utils.generate_sample_feature_tifs(
            fovs=['fov0_otherinfo-MassCorrected-Filtered', 'fov1-MassCorrected-Filtered'],
            deepcell_output_dir=output_dir,
            delimiter="_feature_0")

        # generate _feature_1 tif files that would normally be handled by create_deepcell_output
        notebooks_test_utils.generate_sample_feature_tifs(
            fovs=['fov0', 'fov1'],
            deepcell_output_dir=output_dir,
            delimiter="_feature_1")

        # run the segmentation labels saving and summed channel overlay processes
        notebooks_test_utils.save_seg_labels(tb, xr_channel_names=['whole_cell', 'nuclear'])

        # create the expression matrix
        notebooks_test_utils.create_exp_mat(tb, is_mibitiff=True)

        # create the expression matrix with nuclear counts
        notebooks_test_utils.create_exp_mat(tb, is_mibitiff=True, nuclear_counts=True)


# test folder segmentation
@testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000)
def test_segment_image_data_folder(tb):
    with tdir() as tiff_dir, tdir() as input_dir, tdir() as output_dir, \
         tdir() as single_cell_dir, tdir() as viz_dir:
        # create input files
        notebooks_test_utils.segment_notebook_setup(tb,
                                                    deepcell_tiff_dir=tiff_dir,
                                                    deepcell_input_dir=input_dir,
                                                    deepcell_output_dir=output_dir,
                                                    single_cell_dir=single_cell_dir,
                                                    viz_dir=viz_dir)

        # hard coded fov setting
        notebooks_test_utils.fov_channel_input_set(
            tb,
            fovs=['fov0', 'fov1'],
            nucs_list=['chan0'],
            mems_list=['chan1', 'chan2'])

        # generate _feature_0 tif files that would normally be handled by create_deepcell_output
        notebooks_test_utils.generate_sample_feature_tifs(
            fovs=['fov0', 'fov1'],
            deepcell_output_dir=output_dir,
            delimiter="_feature_0")

        # generate _feature_1 tif files that would normally be handled by create_deepcell_output
        notebooks_test_utils.generate_sample_feature_tifs(
            fovs=['fov0', 'fov1'],
            deepcell_output_dir=output_dir,
            delimiter="_feature_1")

        # run the segmentation labels saving and summed channel overlay processes
        notebooks_test_utils.save_seg_labels(tb, xr_channel_names=['whole_cell', 'nuclear'])

        # create the expression matrix
        notebooks_test_utils.create_exp_mat(tb)

        # create the expression matrix with nuclear counts
        notebooks_test_utils.create_exp_mat(tb, nuclear_counts=True)


# # test mibitiff clustering
# @testbook(FLOWSOM_CLUSTER_PATH, timeout=6000)
# def test_flowsom_cluster_mibitiff(tb):
#     with tdir() as base_dir:
#         # create input files
#         notebooks_test_utils.flowsom_setup(tb, flowsom_dir=base_dir, is_mibitiff=True)

#         # load img data in
#         notebooks_test_utils.flowsom_set_fovs_channels(tb,
#                                                        channels=['chan0', 'chan1'],
#                                                        fovs=['fov0_otherinfo-MassCorrected-Filtered.tiff',
#                                                              'fov1-MassCorrected-Filtered.tiff'])

#         # run the FlowSOM preprocessing and clustering
#         notebooks_test_utils.flowsom_run(tb,
#                                          fovs=['fov0_otherinfo-MassCorrected-Filtered.tiff',
#                                                'fov1-MassCorrected-Filtered.tiff'],
#                                          channels=['chan0', 'chan1'],
#                                          is_mibitiff=True)


# test folder clustering
@testbook(FLOWSOM_CLUSTER_PATH, timeout=6000)
def test_flowsom_cluster_folder(tb):
    with tdir() as base_dir:
        # create input files
        notebooks_test_utils.flowsom_setup(tb, flowsom_dir=base_dir)

        # run the FlowSOM preprocessing and clustering
        notebooks_test_utils.flowsom_pixel_run(
            tb, fovs=['fov0', 'fov1'], channels=['chan0', 'chan1']
        )

        # TODO: see what Brian discovers about R testing, then add cell clustering tests
