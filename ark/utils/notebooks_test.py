import subprocess
import os
import tempfile

from testbook import testbook
from tempfile import TemporaryDirectory as tdir

from ark.utils import notebooks_test_utils


SEGMENT_IMAGE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..', '..', 'templates', 'Segment_Image_Data.ipynb')


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


# testing specific inputs for Segment_Image_Data
# test mibitiff, 6000 seconds = default timeout on Travis
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
            deepcell_output_dir=output_dir)

        # generate the deepcell output files from the server
        # tb.execute_cell('create_output')

        # run the segmentation labels saving and summed channel overlay processes
        notebooks_test_utils.save_seg_labels(tb, xr_channel_names=['whole_cell'])

        # create the expression matrix
        notebooks_test_utils.create_exp_mat(tb, is_mibitiff=True)


# test folder loading
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
            deepcell_output_dir=output_dir)

        # generate the deepcell output files from the server
        # tb.execute_cell('create_output')

        # run the segmentation labels saving and summed channel overlay processes
        notebooks_test_utils.save_seg_labels(tb, xr_channel_names=['whole_cell'])

        # create the expression matrix
        notebooks_test_utils.create_exp_mat(tb)
