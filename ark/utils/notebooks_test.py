import subprocess
import tempfile
import os

from testbook import testbook

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
def test_segment_image_data(mocker):
    _exec_notebook('Segment_Image_Data.ipynb')


def test_example_spatial_analysis():
    _exec_notebook('example_spatial_analysis_script.ipynb')


def test_example_neighborhood_analysis():
    _exec_notebook('example_neighborhood_analysis_script.ipynb')


# testing specific inputs for Segment_Image_Data
# test mibitiff, 6000 seconds = default timeout on Travis
@testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000)
def test_segment_image_data_mibitiff(tb):
    # create input files, set separate names for mibitiffs to avoid confusion
    notebooks_test_utils.segment_notebook_setup(tb,
                                                deepcell_tiff_dir="test_mibitiff",
                                                deepcell_input_dir="test_mibitiff_input",
                                                deepcell_output_dir="test_mibitiff_output",
                                                single_cell_dir="test_mibitiff_single_cell",
                                                is_mibitiff=True)

    # default fov setting, standard nucs/mems setting
    notebooks_test_utils.fov_channel_input_set(tb,
                                               nucs_list=['chan0'],
                                               mems_list=['chan1', 'chan2'])

    # default fov setting, nucs set to None
    notebooks_test_utils.fov_channel_input_set(tb,
                                               nucs_list=None,
                                               mems_list=['chan1', 'chan2'])

    # default fov setting, mems set to None
    notebooks_test_utils.fov_channel_input_set(tb,
                                               nucs_list=['chan0', 'chan1'],
                                               mems_list=None)

    # hard coded fov setting, standard nucs/mems setting, this is what we'll be testing on
    # TODO: this will fail if fovs_to_load is set without file extensions
    notebooks_test_utils.fov_channel_input_set(
        tb,
        fovs_to_load=['fov0_otherinfo-MassCorrected-Filtered.tiff',
                      'fov1-MassCorrected-Filtered.tiff'],
        nucs_list=['chan0'],
        mems_list=['chan1', 'chan2'])

    # generate the deepcell output files from the server
    tb.execute_cell('create_output')

    # run the segmentation labels saving and summed channel overlay processes
    notebooks_test_utils.save_seg_labels(tb, xr_channel_names=['whole_cell'])

    # create the expression matrix
    notebooks_test_utils.create_exp_mat(tb, is_mibitiff=True)

    # clean up the directories
    notebooks_test_utils.remove_dirs(tb)


# test folder loading
@testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000)
def test_segment_image_data_folder(tb):
    # create input files
    notebooks_test_utils.segment_notebook_setup(tb)

    # default fov setting, standard nucs/mems setting
    notebooks_test_utils.fov_channel_input_set(tb,
                                               nucs_list=['chan0'],
                                               mems_list=['chan1', 'chan2'])

    # default fov setting, nucs set to None
    notebooks_test_utils.fov_channel_input_set(tb,
                                               nucs_list=None,
                                               mems_list=['chan1', 'chan2'])

    # default fov setting, mems set to None
    notebooks_test_utils.fov_channel_input_set(tb,
                                               nucs_list=['chan0', 'chan1'],
                                               mems_list=None)

    # hard coded fov setting, standard nucs/mems setting, this is what we'll be testing on
    notebooks_test_utils.fov_channel_input_set(
        tb,
        fovs_to_load=['fov0', 'fov1'],
        nucs_list=['chan0'],
        mems_list=['chan1', 'chan2'])

    # generate the deepcell output files from the server
    tb.execute_cell('create_output')

    # run the segmentation labels saving and summed channel overlay processes
    notebooks_test_utils.save_seg_labels(tb, xr_channel_names=['whole_cell'])

    # create the expression matrix
    notebooks_test_utils.create_exp_mat(tb)

    # clean up the directories
    notebooks_test_utils.remove_dirs(tb)
