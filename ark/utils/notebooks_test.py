import subprocess
import tempfile
import os
from shutil import rmtree

from testbook import testbook

from ark.utils import test_utils
from ark.utils import misc_utils

import numpy as np
import skimage.io as io


SEGMENT_IMAGE_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  '..', '..', 'templates', 'Segment_Image_Data.ipynb')


# def _exec_notebook(nb_filename):
#     path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                         '..', '..', 'templates', nb_filename)
#     with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
#         args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
#                 "--ExecutePreprocessor.timeout=1000",
#                 "--output", fout.name, path]
#         subprocess.check_call(args)


# test overall runs
# def test_segment_image_data(mocker):
#     _exec_notebook('Segment_Image_Data.ipynb')


# def test_example_spatial_analysis():
#     _exec_notebook('example_spatial_analysis_script.ipynb')


# def test_example_neighborhood_analysis():
#     _exec_notebook('example_neighborhood_analysis_script.ipynb')


def segment_notebook_setup(tb, deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir,
                           single_cell_dir, is_mibitiff, mibitiff_suffix,
                           num_fovs, num_chans, dtype):
    # import modules and define file paths
    tb.execute_cell('import')

    # create the path to the directory containing the input data
    tiff_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', '..', 'data', 'example_dataset',
                             'input_data', deepcell_tiff_dir)

    if os.path.exists(tiff_path):
        rmtree(tiff_path)

    os.mkdir(tiff_path)

    # generate sample data in deepcell_tiff_dir
    if is_mibitiff:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    use_delimiter=True)
        fovs = [f + mibitiff_suffix for f in fovs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_path, fovs, chans, img_shape=(1024, 1024), mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                          num_chans=num_chans,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_path, fovs, chans, img_shape=(1024, 1024), delimiter='_', fills=False,
            sub_dir="TIFs", dtype=dtype)

    # define custom mibitiff paths
    define_mibitiff_paths = """
        base_dir = "../data/example_dataset"
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = "%s/"
        deepcell_input_dir = os.path.join(input_dir, "%s/")
        deepcell_output_dir = os.path.join(base_dir, "%s/")
        single_cell_dir = os.path.join(base_dir, "%s/")
    """ % (tiff_path, deepcell_input_dir, deepcell_output_dir, single_cell_dir)
    tb.inject(define_mibitiff_paths, after='file_path')

    # create the directories as listed by define_mibitiff_paths
    tb.execute_cell('create_dirs')

    # validate the paths, and in Jupyter, this should always pass
    # NOTE: any specific testing of validate_paths should be done in io_utils_test.py
    tb.execute_cell('validate_path')

    # will set MIBItiff and MIBItiff_suffix
    # if is_mibitiff is True, then we need to correct MIBITiff to True
    tb.execute_cell('mibitiff_set')
    if is_mibitiff:
        tb.inject("MIBItiff = True", after='mibitiff_set')

    # now load the fovs in the notebook
    # NOTE: any specific testing of list_files and list_folders should be done in io_utils_test
    tb.execute_cell('load_fovs')


def create_deepcell_input_output(tb, nucs_list, mems_list):
    # we need to set the nucs_list and the mems_list accordingly
    if nucs_list is None:
        nucs_list_str = "None"
    else:
        nucs_list_str = "%s" % str(nucs_list)

    if mems_list is None:
        mems_list_str = "None"
    else:
        mems_list_str = "%s" % str(mems_list)

    nuc_mem_set = """
        nucs = %s\n
        mems = %s
    """ % (nucs_list_str, mems_list_str)
    tb.inject(nuc_mem_set, after='nuc_mem_set')

    # set the channels accordingly
    tb.execute_cell('set_channels')

    # load data accordingly
    try:
        tb.execute_cell('load_data_xr')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)

    # generate the deepcell input files
    # NOTE: any specific testing of generate_deepcell_input should be done in data_utils_test
    try:
        tb.execute_cell('gen_input')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)

    try:
        tb.execute_cell('create_output')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)

    # NOTE: the following is alternative code in case try-except blocks don't handle these correct
    # create_output_text = tb.cell_output_text('create_output')
    # remove on final commit
    # if 'Error' in create_output_text or 'warnings' in create_output_text:
    #     print(tb.cell_output_text('create_output'))
    #     remove_dirs(tb)
    #     raise ValueError('Could not create deepcell output')


def save_seg_labels(tb, delimiter=None, xr_dim_name='compartments',
                    xr_channel_names=None, force_ints=False):
    delimiter_str = delimiter if delimiter is not None else "None"
    xr_channel_names_str = str(xr_channel_names) if xr_channel_names is not None else "None"

    # load the segmentation label with the proper command
    # NOTE: any specific testing of load_imgs_from_dir should be done in load_utils_test.py
    load_seg_cmd = """
        segmentation_labels = load_utils.load_imgs_from_dir(data_dir=deepcell_output_dir,
            delimiter="%s",
            xr_dim_name="%s",
            xr_channel_names=%s,
            force_ints=%s
        )
    """ % (delimiter_str,
           xr_dim_name,
           xr_channel_names,
           str(force_ints))
    try:
        tb.inject(load_seg_cmd, after='load_seg_labels')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)

    try:
        tb.execute_cell('save_seg_labels')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)


def data_xr_overlay(tb, files, xr_dim_name='compartments', xr_channel_names=None):
    files_str = str(files) if files is not None else "None"

    load_data_xr_sum = """
        data_xr_summed = load_utils.load_imgs_from_dir(data_dir=deepcell_input_dir,
            files=%s,
            xr_dim_name="%s",
            xr_channel_names=%s
        )
    """ % (files_str, xr_dim_name, str(xr_channel_names))

    try:
        tb.inject(load_data_xr_sum, after='load_summed')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)

    # NOTE: this will fail if load_data_xr_sum fails
    try:
        tb.execute_cell('overlay_mask')
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)


def create_exp_mat(tb, is_mibitiff=False, batch_size=5):
    # NOTE: segmentation labels will already have been created
    exp_mat_gen = """
        cell_table_size_normalized, cell_table_arcsinh_transformed = \
            marker_quantification.generate_cell_table(segmentation_labels=segmentation_labels,
                                                      tiff_dir=deepcell_output_dir,
                                                      img_sub_folder="TIFs",
                                                      is_mibitiff=%s,
                                                      fovs=fovs,
                                                      batch_size=%s)
    """ % (is_mibitiff, str(batch_size))

    try:
        tb.inject(exp_mat_gen)
    except ValueError as ve:
        print(ve)
        remove_dirs(tb)

    tb.execute_cell('save_exp_mat')


def remove_dirs(tb):
    remove_dirs = """
        from shutil import rmtree
        rmtree(tiff_dir)
        rmtree(deepcell_input_dir)
        rmtree(deepcell_output_dir)
        rmtree(single_cell_dir)
    """
    tb.inject(remove_dirs)


# test mibitiff
@testbook(SEGMENT_IMAGE_DATA, timeout=6000)
def test_mibitiff_segmentation(tb):
    segment_notebook_setup(tb, deepcell_tiff_dir="sample_tiff", deepcell_input_dir="sample_input",
                           deepcell_output_dir="sample_output",
                           single_cell_dir="sample_single_cell", is_mibitiff=True,
                           mibitiff_suffix="-MassCorrected-Filtered",
                           num_fovs=3, num_chans=3, dtype=np.uint16)
    create_deepcell_input_output(tb, nucs_list=['chan0'], mems_list=['chan1', 'chan2'])
    save_seg_labels(tb, delimiter='_feature_0', xr_channel_names=['whole_cell'], force_ints=True)
    data_xr_overlay(tb, files=['fov0_otherinfo-MassCorrected-Filtered.tif',
                               'fov1-MassCorrected-Filtered.tif'],
                    xr_channel_names=['nuclear', 'membrane'])
    create_exp_map(tb, is_mibitiff=True)
    remove_dirs()
