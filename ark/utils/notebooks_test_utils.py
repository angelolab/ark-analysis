import os
import numpy as np

from ark.utils import test_utils


def segment_notebook_setup(tb, deepcell_tiff_dir="test_tiff", deepcell_input_dir="test_input",
                           deepcell_output_dir="test_output",
                           single_cell_dir="test_single_cell",
                           is_mibitiff=False, mibitiff_suffix="-MassCorrected-Filtered",
                           num_fovs=3, num_chans=3, dtype=np.uint16):
    """Creates the diretories and data needed and sets the MIBITiff variable accordingly

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        deepcell_tiff_dir (str):
            The name of the directory holding the images
        deepcell_input_dir (str):
            The name of the directory to hold the processed images to send to DeepCell
        deepcell_output_dir (str):
            The name of the directory to hold the DeepCell output
        single_cell_dir (str):
            The name of the directory to hold the processed expression matrix files
        is_mibitiff (bool):
            Whether we're working with mibitiff files or not
        mibitiff_suffix (str):
            If is_mibitiff = True, the suffix to append to each fov.
            Ignored if is_mibitiff = False.
        num_fovs (int):
            The number of test fovs to generate
        num_chans (int):
            The number of test channels to generate
        dtype (numpy.dtype):
            The datatype of each test image generated
    """

    # import modules and define file paths
    tb.execute_cell('import')

    # define custom mibitiff paths
    define_mibitiff_paths = """
        base_dir = "../data/example_dataset"
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = os.path.join(input_dir, "%s/")
        deepcell_input_dir = os.path.join(input_dir, "%s/")
        deepcell_output_dir = os.path.join(base_dir, "%s/")
        single_cell_dir = os.path.join(base_dir, "%s/")
    """ % (deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir, single_cell_dir)
    tb.inject(define_mibitiff_paths, after='file_path')

    # create the tif files, don't do this in notebook it's too tedious to format this
    # also, because this is an input that would be created beforehand
    tiff_path = os.path.join('data', 'example_dataset', 'input_data', deepcell_tiff_dir)
    os.mkdir(tiff_path)

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
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_path, fovs, chans, img_shape=(1024, 1024), delimiter='_', fills=False,
            sub_dir="TIFs", dtype=dtype)

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


def fov_channel_input_set(tb, fovs_to_load=None, nucs_list=None, mems_list=None):
    """Sets the fovs and channels and creates the input directory for DeepCell

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        fovs_to_load (list):
            If set, assigns the fovs variable to this list.
            If None, executes the default fov loading scheme in the 'load_fovs' cell.
        nucs_list (list):
            Assigns the nucs variable to this list
        mems_list (list):
            Assigns the mems variable to this list
    """

    # load the fovs in the notebook
    if fovs_to_load is not None:
        tb.inject("fovs = %s" % str(fovs_to_load))
    else:
        tb.execute_cell('load_fovs')

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
        nucs = %s
        mems = %s
    """ % (nucs_list_str, mems_list_str)
    tb.inject(nuc_mem_set, after='nuc_mem_set')

    # set the channels accordingly
    tb.execute_cell('set_channels')

    # load data accordingly
    tb.execute_cell('load_data_xr')

    # generate the deepcell input files
    # NOTE: any specific testing of generate_deepcell_input should be done in data_utils_test
    tb.execute_cell('gen_input')


def save_seg_labels(tb, delimiter='_feature_0', xr_dim_name='compartments',
                    xr_channel_names=None, force_ints=True):
    """Processes and saves the generated segmentation labels and runs the summed channel overlay

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        delimiter (str):
            The suffix of the files to read from deepcell_output_dir
        xr_dim_name (str):
            The dimension containing the channel names to read in
        xr_channel_names (list):
            The list of the channels we wish to read in
        force_ints (bool):
            Whether to convert the segmentation labels to integer type
    """

    delimiter_str = delimiter if delimiter is not None else "None"
    xr_channel_names_str = str(xr_channel_names) if xr_channel_names is not None else "None"

    # load the segmentation label with the proper command
    load_seg_cmd = """
        segmentation_labels = load_utils.load_imgs_from_dir(
            data_dir=deepcell_output_dir,
            delimiter="%s",
            xr_dim_name="%s",
            xr_channel_names=%s,
            force_ints=%s
        )
    """ % (delimiter_str,
           xr_dim_name,
           xr_channel_names,
           str(force_ints))
    tb.inject(load_seg_cmd, after='load_seg_labels')

    tb.execute_cell('save_seg_labels')

    # now overlay data_xr
    tb.execute_cell('load_summed')
    tb.execute_cell('overlay_mask')


def create_exp_mat(tb, is_mibitiff=False, batch_size=5):
    """Creates the expression matrices from the generated segmentation labels

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        is_mibitiff (bool):
            Whether we're working with mibitiff images
        batch_size (int):
            The number of fovs we want to process at a time
    """

    exp_mat_gen = """
        cell_table_size_normalized, cell_table_arcsinh_transformed = \
            marker_quantification.generate_cell_table(segmentation_labels=segmentation_labels,
                                                      tiff_dir=tiff_dir,
                                                      img_sub_folder="TIFs",
                                                      is_mibitiff=%s,
                                                      fovs=fovs,
                                                      batch_size=%s)
    """ % (is_mibitiff, str(batch_size))
    tb.inject(exp_mat_gen)

    # save expression matrix
    tb.execute_cell('save_exp_mat')


def remove_dirs(tb):
    """Removes all of the test folders created

    Args:
        tb (testbook.testbook):
            The testbook runner instance
    """
    remove_dirs = """
        from shutil import rmtree
        rmtree(tiff_dir)
        rmtree(deepcell_input_dir)
        rmtree(deepcell_output_dir)
        rmtree(single_cell_dir)
    """
    tb.inject(remove_dirs)
