import os
import numpy as np
import skimage.io as io

from ark.utils import test_utils


def segment_notebook_setup(tb, deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir,
                           single_cell_dir, viz_dir, is_mibitiff=False,
                           mibitiff_suffix="-MassCorrected-Filtered",
                           num_fovs=3, num_chans=3, dtype=np.uint16):
    """Creates the directories and data needed and sets the MIBITiff variable accordingly

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        deepcell_tiff_dir (str):
            The path to the tiff directory
        deepcell_input_dir (str):
            The path to the input directory
        deepcell_output_dir (str):
            The path to the output directory
        single_cell_dir (str):
            The path to the single cell directory
        viz_dir (str):
            The path to the directory to store visualizations
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

    if is_mibitiff:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    use_delimiter=True)
        fovs = [f + mibitiff_suffix for f in fovs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            deepcell_tiff_dir, fovs, chans, img_shape=(1024, 1024), mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            deepcell_tiff_dir, fovs, chans, img_shape=(1024, 1024), delimiter='_', fills=False,
            sub_dir="TIFs", dtype=dtype)

    # define custom paths, leaving base_dir and input_dir for simplicity
    define_paths = """
        tiff_dir = "%s"
        deepcell_input_dir = "%s"
        deepcell_output_dir = "%s"
        single_cell_dir = "%s"
        viz_dir = "%s"
    """ % (deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir, single_cell_dir, viz_dir)
    tb.inject(define_paths, after='file_path')

    # will set MIBItiff and MIBItiff_suffix
    tb.execute_cell('mibitiff_set')
    if is_mibitiff:
        # default setting is MIBItiff = False, change to True if user has mibitiff inputs
        tb.inject("MIBItiff = True", after='mibitiff_set')


def fov_channel_input_set(tb, fovs=None, nucs_list=None, mems_list=None, is_mibitiff=False):
    """Sets the fovs and channels and creates the input directory for DeepCell

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        fovs (list):
            If set, assigns the fovs variable to this list.
            If None, executes the default fov loading scheme in the 'load_fovs' cell.
        nucs_list (list):
            Assigns the nucs variable to this list
        mems_list (list):
            Assigns the mems variable to this list
        is_mibitiff (bool):
            Whether we're working with mibitiff files or not
    """

    # load the fovs in the notebook
    if fovs is not None:
        # handles the case when the user assigns fovs to an explicit list
        tb.inject("fovs = %s" % str(fovs))
    else:
        # handles the case when the user allows list_files or list_folders to do the fov loading
        tb.execute_cell('load_fovs')

    # set the nucs_list and the mems_list accordingly
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

    # generate the deepcell input files, explicitly set is_mibitiff if True
    mibitiff_deepcell = """
        data_utils.generate_deepcell_input(
            deepcell_input_dir, tiff_dir, nucs, mems, fovs,
            is_mibitiff=%s, img_sub_folder="TIFs", batch_size=5
        )
    """ % str(is_mibitiff)
    tb.inject(mibitiff_deepcell, after='gen_input')


def generate_sample_feature_tifs(fovs, deepcell_output_dir):
    """Generate a sample _feature_0 tif file for each fov

    Done to bypass the bottleneck of create_deepcell_output, for testing purposes we don't care
    about correct segmentation labels.

    Args:
        fovs (list):
            The list of fovs to generate sample _feature_0 tif files for
        deepcell_output_dir (str):
            The path to the output directory
    """

    # generate a random image for each fov, set as both whole cell and nuclear
    for fov in fovs:
        rand_img = np.random.randint(0, 16, size=(1024, 1024))
        io.imsave(os.path.join(deepcell_output_dir, fov + "_feature_0.tif"), rand_img)
        io.imsave(os.path.join(deepcell_output_dir, fov + "_feature_1.tif"), rand_img)


def overlay_mask(tb, channels=None):
    """Overlays the segmentation labels overlay (with channel data if specified)

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        channels (list):
            List of channels to overlay
    """

    # generate segmentation mask overlay
    tb.execute_cell('overlay_mask')

    # save the overlay, channels needs to be explicitly set if not None
    if channels is not None:
        save_seg = """
            segmentation_utils.save_segmentation_labels(
                segmentation_dir=deepcell_output_dir,
                data_dir=deepcell_input_dir,
                output_dir=viz_dir,
                fovs=io_utils.remove_file_extensions(fovs),
                channels=%s
            )
        """ % str(channels)
        tb.inject(save_seg, after='save_mask')
    else:
        tb.execute_cell('save_mask')


def create_exp_mat(tb, is_mibitiff=False, batch_size=5, nuclear_counts=False):
    """Creates the expression matrices from the generated segmentation labels

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        is_mibitiff (bool):
            Whether we're working with mibitiff images
        batch_size (int):
            The number of fovs we want to process at a time
        nuclear_counts (bool):
            Whether to include nuclear properties in the cell table
    """

    # explicitly set is_mibitiff and nuclear_counts if default overridden
    exp_mat_gen = """
        cell_table_size_normalized, cell_table_arcsinh_transformed = \
            marker_quantification.generate_cell_table(segmentation_dir=deepcell_output_dir,
                                                      tiff_dir=tiff_dir,
                                                      img_sub_folder="TIFs",
                                                      is_mibitiff=%s,
                                                      fovs=fovs,
                                                      batch_size=%s,
                                                      nuclear_counts=%s)
    """ % (is_mibitiff, str(batch_size), nuclear_counts)
    tb.inject(exp_mat_gen, after='create_exp_mat')

    # save expression matrices
    tb.execute_cell('save_exp_mat')
