import os
import numpy as np
import skimage.io as io

from ark.utils import test_utils


def create_tiff_files(num_fovs, num_chans, tiff_dir, sub_dir="TIFs", is_mibitiff=False,
                      mibitiff_suffix="-MassCorrected-Filtered", dtype=np.uint16):
    """Creates the desired input tiff data for testing a notebook

    Args:
        num_fovs (int):
            The number of test fovs to generate
        num_chans (int):
            The number of test channels to generate
        tiff_dir (str):
            The path to the tiff directory
        is_mibitiff (bool):
            Whether we're working with mibitiff files or not
        mibitiff_suffix (str):
            If is_mibitiff = True, the suffix to append to each fov.
            Ignored if is_mibitiff = False.
        dtype (numpy.dtype):
            The datatype of each test image generated

    Returns:
        tuple:

        - A list of fovs created
        - A list of channels created
    """

    if is_mibitiff:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    use_delimiter=True)
        fovs = [f + mibitiff_suffix for f in fovs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_dir, fovs, chans, img_shape=(1024, 1024), mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_dir, fovs, chans, img_shape=(1024, 1024), delimiter='_', fills=False,
            sub_dir=sub_dir, dtype=dtype)

    return fovs, chans


def segment_notebook_setup(tb, deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir,
                           single_cell_dir, viz_dir, sub_dir="TIFs", is_mibitiff=False,
                           mibitiff_suffix="-MassCorrected-Filtered",
                           num_fovs=3, num_chans=3, dtype=np.uint16):
    """Creates the directories and data needed for image segemntation
    and sets the MIBITiff variable accordingly

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
        sub_dir (str):
            The name of the subdirectory to use for non-mibitiff image folders
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

    # create the input tiff files
    create_tiff_files(num_fovs, num_chans, deepcell_tiff_dir, sub_dir, is_mibitiff,
                      mibitiff_suffix, dtype)

    # define custom paths, leave base_dir and input_dir for simplicity
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


def qc_notebook_setup(tb, tiff_dir, sub_dir=None, is_mibitiff=False,
                      mibitiff_suffix="-MassCorrected-Filtered",
                      num_fovs=3, num_chans=3, gaussian_blur=True, dtype=np.uint16):
    """Creates the directories and data needed for qc metric computation
    and sets the MIBITiff, fovs, chans, and gaussian_blur/blur_factor vairables

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        deepcell_tiff_dir (str):
            The path to the tiff directory
        sub_dir (str):
            The name of the subdirectory to use for non-mibitiff image folders
        is_mibitiff (bool):
            Whether we're working with mibitiff files or not
        mibitiff_suffix (str):
            If is_mibitiff = True, the suffix to append to each fov.
            Ignored if is_mibitiff = False.
        num_fovs (int):
            The number of test fovs to generate
        num_chans (int):
            The number of test channels to generate
        gaussian_blur (bool):
            Whether to set Gaussian blurring or not
        dtype (numpy.dtype):
            The datatype of each test image generated
    """

    # import modules and define file paths
    tb.execute_cell('import')

    # create the input tiff files
    create_tiff_files(num_fovs, num_chans, tiff_dir, sub_dir, is_mibitiff, mibitiff_suffix, dtype)

    # define custom paths, leave base_dir for simplicity
    define_paths = """
        base_dir = "%s"
        tiff_dir = "%s"
    """ % (tiff_dir, tiff_dir)
    tb.inject(define_paths, after='file_path')

    # set is_mibitiff to True if corresponding arg is True
    if is_mibitiff:
        tb.inject("MIBItiff = True", after='mibitiff_set')
    else:
        tb.execute_cell('mibitiff_set')

    # specify a list of fovs
    fovs_set = """
        fovs = ["fov0", "fov1", "fov2"]
    """
    tb.inject(fovs_set, after='load_fovs')

    # specify a list of chans
    chans_set = """
        chans = ["chan0", "chan1", "chan2"]
    """
    tb.inject(chans_set, after='set_chans')

    # set the blur factor
    tb.execute_cell('set_gaussian_blur')

    # if Gaussian blurring is set to True we need to set it in the notebook too
    if gaussian_blur:
        tb.inject("gaussian_blur = True", after='set_gaussian_blur')


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


def run_qc_comp(tb):
    """Runs the QC computation process with the hard-coded inputs from qc_notebook_setup

    Args:
        tb (testbook.testbook):
            The testbook runner instance
    """

    # run compute_qc_metrics
    tb.execute_cell('compute_qc_data')

    # extract from a dictionary the final results
    tb.execute_cell('assign_qc_data')

    # extract just the numeric value from fovs
    tb.execute_cell('rename_fovs')

    # sort the fovs by fov number
    tb.execute_cell('sort_by_fov')

    # save the QC data to CSV
    tb.execute_cell('save_qc_data')

    # melt the QC data for visualization
    tb.execute_cell('melt_qc')

    # visualize the non-zero mean intensity
    tb.execute_cell('viz_nonzero_mean')

    # visualize the total intensity
    tb.execute_cell('viz_total_intensity')

    # visualize the 99.9% intensity value
    tb.execute_cell('viz_99_9')


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
