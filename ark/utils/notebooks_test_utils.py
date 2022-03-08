from datetime import datetime as dt
import os
import numpy as np
import skimage.io as io

from ark.utils import test_utils


def create_tiff_files(num_fovs, num_chans, tiff_dir, sub_dir="TIFs", is_mibitiff=False,
                      mibitiff_suffix="-MassCorrected-Filtered", img_shape=(50, 50),
                      dtype=np.uint16):
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
        img_shape (tuple):
            The shape of the image to generate
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
            tiff_dir, fovs, chans, img_shape=(50, 50), mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_dir, fovs, chans, img_shape=(50, 50), delimiter='_', fills=False,
            sub_dir=sub_dir, dtype=dtype
        )

    return fovs, chans


def segment_notebook_setup(tb, deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir,
                           single_cell_dir, viz_dir, sub_dir="TIFs", is_mibitiff=False,
                           mibitiff_suffix="-MassCorrected-Filtered",
                           img_shape=(50, 50), num_fovs=3, num_chans=3, dtype=np.uint16):
    """Creates the directories, data, and MIBItiff settings for testing segmentation process

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
        img_shape (tuple):
            The shape of the image to generate
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
                      mibitiff_suffix, img_shape, dtype)

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


def flowsom_pixel_setup(tb, flowsom_dir, img_shape=(50, 50), num_fovs=3, num_chans=3,
                        is_mibitiff=False, mibitiff_suffix="-MassCorrected-Filtered",
                        dtype=np.uint16):
    """Creates the directories, data, and MIBItiff settings for testing pixel clustering

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        flowsom_dir (str):
            The path to the FlowSOM data directory
        img_shape (tuple):
            The shape of the image to generate
        num_fovs (int):
            The number of test fovs to generate
        num_chans (int):
            The number of test channels to generate
        is_mibitiff (bool):
            Whether we're working with mibitiff files or not
        mibitiff_suffix (str):
            If is_mibitiff = True, the suffix to append to each fov.
            Ignored if is_mibitiff = False.
        dtype (numpy.dtype):
            The datatype of each test image generated
    """

    tb.execute_cell('import')

    # create data which will be loaded into img_xr
    tiff_dir = os.path.join(flowsom_dir, "input_data")
    os.mkdir(tiff_dir)

    if is_mibitiff:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    use_delimiter=True)
        fovs = [f + mibitiff_suffix for f in fovs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_dir, fovs, chans, img_shape=img_shape, mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_dir, fovs, chans, img_shape=img_shape, delimiter='_', fills=False, dtype=dtype
        )

    # generate sample segmentation labels so we can load them in
    seg_dir = os.path.join(flowsom_dir, "deepcell_output")
    os.mkdir(seg_dir)
    generate_sample_feature_tifs(fovs, seg_dir, img_shape)

    # define custom data paths
    define_data_paths = """
        base_dir = "%s"
        tiff_dir = "%s"
        img_sub_folder = None
        segmentation_dir = "%s"
        seg_suffix = '_feature_0.tif'
        MIBItiff = %s
        mibitiff_suffix = '%s'
    """ % (flowsom_dir, tiff_dir, seg_dir, is_mibitiff, mibitiff_suffix)
    tb.inject(define_data_paths, after='file_path')


def flowsom_cell_setup(tb, flowsom_dir, ):
    """Creates the directories, data, and MIBItiff settings for testing cell clustering

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        flowsom_dir (str):
            The path to the FlowSOM data directory
        img_shape (tuple):
            The shape of the image to generate
        num_fovs (int):
            The number of test fovs to generate
        num_chans (int):
            The number of test channels to generate
        is_mibitiff (bool):
            Whether we're working with mibitiff files or not
        mibitiff_suffix (str):
            If is_mibitiff = True, the suffix to append to each fov.
            Ignored if is_mibitiff = False.
        dtype (numpy.dtype):
            The datatype of each test image generated
    """
    pass


def flowsom_pixel_run(tb, fovs, channels, cluster_prefix='test', is_mibitiff=False):
    """Run the FlowSOM pixel-level clustering

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        fovs (list):
            The list of fovs
        channels (list):
            The list of channels
        cluster_prefix (str):
            The name of the prefix to use for each directory/file created by pixel/cell clustering
        is_mibitiff (bool):
            Whether we're working with mibitiff images
    """

    if fovs is not None:
        # handles the case when the user assigns fovs to an explicit list
        tb.inject(
            """
                fovs = %s
            """ % str(fovs),
            after='load_fovs'
        )
    else:
        # handles the case when the user allows list_files or list_folders to do the fov loading
        tb.execute_cell('load_fovs')

    # set the names of the preprocessed and segmented directories
    set_pre_seg_dirs = """
        preprocessed_dir = '%s_preprocessed_dir'
        subsetted_dir = '%s_subsetted_dir'
    """ % (cluster_prefix, cluster_prefix)
    tb.inject(set_pre_seg_dirs, after='pre_sub_dir_set')

    # sets the channels to include
    tb.inject(
        """
            channels = %s
            blur_factor = 2
            subset_proportion = 0.1
        """ % str(channels),
        after='channel_set'
    )

    # test the preprocessing works, we won't save nor run the actual FlowSOM clustering
    if is_mibitiff:
        mibitiff_preprocess = """
            som_utils.create_pixel_matrix(
                fovs, channels, base_dir, tiff_dir, segmentation_dir,
                pre_dir=preprocessed_dir, sub_dir=subsetted_dir, is_mibitiff=True,
                blur_factor=blur_factor, subset_proportion=subset_proportion, seed=seed
            )
        """

        tb.inject(mibitiff_preprocess, after='gen_pixel_mat')
    else:
        tb.execute_cell('gen_pixel_mat')

    # define a custom prefix for the SOM and cell cluster assignments
    prefix_set = """
        cluster_prefix = '%s'

        pixel_clustered_dir = '%s_pixel_mat_clustered'
        pixel_consensus_dir = '%s_pixel_mat_consensus'
        pixel_weights_name = '%s_pixel_weights.feather'
    """ % (cluster_prefix, cluster_prefix, cluster_prefix, cluster_prefix)
    tb.inject(prefix_set, after='pixel_som_path_set')

    # create a dummy weights feather
    dummy_weights = """
        import feather
        weights = pd.DataFrame(np.random.rand(100, len(channels)), columns=channels)

        feather.write_dataframe(weights, os.path.join(base_dir, pixel_weights_name))
    """
    tb.inject(dummy_weights, after='train_pixel_som')

    # create dummy clustered feathers for each fov
    cluster_setup = """
        if not os.path.exists(os.path.join(base_dir, pixel_clustered_dir)):
            os.mkdir(os.path.join(base_dir, pixel_clustered_dir))
    """
    tb.inject(cluster_setup, after='cluster_pixel_mat')

    for fov in fovs:
        dummy_cluster_cmd = """
            sample_df = pd.DataFrame(np.random.rand(100, 6),
                                     columns=%s +
                                     ['fov', 'row_index', 'col_index', 'segmentation_label'])
            sample_df['fov'] = '%s'
            sample_df['clusters'] = np.random.randint(0, 100, size=100)

            feather.write_dataframe(sample_df, os.path.join(base_dir,
                                                            pixel_clustered_dir,
                                                            '%s' + '.feather'))
        """ % (str(channels), fov, fov)

        tb.inject(dummy_cluster_cmd, after='cluster_pixel_mat')

    # create dummy clustered feathers for each fov
    consensus_setup = """
        if not os.path.exists(os.path.join(base_dir, pixel_consensus_dir)):
            os.mkdir(os.path.join(base_dir, pixel_consensus_dir))
    """
    tb.inject(consensus_setup, after='pixel_consensus_cluster')

    for fov in fovs:
        dummy_consensus_cmd = """
            sample_consensus = pd.DataFrame(np.random.rand(100, len(channels)), columns=channels)
            sample_consensus['clusters'] = np.arange(100)
            sample_consensus['hCluster_cap'] = np.repeat(np.arange(20), repeats=5)

            feather.write_dataframe(sample_consensus, os.path.join(base_dir,
                                                                   pixel_consensus_dir,
                                                                   '%s' + '.feather'))
        """ % fov

        tb.inject(dummy_consensus_cmd, after='pixel_consensus_cluster')


def flowsom_cell_run(tb):
    pass


def qc_notebook_setup(tb, base_dir, tiff_dir, sub_dir=None, fovs=None, chans=None):
    """Explicitly set the file parameters and desired fovs and channels needed
    Args:
        base_dir (str):
            The directory to store the tiff dir
        tiff_dir (str):
            The name of the tiff directory
        sub_dir (str):
            The name of the subdirectory to use for non-mibitiff image folders
        fovs (list):
            The list of fovs to subset over
        chans (list):
            The list of channels to subset over
    """

    # import modules
    tb.execute_cell('import')

    # set the user's MIBItracker info (use hard-coded values)
    tb.execute_cell('set_mibitracker_info')

    # convert sub_dir, fovs, and chans to appropriate string representations
    img_sub_folder = "None" if sub_dir is None else "\"%s\"" % sub_dir
    fov_list = "None" if fovs is None else str(fovs)
    chan_list = "None" if chans is None else str(chans)

    data_paths = """
        base_dir = '%s'
        tiff_dir = '%s'
        img_sub_folder = %s
    """ % (base_dir, tiff_dir, img_sub_folder)
    tb.inject(data_paths, after='set_data_info')

    fov_chans = """
        fovs = %s
        channels = %s
    """ % (fov_list, chan_list)
    tb.inject(fov_chans, after='set_fovs_chans')


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
        tb.inject("fovs = %s" % str(fovs), after='load_fovs')
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


def run_qc_comp(tb, gauss_blur=False):
    """Runs the QC computation process with the hard-coded inputs from qc_notebook_setup

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        gauss_blur (bool):
            Whether to include Gaussian blurring
    """

    # download the data off the MIBItracker
    tb.execute_cell('download_mibitracker')

    # set the Gaussian blurring parameters
    gauss_blur = """
        gaussian_blur = %s
        blur_factor = 1
    """ % str(gauss_blur)
    tb.inject(gauss_blur, after='set_gaussian_blur')

    # run QC metric extraction
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


def generate_sample_feature_tifs(fovs, deepcell_output_dir, img_shape=(50, 50)):
    """Generate a sample _feature_0 tif file for each fov

    Done to bypass the bottleneck of create_deepcell_output, for testing purposes we don't care
    about correct segmentation labels.

    Args:
        fovs (list):
            The list of fovs to generate sample _feature_0 tif files for
        deepcell_output_dir (str):
            The path to the output directory
        img_shape (tuple):
            Dimensions of the tifs to create
    """

    # generate a random image for each fov, set as both whole cell and nuclear
    for fov in fovs:
        rand_img = np.random.randint(0, 16, size=img_shape)
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
