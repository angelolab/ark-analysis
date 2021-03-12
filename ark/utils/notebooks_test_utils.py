import os
import numpy as np
import skimage.io as io

from ark.utils import test_utils


def segment_notebook_setup(tb, deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir,
                           single_cell_dir, viz_dir, is_mibitiff=False,
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

    if is_mibitiff:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    use_delimiter=True)
        fovs = [f + mibitiff_suffix for f in fovs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            deepcell_tiff_dir, fovs, chans, img_shape=img_shape, mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            deepcell_tiff_dir, fovs, chans, img_shape=img_shape, delimiter='_', fills=False,
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


def flowsom_setup(tb, flowsom_dir, img_shape=(50, 50), num_fovs=3, num_chans=3,
                  is_mibitiff=False, mibitiff_suffix="-MassCorrected-Filtered", dtype=np.uint16):
    """Creates the directories, data, and MIBItiff settings for testing FlowSOM clustering

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
            tiff_dir, fovs, chans, img_shape=img_shape, delimiter='_', fills=False, dtype=dtype)

    # generate sample segmentation labels so we can load them in
    seg_dir = os.path.join(flowsom_dir, "deepcell_output")
    os.mkdir(seg_dir)
    generate_sample_feature_tifs(fovs, seg_dir, delimiter='_feature_0')

    # define custom paths
    define_paths = """
        base_dir = "%s"
        tiff_dir = "%s"
        segmentation_dir = "%s"
        preprocessed_dir = "pixel_mat_preprocessed"
        subsetted_dir = "pixel_mat_subsetted"
        som_clustered_dir = "pixel_mat_clustered"
        consensus_clustered_dir = "pixel_mat_consensus"
    """ % (flowsom_dir, tiff_dir, seg_dir)
    tb.inject(define_paths, after='file_path')

    # will set MIBItiff and MIBItiff_suffix
    tb.execute_cell('mibitiff_set')
    if is_mibitiff:
        # default setting is MIBItiff = False, change to True if user has mibitiff inputs
        tb.inject("MIBItiff = True", after='mibitiff_set')


def flowsom_set_fovs_channels(tb, channels, fovs=None):
    """Sets the fovs and channels variables

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        channels (list):
            The list of channels to subset in the image.
        fovs (list):
            If set, assigns the fovs variable to this list.
            If None, executes the default fov loading scheme in the 'load_fovs' cell.
    """

    if fovs is not None:
        # handles the case when the user assigns fovs to an explicit list
        tb.inject("fovs = %s" % str(fovs), after='load_fovs')
    else:
        # handles the case when the user allows list_files or list_folders to do the fov loading
        tb.execute_cell('load_fovs')

    # sets the channels accordingly
    tb.inject("channels = %s" % str(channels), after='set_channels')


def flowsom_run(tb, fovs, channels, is_mibitiff=False):
    """Run the FlowSOM clustering

    Args:
        tb (testbook.testbook):
            The testbook runner instance
        fovs (list):
            The list of fovs
        channels (list):
            The list of channels
        is_mibitiff (bool):
            Whether we're working with mibitiff im
    """

    # test the preprocessing works, we won't save nor run the actual FlowSOM clustering
    if is_mibitiff:
        mibitiff_preprocess = """
            som_utils.create_pixel_matrix(
                fovs, channels, base_dir, tiff_dir, segmentation_dir, is_mibitiff=True
            )
        """

        tb.inject(mibitiff_preprocess, after='gen_pixel_mat')
    else:
        tb.execute_cell('gen_pixel_mat')

    # create a dummy weights .feather
    dummy_weights = """
        import feather
        weights = pd.DataFrame(np.random.rand(100, len(channels)), columns=channels)

        feather.write_dataframe(weights, os.path.join(base_dir, 'weights.hdf5'))
    """

    tb.inject(dummy_weights, after='train_som')

    # create dummy clustered .feathers for each fov
    cluster_setup = """
        if not os.path.exists(os.path.join(base_dir, 'pixel_mat_preprocessed')):
            os.mkdir(os.path.join(base_dir, 'pixel_mat_preprocessed'))
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
                                                            'pixel_mat_preprocessed',
                                                            '%s' + '.feather'))
        """ % (str(channels), fov, fov)

        tb.inject(dummy_cluster_cmd, after='cluster_pixel_mat')

    # create a dummy consensus clustered data file
    dummy_consensus = """
        sample_consensus = pd.DataFrame(np.random.rand(100, len(channels)), columns=channels)
        sample_consensus['clusters'] = np.arange(100)
        sample_consensus['hCluster_cap'] = np.repeat(np.arange(20), repeats=5)

        feather.write_dataframe(sample_consensus, os.path.join(base_dir,
                                                               'cluster_consensus.feather'))
    """

    tb.inject(dummy_consensus, after='consensus_cluster')


def fov_channel_input_set(tb, fovs=None, nucs_list=None, mems_list=None):
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

    # set the channels accordingly
    tb.execute_cell('set_channels')

    # load data accordingly
    tb.execute_cell('load_data_xr')

    # generate the deepcell input files
    tb.execute_cell('gen_input')


def generate_sample_feature_tifs(fovs, deepcell_output_dir, img_shape=(50, 50), delimiter=None):
    """Generate a sample _feature_0 tif file for each fov

    Done to bypass the bottleneck of create_deepcell_output, for testing purposes we don't care
    about correct segmentation labels.

    Args:
        fovs (list):
            The list of fovs to generate sample _feature_0 tif files for
        deepcell_output_dir (str):
            The path to the output directory
        img_shape (tuple):
            The dimensions of the image to generate
        delimiter (str):
            The name of the delimiter to add to the TIF, if specified
    """

    for fov in fovs:
        rand_img = np.random.randint(0, 16, size=img_shape)

        # add the delimiter if specified
        if delimiter is not None:
            file_name = fov + "%s.tif" % delimiter
        else:
            file_name = fov + ".tif"

        io.imsave(os.path.join(deepcell_output_dir, file_name), rand_img)


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
    tb.execute_cell('save_mask')


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
