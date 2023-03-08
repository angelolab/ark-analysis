import os
import tempfile
from shutil import rmtree

import feather
import numpy as np
import pandas as pd
import pytest
import skimage.io as io
from pytest_cases import parametrize_with_cases
from alpineer import image_utils, io_utils, load_utils, misc_utils, test_utils

import ark.phenotyping.cluster_helpers as cluster_helpers
import ark.phenotyping.pixel_som_clustering as pixel_som_clustering

parametrize = pytest.mark.parametrize

PIXEL_MATRIX_FOVS = ['fov0', 'fov1', 'fov2']
PIXEL_MATRIX_CHANS = ['chan0', 'chan1', 'chan2']


class CreatePixelMatrixBaseCases:
    def case_all_fovs_all_chans(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, False, False, False

    def case_some_fovs_some_chans(self):
        return PIXEL_MATRIX_FOVS[:2], PIXEL_MATRIX_CHANS[:2], 'TIFs', True, False, False, False

    def case_no_sub_dir(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, None, True, False, False, False

    def case_no_seg_dir(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', False, False, False, False

    def case_existing_channel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, False, False

    def case_existing_pixel_thresh(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, False, True, False

    def case_existing_pixel_and_channel_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, True, False

    def case_new_channels_norm(self):
        return PIXEL_MATRIX_FOVS, PIXEL_MATRIX_CHANS, 'TIFs', True, True, True, True


def mocked_create_fov_pixel_data(fov, channels, img_data, seg_labels, blur_factor,
                                 subset_proportion, pixel_thresh_val):
    # create fake data to be compatible with downstream functions
    data = np.random.rand(len(channels) * 5).reshape(5, len(channels))
    df = pd.DataFrame(data, columns=channels)

    # hard code the metadata columns
    df['fov'] = fov
    df['row_index'] = -1
    df['column_index'] = -1
    if seg_labels is not None:
        df['seg_labels'] = -1

    # verify that each channel is 2x the previous
    for i in range(len(channels) - 1):
        assert np.allclose(img_data[..., i] * 2, img_data[..., i + 1])

    return df, df


def mocked_preprocess_fov(base_dir, tiff_dir, data_dir, subset_dir, seg_dir, seg_suffix,
                          img_sub_folder, is_mibitiff, channels, blur_factor,
                          subset_proportion, pixel_thresh_val, seed, channel_norm_df, fov):
    # load img_xr from MIBITiff or directory with the fov
    if is_mibitiff:
        img_xr = load_utils.load_imgs_from_mibitiff(
            tiff_dir, mibitiff_files=[fov])
    else:
        img_xr = load_utils.load_imgs_from_tree(
            tiff_dir, img_sub_folder=img_sub_folder, fovs=[fov])

    # ensure the provided channels will actually exist in img_xr
    misc_utils.verify_in_list(
        provided_chans=channels,
        pixel_mat_chans=img_xr.channels.values
    )

    # if seg_dir is None, leave seg_labels as None
    seg_labels = None

    # otherwise, load segmentation labels in for fov
    if seg_dir is not None:
        seg_labels = io.imread(os.path.join(seg_dir, fov + seg_suffix))

    # subset for the channel data
    img_data = img_xr.loc[fov, :, :, channels].values.astype(np.float32)

    # create vector for normalizing image data
    norm_vect = channel_norm_df['norm_val'].values
    norm_vect = np.array(norm_vect).reshape([1, 1, len(norm_vect)])

    # normalize image data
    img_data = img_data / norm_vect

    # set seed for subsetting
    np.random.seed(seed)

    # create the full and subsetted fov matrices
    pixel_mat, pixel_mat_subset = mocked_create_fov_pixel_data(
        fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
        pixel_thresh_val=pixel_thresh_val, blur_factor=blur_factor,
        subset_proportion=subset_proportion
    )

    # write complete dataset to feather, needed for cluster assignment
    feather.write_dataframe(pixel_mat,
                            os.path.join(base_dir,
                                         data_dir,
                                         fov + ".feather"),
                            compression='uncompressed')

    # write subseted dataset to feather, needed for training
    feather.write_dataframe(pixel_mat_subset,
                            os.path.join(base_dir,
                                         subset_dir,
                                         fov + ".feather"),
                            compression='uncompressed')

    return pixel_mat


# TODO: leaving out MIBItiff testing until someone needs it
@parametrize_with_cases(
    'fovs,chans,sub_dir,seg_dir_include,channel_norm_include,pixel_thresh_include,norm_diff_chan',
    cases=CreatePixelMatrixBaseCases
)
@parametrize('multiprocess', [True, False])
def test_create_pixel_matrix_base(fovs, chans, sub_dir, seg_dir_include,
                                  channel_norm_include, pixel_thresh_include,
                                  norm_diff_chan, multiprocess, mocker, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a directory to store the image data
        tiff_dir = os.path.join(temp_dir, 'sample_image_data')
        os.mkdir(tiff_dir)

        # invalid subset proportion specified
        with pytest.raises(ValueError):
            pixel_som_clustering.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                seg_dir=None,
                subset_proportion=1.1
            )

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            pixel_som_clustering.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir='bad_base_dir',
                tiff_dir=tiff_dir,
                seg_dir=None
            )

        # pass invalid tiff directory
        with pytest.raises(FileNotFoundError):
            pixel_som_clustering.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir='bad_tiff_dir',
                seg_dir=None
            )

        # pass invalid pixel output directory
        with pytest.raises(FileNotFoundError):
            pixel_som_clustering.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir=temp_dir,
                seg_dir=None,
                pixel_output_dir='bad_output_dir'
            )

        # make a dummy pixel_output_dir
        sample_pixel_output_dir = os.path.join(temp_dir, 'pixel_output_dir')
        os.mkdir(sample_pixel_output_dir)

        # create a dummy seg_dir with data if we're on a test that requires segmentation labels
        if seg_dir_include:
            seg_dir = os.path.join(temp_dir, 'segmentation')
            os.mkdir(seg_dir)

            # create sample segmentation data
            for fov in fovs:
                rand_img = np.random.randint(0, 16, size=(10, 10))
                file_name = fov + "_whole_cell.tiff"
                image_utils.save_image(os.path.join(seg_dir, file_name), rand_img)
        # otherwise, set seg_dir to None
        else:
            seg_dir = None

        # create sample image data
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir, fov_names=fovs,
            channel_names=['chan0', 'chan1', 'chan2'], sub_dir=sub_dir, img_shape=(10, 10)
        )

        # pass invalid fov names (fails in load_imgs_from_tree)
        with pytest.raises(FileNotFoundError):
            pixel_som_clustering.create_pixel_matrix(
                fovs=['fov1', 'fov2', 'fov3'],
                channels=chans,
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                img_sub_folder=sub_dir,
                seg_dir=seg_dir
            )

        # pass invalid channel names
        with pytest.raises(ValueError):
            pixel_som_clustering.create_pixel_matrix(
                fovs=fovs,
                channels=['chan1', 'chan2', 'chan3'],
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                img_sub_folder=sub_dir,
                seg_dir=seg_dir
            )

        # make the channel_norm.feather file if the test requires it
        # NOTE: pixel_mat_data already created in the previous validation tests
        if channel_norm_include:
            # helps test if channel_norm.feather contains a different set of channels
            norm_chans = [chans[0]] if norm_diff_chan else chans
            sample_channel_norm_df = pd.DataFrame(
                np.expand_dims(np.random.rand(len(norm_chans)), axis=0),
                columns=norm_chans
            )

            feather.write_dataframe(
                sample_channel_norm_df,
                os.path.join(temp_dir, sample_pixel_output_dir, 'channel_norm.feather'),
                compression='uncompressed'
            )

        # make the pixel_thresh.feather file if the test requires it
        if pixel_thresh_include:
            sample_pixel_thresh_df = pd.DataFrame({'pixel_thresh_val': np.random.rand(1)})
            feather.write_dataframe(
                sample_pixel_thresh_df,
                os.path.join(temp_dir, sample_pixel_output_dir, 'pixel_thresh.feather'),
                compression='uncompressed'
            )

        # create the pixel matrices
        pixel_som_clustering.create_pixel_matrix(
            fovs=fovs,
            channels=chans,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=sub_dir,
            seg_dir=seg_dir,
            multiprocess=multiprocess
        )

        # assert we overwrote the original channel_norm and pixel_thresh files
        # if new set of channels provided
        if norm_diff_chan:
            output_capture = capsys.readouterr().out
            assert 'New channels provided: overwriting whole cohort' in output_capture

        # check that we actually created a data directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_data'))

        # check that we actually created a subsetted directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted'))

        # if there wasn't originally a channel_norm.feather or if overwritten, assert one created
        if not channel_norm_include or norm_diff_chan:
            assert os.path.exists(
                os.path.join(temp_dir, sample_pixel_output_dir, 'channel_norm.feather')
            )

        # if there wasn't originally a pixel_thresh.feather or if overwritten, assert one created
        if not pixel_thresh_include or norm_diff_chan:
            assert os.path.exists(
                os.path.join(temp_dir, sample_pixel_output_dir, 'pixel_thresh.feather')
            )

        # check that we created a norm vals file
        assert os.path.exists(os.path.join(temp_dir, 'channel_norm_post_rowsum.feather'))

        for fov in fovs:
            fov_data_path = os.path.join(
                temp_dir, 'pixel_mat_data', fov + '.feather'
            )
            fov_sub_path = os.path.join(
                temp_dir, 'pixel_mat_subsetted', fov + '.feather'
            )

            # assert we actually created a .feather preprocessed file for each fov
            assert os.path.exists(fov_data_path)

            # assert that we actually created a .feather subsetted file for each fov
            assert os.path.exists(fov_sub_path)

            # get the data for the specific fov
            flowsom_data_fov = feather.read_dataframe(fov_data_path)
            flowsom_sub_fov = feather.read_dataframe(fov_sub_path)

            # assert the channel names are the same
            chan_index_stop = -3 if seg_dir is None else -4
            misc_utils.verify_same_elements(
                flowsom_chans=flowsom_data_fov.columns.values[:chan_index_stop],
                provided_chans=chans
            )

            # assert no rows sum to 0
            assert np.all(flowsom_data_fov.loc[:, chans].sum(
                axis=1
            ).values != 0)

            # assert the subsetted DataFrame size is 0.1 of the preprocessed DataFrame
            # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
            assert round(flowsom_data_fov.shape[0] * 0.1) == flowsom_sub_fov.shape[0]

        # check that correct values are passed to helper function
        mocker.patch(
            'ark.phenotyping.pixel_cluster_utils.preprocess_fov',
            mocked_preprocess_fov
        )

        if sub_dir is None:
            sub_dir = ''

        # create fake data where all channels in each fov are the same
        new_tiff_dir = os.path.join(temp_dir, 'new_tiff_dir')
        os.makedirs(new_tiff_dir)
        for fov in fovs:
            img = (np.random.rand(100) * 100).reshape((10, 10))
            fov_dir = os.path.join(new_tiff_dir, fov, sub_dir)
            os.makedirs(fov_dir)
            for chan in chans:
                image_utils.save_image(os.path.join(fov_dir, chan + '.tiff'), img)

        # recreate the output directory
        rmtree(sample_pixel_output_dir)
        os.mkdir(sample_pixel_output_dir)

        # create normalization file
        data_dir = os.path.join(temp_dir, 'pixel_mat_data')

        # generate the data
        mults = [(1 / 2) ** i for i in range(len(chans))]

        sample_channel_norm_df = pd.DataFrame(
            np.expand_dims(mults, axis=0),
            columns=chans
        )

        feather.write_dataframe(
            sample_channel_norm_df,
            os.path.join(temp_dir, sample_pixel_output_dir, 'channel_norm.feather'),
            compression='uncompressed'
        )

        pixel_som_clustering.create_pixel_matrix(
            fovs=fovs,
            channels=chans,
            base_dir=temp_dir,
            tiff_dir=new_tiff_dir,
            img_sub_folder=sub_dir,
            seg_dir=seg_dir,
            multiprocess=multiprocess
        )


# TODO: clean up the following tests
def generate_create_pixel_matrix_test_data(temp_dir):
    # create a directory to store the image data
    tiff_dir = os.path.join(temp_dir, 'sample_image_data')
    os.mkdir(tiff_dir)

    # create sample image data
    test_utils.create_paired_xarray_fovs(
        base_dir=tiff_dir, fov_names=PIXEL_MATRIX_FOVS,
        channel_names=PIXEL_MATRIX_CHANS, sub_dir=None, img_shape=(10, 10)
    )

    # make a sample pixel_output_dir
    os.mkdir(os.path.join(temp_dir, 'pixel_output_dir'))

    # create the data, this is just for generation
    pixel_som_clustering.create_pixel_matrix(
        fovs=PIXEL_MATRIX_FOVS,
        channels=PIXEL_MATRIX_CHANS,
        base_dir=temp_dir,
        tiff_dir=tiff_dir,
        img_sub_folder=None,
        seg_dir=None
    )


@parametrize('multiprocess', [True, False])
def test_create_pixel_matrix_missing_fov(multiprocess, capsys):
    fov_files = [fov + '.feather' for fov in PIXEL_MATRIX_FOVS]

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_create_pixel_matrix_test_data(temp_dir)
        capsys.readouterr()

        tiff_dir = os.path.join(temp_dir, 'sample_image_data')

        # test the case where we've already written FOVs to both data and subset folder
        os.remove(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'))
        os.remove(os.path.join(temp_dir, 'pixel_mat_subsetted', 'fov1.feather'))
        sample_quant_data = pd.DataFrame(
            np.random.rand(3, 2),
            index=PIXEL_MATRIX_CHANS,
            columns=['fov0', 'fov2']
        )
        feather.write_dataframe(
            sample_quant_data,
            os.path.join(temp_dir, 'pixel_output_dir', 'quant_dat.feather')
        )

        pixel_som_clustering.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None,
            multiprocess=multiprocess
        )

        output_capture = capsys.readouterr().out
        assert output_capture == (
            "Restarting preprocessing from FOV fov1, 1 fovs left to process\n"
            "Processed 1 fovs\n"
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )

        capsys.readouterr()

        # test the case where we've written a FOV to data but not subset
        # NOTE: in this case, the value in quant_dat will also not have been written
        os.remove(os.path.join(temp_dir, 'pixel_mat_subsetted', 'fov1.feather'))
        feather.write_dataframe(
            sample_quant_data,
            os.path.join(temp_dir, 'pixel_output_dir', 'quant_dat.feather')
        )

        pixel_som_clustering.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None,
            multiprocess=multiprocess
        )

        output_capture = capsys.readouterr().out
        assert output_capture == (
            "Restarting preprocessing from FOV fov1, 1 fovs left to process\n"
            "Processed 1 fovs\n"
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )

        # test the case where we've written a FOV to subset but not data (very rare)
        # NOTE: in this case, the value in quant_dat will also not have been written
        os.remove(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'))
        feather.write_dataframe(
            sample_quant_data,
            os.path.join(temp_dir, 'pixel_output_dir', 'quant_dat.feather')
        )

        pixel_som_clustering.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None,
            multiprocess=multiprocess
        )

        output_capture = capsys.readouterr().out
        assert output_capture == (
            "Restarting preprocessing from FOV fov1, 1 fovs left to process\n"
            "Processed 1 fovs\n"
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )


def test_create_pixel_matrix_all_fovs(capsys):
    fov_files = [fov + '.feather' for fov in PIXEL_MATRIX_FOVS]

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_create_pixel_matrix_test_data(temp_dir)
        capsys.readouterr()

        tiff_dir = os.path.join(temp_dir, 'sample_image_data')

        pixel_som_clustering.create_pixel_matrix(
            fovs=PIXEL_MATRIX_FOVS,
            channels=PIXEL_MATRIX_CHANS,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=None,
            seg_dir=None
        )

        output_capture = capsys.readouterr().out
        assert output_capture == "There are no more FOVs to preprocess, skipping\n"
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=fov_files
        )
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_subsetted')),
            written_files=fov_files
        )


# NOTE: overwrite functionality tested in cluster_helpers_test.py
def test_train_pixel_som():
    # basic error check: bad path to subsetted matrix
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            pixel_som_clustering.train_pixel_som(
                fovs=['fov0'], channels=['Marker1'],
                base_dir=temp_dir, subset_dir='bad_path'
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'column_index', 'segmentation_label']

        # make a dummy sub directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_subsetted'))

        for fov in fovs:
            # create the dummy data for each fov
            fov_sub_matrix = pd.DataFrame(np.random.rand(100, 8), columns=colnames)

            # write the dummy data to the subsetted data dir
            feather.write_dataframe(fov_sub_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_subsetted',
                                                                 fov + '.feather'))

        # create a dummy normalized file
        sample_norm_vals = pd.DataFrame(np.random.rand(1, len(chan_list)), columns=chan_list)
        feather.write_dataframe(
            sample_norm_vals, os.path.join(temp_dir, 'post_rowsum_chan_norm.feather')
        )

        # not all of the provided fovs exist
        with pytest.raises(ValueError):
            pixel_som_clustering.train_pixel_som(
                fovs=['fov2', 'fov3'], channels=chan_list, base_dir=temp_dir
            )

        # column mismatch between provided channels and subsetted data
        with pytest.raises(ValueError):
            pixel_som_clustering.train_pixel_som(
                fovs=fovs, channels=['Marker3', 'Marker4', 'MarkerBad'],
                base_dir=temp_dir
            )

        # train the pixel SOM
        pixel_pysom = pixel_som_clustering.train_pixel_som(
            fovs=fovs, channels=chan_list, base_dir=temp_dir
        )

        # assert the weights file has been created
        assert os.path.exists(pixel_pysom.weights_path)

        # assert that the dimensions of the weights are correct
        weights = feather.read_dataframe(pixel_pysom.weights_path)
        assert weights.shape == (100, 4)

        # assert that the SOM weights columns are the same as chan_list
        misc_utils.verify_same_elements(som_weights_channels=weights.columns.values,
                                        provided_channels=chan_list)


def generate_test_pixel_som_cluster_data(temp_dir, fovs, chans,
                                         generate_temp=False):
    # make it easy to name metadata columns
    meta_colnames = ['fov', 'row_index', 'column_index', 'segmentation_label']

    # create a dummy clustered matrix
    os.mkdir(os.path.join(temp_dir, 'pixel_mat_data'))

    # create a dummy temp directory if specified
    if generate_temp:
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_data_temp'))

    # store the intermediate FOV data in a dict for future comparison
    fov_data = {}

    # write dummy clustered data for each fov
    for fov in fovs:
        # create dummy preprocessed data for each fov
        fov_cluster_matrix = pd.DataFrame(
            np.random.rand(1000, len(chans)),
            columns=chans
        )

        # assign dummy metadata labels
        fov_cluster_matrix['fov'] = fov
        fov_cluster_matrix['row_index'] = np.repeat(np.arange(1, 101), repeats=10)
        fov_cluster_matrix['column_index'] = np.tile(np.arange(1, 101), reps=10)
        fov_cluster_matrix['segmentation_label'] = np.arange(1, 1001)

        # write the dummy data to pixel_mat_data
        feather.write_dataframe(fov_cluster_matrix, os.path.join(temp_dir,
                                                                 'pixel_mat_data',
                                                                 fov + '.feather'))

        fov_data[fov] = fov_cluster_matrix

    # if specified, write fov0 to pixel_mat_data_temp with sample pixel SOM clusters
    if generate_temp:
        # append a dummy meta column to fov0
        fov0_cluster_matrix = feather.read_dataframe(
            os.path.join(temp_dir, 'pixel_mat_data', 'fov0.feather')
        )
        fov0_cluster_matrix['pixel_som_cluster'] = np.repeat(np.arange(10), repeats=100)
        feather.write_dataframe(fov0_cluster_matrix, os.path.join(temp_dir,
                                                                  'pixel_mat_data_temp',
                                                                  'fov0.feather'))

    # generate example norm data
    norm_vals_path = os.path.join(temp_dir, 'sample_norm.feather')
    norm_data = pd.DataFrame(np.random.rand(1, len(chans)), columns=chans)
    feather.write_dataframe(norm_data, norm_vals_path)

    # generate example weights
    som_weights_path = os.path.join(temp_dir, 'pixel_weights.feather')
    som_weights_data = pd.DataFrame(np.random.rand(100, len(chans)), columns=chans)
    feather.write_dataframe(som_weights_data, som_weights_path)

    return fov_data, norm_vals_path, som_weights_path


@parametrize('multiprocess', [True, False])
def test_cluster_pixels_base(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        fov_data, norm_vals_path, som_weights_path = generate_test_pixel_som_cluster_data(
            temp_dir, fovs, chan_list
        )

        # error test: weights not assigned to PixelSOMCluster object
        with pytest.raises(ValueError):
            pixel_pysom_bad = cluster_helpers.PixelSOMCluster(
                os.path.join(temp_dir, 'pixel_mat_data'), norm_vals_path,
                'bad_path.feather', fovs, chan_list
            )
            pixel_som_clustering.cluster_pixels(fovs, chan_list, temp_dir, pixel_pysom_bad)

        # create a sample PixelSOMCluster object
        pixel_pysom = cluster_helpers.PixelSOMCluster(
            os.path.join(temp_dir, 'pixel_mat_data'), norm_vals_path, som_weights_path,
            fovs, chan_list
        )

        # run SOM cluster assignment
        pixel_som_clustering.cluster_pixels(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_mat_data', multiprocess=multiprocess
        )

        for fov in fovs:
            fov_cluster_data = feather.read_dataframe(os.path.join(temp_dir,
                                                                   'pixel_mat_data',
                                                                   fov + '.feather'))

            # assert we didn't assign any cluster 100 or above
            cluster_ids = fov_cluster_data['pixel_som_cluster']
            assert np.all(cluster_ids <= 100)

        # test overwrite functionality
        capsys.readouterr()

        # run SOM cluster assignment with overwrite flag
        pixel_som_clustering.cluster_pixels(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_mat_data', multiprocess=multiprocess,
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, reassigning SOM cluster labels to all FOVs\n"
        assert desired_status_updates in output

        # further ensures that all FOVs were overwritten
        assert "There are no more FOVs to assign SOM labels to" not in output


@parametrize('multiprocess', [True, False])
def test_cluster_pixels_corrupt(multiprocess, capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define fovs and channels
        fovs = ['fov0', 'fov1', 'fov2']
        chans = ['Marker1', 'Marker2', 'Marker3', 'Marker4']

        fov_data, norm_vals_path, som_weights_path = generate_test_pixel_som_cluster_data(
            temp_dir, fovs, chans
        )

        # create a sample PixelSOMCluster object
        pixel_pysom = cluster_helpers.PixelSOMCluster(
            os.path.join(temp_dir, 'pixel_mat_data'), norm_vals_path, som_weights_path, fovs, chans
        )

        # corrupt a fov for this test
        with open(os.path.join(temp_dir, 'pixel_mat_data', 'fov1.feather'), 'w') as outfile:
            outfile.write('baddatabaddatabaddata')

        capsys.readouterr()

        # run SOM cluster assignment
        pixel_som_clustering.cluster_pixels(
            fovs, chans, temp_dir, pixel_pysom, 'pixel_mat_data', multiprocess=multiprocess
        )

        # assert the _temp folder is now gone
        assert not os.path.exists(os.path.join(temp_dir, 'pixel_mat_data_temp'))

        output = capsys.readouterr().out
        desired_status_updates = "The data for FOV fov1 has been corrupted, skipping\n"
        assert desired_status_updates in output

        # verify that the FOVs in pixel_mat_data are correct
        # NOTE: fov1 should not be written because it was corrupted
        misc_utils.verify_same_elements(
            data_files=io_utils.list_files(os.path.join(temp_dir, 'pixel_mat_data')),
            written_files=['fov0.feather', 'fov2.feather']
        )


def test_generate_som_avg_files(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create list of markers and fovs we want to use
        chan_list = ['Marker1', 'Marker2', 'Marker3', 'Marker4']
        fovs = ['fov0', 'fov1', 'fov2']

        # make it easy to name columns
        colnames = chan_list + ['fov', 'row_index', 'column_index', 'segmentation_label']

        # define sample pixel data for each FOV
        pixel_data_path = os.path.join(temp_dir, 'pixel_data_dir')
        os.mkdir(pixel_data_path)
        for i, fov in enumerate(fovs):
            fov_cluster_data = pd.DataFrame(np.random.rand(100, len(colnames)), columns=colnames)
            fov_cluster_data['pixel_som_cluster'] = i + 1
            feather.write_dataframe(
                fov_cluster_data, os.path.join(pixel_data_path, fov + '.feather')
            )

        # define a sample norm vals file
        norm_vals_path = os.path.join(temp_dir, 'norm_vals.feather')
        norm_vals = pd.DataFrame(np.random.rand(1, 4), columns=chan_list)
        feather.write_dataframe(norm_vals, norm_vals_path)

        # define a sample weights file
        weights_path = os.path.join(temp_dir, 'pixel_weights.feather')
        weights = pd.DataFrame(np.random.rand(3, 4), columns=chan_list)
        feather.write_dataframe(weights, weights_path)

        # error test: weights not assigned to PixelSOMCluster object
        with pytest.raises(ValueError):
            pixel_pysom_bad = cluster_helpers.PixelSOMCluster(
                pixel_data_path, norm_vals_path, 'bad_path.feather', fovs, chan_list
            )
            pixel_som_clustering.generate_som_avg_files(fovs, chan_list, temp_dir, pixel_pysom_bad)

        # define an example PixelSOMCluster object
        pixel_pysom = cluster_helpers.PixelSOMCluster(
            pixel_data_path, norm_vals_path, weights_path, fovs, chan_list
        )

        # test base generation with all subsetted FOVs
        pixel_som_clustering.generate_som_avg_files(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=3
        )

        # assert we created SOM avg file
        pc_som_avg_file = os.path.join(temp_dir, 'pixel_channel_avg_som_cluster.csv')
        assert os.path.exists(pc_som_avg_file)

        # load in the SOM avg file, assert all clusters and counts are correct
        # NOTE: more intensive testing done by compute_pixel_cluster_channel_avg
        pc_som_avg_data = pd.read_csv(pc_som_avg_file)
        assert list(pc_som_avg_data['pixel_som_cluster']) == [1, 2, 3]
        assert np.all(pc_som_avg_data['count'] == 100)

        # test that process doesn't run if SOM cluster file already generated
        capsys.readouterr()

        pixel_som_clustering.generate_som_avg_files(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=1
        )

        output = capsys.readouterr().out
        assert output == "Already generated SOM cluster channel average file, skipping\n"

        # test overwrite functionality
        capsys.readouterr()

        # run SOM averaging with overwrite flg
        pixel_som_clustering.generate_som_avg_files(
            fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=3,
            overwrite=True
        )

        # ensure we reach the overwrite functionality logic
        output = capsys.readouterr().out
        desired_status_updates = \
            "Overwrite flag set, regenerating SOM cluster channel average file\n"
        assert desired_status_updates in output

        # remove average SOM file for final test
        os.remove(pc_som_avg_file)

        # ensure error gets thrown when not all SOM clusters make it in
        with pytest.raises(ValueError):
            pixel_som_clustering.generate_som_avg_files(
                fovs, chan_list, temp_dir, pixel_pysom, 'pixel_data_dir', num_fovs_subset=1
            )
