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

# import ark.phenotyping.pixie_preprocessing as pixie_preprocessing
import ark.phenotyping.pixie_preprocessing as pixie_preprocessing

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


def test_create_fov_pixel_data():
    # tests for all fovs and some fovs
    fovs = ['fov0', 'fov1']
    chans = ['chan0', 'chan1', 'chan2']

    # create sample data
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=fovs,
                                                  channel_names=chans)

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=fovs,
                                                  compartment_names=['whole_cell'])

    # test for each fov
    for fov in fovs:
        sample_img_data = sample_img_xr.loc[fov, ...].values.astype(np.float32)
        seg_labels = sample_labels.loc[fov, ...].values.reshape(10, 10)

        # TEST 1: run fov preprocessing for one fov with seg_labels and no blank pixels
        sample_pixel_mat, sample_pixel_mat_subset = pixie_preprocessing.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=seg_labels
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-4],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-4],
                                        provided_chans=chans)

        # assert all rows sum to 1 (within tolerance because of floating-point errors)
        assert np.all(np.allclose(sample_pixel_mat.loc[:, chans].sum(axis=1).values, 1))

        # assert we didn't lose any pixels for this test
        assert sample_pixel_mat.shape[0] == (sample_img_data.shape[0] * sample_img_data.shape[1])

        # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(sample_pixel_mat.shape[0] * 0.1) == sample_pixel_mat_subset.shape[0]

        # TEST 2: run fov preprocessing for one fov without seg_labels and no blank pixels
        sample_pixel_mat, sample_pixel_mat_subset = pixie_preprocessing.create_fov_pixel_data(
            fov=fov, channels=chans, img_data=sample_img_data, seg_labels=None
        )

        # assert the channel names are the same
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat.columns.values[:-3],
                                        provided_chans=chans)
        misc_utils.verify_same_elements(flowsom_chans=sample_pixel_mat_subset.columns.values[:-3],
                                        provided_chans=chans)

        # assert all rows sum to 1 (within tolerance because of floating-point errors)
        assert np.all(np.allclose(sample_pixel_mat.loc[:, chans].sum(axis=1).values, 1))

        # assert we didn't lose any pixels for this test
        assert sample_pixel_mat.shape[0] == (sample_img_data.shape[0] * sample_img_data.shape[1])

        # assert the size of the subsetted DataFrame is 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(sample_pixel_mat.shape[0] * 0.1) == sample_pixel_mat_subset.shape[0]

        # TODO: add a test where after Gaussian blurring one or more rows in sample_pixel_mat
        # are all 0 after, tested successfully via hard-coding values in create_fov_pixel_data


def test_preprocess_fov(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define the channel names
        chans = ['chan0', 'chan1', 'chan2']

        # define sample data_dir and subset_dir paths, and make them
        data_dir = os.path.join(temp_dir, 'pixel_mat_data')
        subset_dir = os.path.join(temp_dir, 'pixel_mat_subsetted')
        os.mkdir(data_dir)
        os.mkdir(subset_dir)

        # create sample image data
        # NOTE: test_create_pixel_matrix tests if the sub_dir is None
        tiff_dir = os.path.join(temp_dir, 'sample_image_data')
        os.mkdir(tiff_dir)
        test_utils.create_paired_xarray_fovs(
            base_dir=tiff_dir, fov_names=['fov0', 'fov1'],
            channel_names=chans, sub_dir='TIFs', img_shape=(10, 10)
        )

        # create a dummy segmentation directory
        # NOTE: test_create_pixel_matrix handles the case with no segmentation labels provided
        seg_dir = os.path.join(temp_dir, 'segmentation')
        os.mkdir(seg_dir)

        # create sample segmentation data
        for fov in ['fov0', 'fov1']:
            rand_img = np.random.randint(0, 16, size=(10, 10))
            file_name = fov + "_whole_cell.tiff"
            image_utils.save_image(os.path.join(seg_dir, file_name), rand_img)

        # run the preprocessing for fov0
        # NOTE: don't test the return value, leave that for test_create_pixel_matrix
        pixie_preprocessing.preprocess_fov(
            temp_dir, tiff_dir, 'pixel_mat_data', 'pixel_mat_subsetted',
            seg_dir, '_whole_cell.tiff', 'TIFs', False, ['chan0', 'chan1', 'chan2'],
            2, 0.1, 42, 'fov0'
        )

        fov_data_path = os.path.join(
            temp_dir, 'pixel_mat_data', 'fov0.feather'
        )
        fov_sub_path = os.path.join(
            temp_dir, 'pixel_mat_subsetted', 'fov0.feather'
        )

        # assert we actually created a .feather preprocessed file
        # for each fov
        assert os.path.exists(fov_data_path)

        # assert that we actually created a .feather subsetted file
        # for each fov
        assert os.path.exists(fov_sub_path)

        # get the data for the specific fov
        flowsom_data_fov = feather.read_dataframe(fov_data_path)
        flowsom_sub_fov = feather.read_dataframe(fov_sub_path)

        # assert the channel names are the same
        misc_utils.verify_same_elements(
            flowsom_chans=flowsom_data_fov.columns.values[:-4],
            provided_chans=chans
        )

        # assert no rows sum to 0
        assert np.all(flowsom_data_fov.loc[:, chans].sum(
            axis=1
        ).values != 0)

        # assert the subsetted DataFrame size is 0.1 of the preprocessed DataFrame
        # NOTE: need to account for rounding if multiplying by 0.1 leads to non-int
        assert round(flowsom_data_fov.shape[0] * 0.1) == flowsom_sub_fov.shape[0]


def mocked_create_fov_pixel_data(fov, channels, img_data, seg_labels, blur_factor,
                                 subset_proportion):
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
                          subset_proportion, seed, fov):
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

    # set seed for subsetting
    np.random.seed(seed)

    # create the full and subsetted fov matrices
    pixel_mat, pixel_mat_subset = mocked_create_fov_pixel_data(
        fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
        blur_factor=blur_factor,
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
            pixie_preprocessing.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                seg_dir=None,
                subset_proportion=1.1
            )

        # pass invalid base directory
        with pytest.raises(FileNotFoundError):
            pixie_preprocessing.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir='bad_base_dir',
                tiff_dir=tiff_dir,
                seg_dir=None
            )

        # pass invalid tiff directory
        with pytest.raises(FileNotFoundError):
            pixie_preprocessing.create_pixel_matrix(
                fovs=['fov1', 'fov2'],
                channels=['chan1'],
                base_dir=temp_dir,
                tiff_dir='bad_tiff_dir',
                seg_dir=None
            )

        # pass invalid pixel output directory
        with pytest.raises(FileNotFoundError):
            pixie_preprocessing.create_pixel_matrix(
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
            pixie_preprocessing.create_pixel_matrix(
                fovs=['fov1', 'fov2', 'fov3'],
                channels=chans,
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                img_sub_folder=sub_dir,
                seg_dir=seg_dir
            )

        # pass invalid channel names
        with pytest.raises(ValueError):
            pixie_preprocessing.create_pixel_matrix(
                fovs=fovs,
                channels=['chan1', 'chan2', 'chan3'],
                base_dir=temp_dir,
                tiff_dir=tiff_dir,
                img_sub_folder=sub_dir,
                seg_dir=seg_dir
            )

        # create the pixel matrices
        pixie_preprocessing.create_pixel_matrix(
            fovs=fovs,
            channels=chans,
            base_dir=temp_dir,
            tiff_dir=tiff_dir,
            img_sub_folder=sub_dir,
            seg_dir=seg_dir,
            multiprocess=multiprocess
        )

        # check that we actually created a data directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_data'))

        # check that we actually created a subsetted directory
        assert os.path.exists(os.path.join(temp_dir, 'pixel_mat_subsetted'))

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
            'ark.phenotyping.pixie_preprocessing.preprocess_fov',
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

        pixie_preprocessing.create_pixel_matrix(
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
    pixie_preprocessing.create_pixel_matrix(
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

        pixie_preprocessing.create_pixel_matrix(
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

        pixie_preprocessing.create_pixel_matrix(
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

        pixie_preprocessing.create_pixel_matrix(
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

        pixie_preprocessing.create_pixel_matrix(
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
