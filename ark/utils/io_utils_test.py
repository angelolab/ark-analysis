import os
import tempfile
import pathlib
import pytest

from ark.utils import io_utils as iou


def test_validate_paths():
    # change cwd to /scripts for more accurate testing
    # we'll ignore templates_qc because that is only run on the commercial MIBI computer
    os.chdir('templates_ark')

    # make a tempdir for testing
    with tempfile.TemporaryDirectory(dir='../data') as valid_path:

        # make valid subdirectory
        os.mkdir(valid_path + '/real_subdirectory')

        # extract parts of valid path to alter for test cases
        valid_parts = [p for p in pathlib.Path(valid_path).parts]
        valid_parts[0] = 'not_a_real_directory'

        # test no '../data' prefix
        starts_out_of_scope = os.path.join(*valid_parts)

        # construct test for bad middle folder path
        valid_parts[0] = '..'
        valid_parts[1] = 'data'
        valid_parts[2] = 'not_a_real_subdirectory'
        valid_parts.append('not_real_but_parent_is_problem')
        bad_middle_path = os.path.join(*valid_parts)

        # construct test for real path until file
        wrong_file = os.path.join(valid_path + '/real_subdirectory', 'not_a_real_file.tiff')

        # test one valid path
        iou.validate_paths(valid_path)

        # test multiple valid paths
        iou.validate_paths([valid_path, '../data', valid_path + '/real_subdirectory'])

        # test out-of-scope
        with pytest.raises(ValueError, match=r".*not_a_real_directory.*prefixed.*"):
            iou.validate_paths(starts_out_of_scope)

        # test mid-directory existence
        with pytest.raises(ValueError, match=r".*bad path.*not_a_real_subdirectory.*"):
            iou.validate_paths(bad_middle_path)

        # test file existence
        with pytest.raises(ValueError, match=r".*The file/path.*not_a_real_file.*"):
            iou.validate_paths(wrong_file)

    # make tempdir for testing outside of docker
    with tempfile.TemporaryDirectory() as valid_path:

        # make valid subdirectory
        valid_parts = [p for p in pathlib.Path(valid_path).parts]
        valid_parts[0] = 'not_a_real_directory'

        # test no '../data' prefix
        starts_out_of_scope = os.path.join(*valid_parts)

        # test out of scope when specifying out of scope
        with pytest.raises(ValueError, match=r".*not_a_real_directory*"):
            iou.validate_paths(starts_out_of_scope, data_prefix=False)

        with pytest.raises(ValueError, match=r".*is not prefixed with.*"):
            iou.validate_paths(starts_out_of_scope, data_prefix=True)

    # reset cwd after testing
    os.chdir('../')


def test_list_files():
    # test extension matching
    with tempfile.TemporaryDirectory() as temp_dir:
        # set up temp_dir files
        filenames = [
            'tf.txt',
            'othertf.txt',
            'test.out',
            'test.csv',
            '._fov-1-scan-1.json',
            '._fov-1-scan-1_pulse_heights.csv'
        ]
        for filename in filenames:
            pathlib.Path(os.path.join(temp_dir, filename)).touch()

        # add extra folder (shouldn't be picked up)
        os.mkdir(os.path.join(temp_dir, 'badfolder_test'))

        # test substrs is None (default)
        get_all = iou.list_files(temp_dir)
        assert sorted(get_all) == sorted(['tf.txt', 'othertf.txt', 'test.out', 'test.csv'])

        # test substrs is not list (single string)
        get_txt = iou.list_files(temp_dir, substrs='.txt')
        assert sorted(get_txt) == sorted(['othertf.txt', 'tf.txt'])

        # test substrs is list
        get_test_and_other = iou.list_files(temp_dir, substrs=['.txt', '.out'])
        assert sorted(get_test_and_other) == sorted(['tf.txt', 'othertf.txt', 'test.out'])

        # Test hidden files
        get_hidden_files = iou.list_files(temp_dir, substrs=['fov-1'], exact_match=False,
                                          ignore_hidden=False)
        assert sorted(get_hidden_files) == sorted(['._fov-1-scan-1.json',
                                                   '._fov-1-scan-1_pulse_heights.csv'])

    # test file name exact matching
    with tempfile.TemporaryDirectory() as temp_dir:
        filenames = ['.chan-metadata.tif', 'chan0.tif', 'chan.tif', 'c.tif']
        for filename in filenames:
            pathlib.Path(os.path.join(temp_dir, filename)).touch()

        # add extra folder (shouldn't be picked up)
        os.mkdir(os.path.join(temp_dir, 'badfolder_test'))

        # test substrs is None (default)
        get_all = iou.list_files(temp_dir, exact_match=True)
        assert sorted(get_all) == sorted(['chan0.tif', 'chan.tif', 'c.tif'])

        # test substrs is not list (single string)
        get_txt = iou.list_files(temp_dir, substrs='c', exact_match=True)
        assert sorted(get_txt) == [filenames[3]]

        # test substrs is list
        get_test_and_other = iou.list_files(temp_dir, substrs=['c', 'chan'], exact_match=True)
        assert sorted(get_test_and_other) == sorted(['chan.tif', 'c.tif'])

        # Test hidden files
        get_hidden_files = iou.list_files(temp_dir, substrs=['.chan-metadata'], exact_match=True,
                                          ignore_hidden=False)
        assert sorted(get_hidden_files) == ['.chan-metadata.tif']


def test_remove_file_extensions():
    # test a mixture of file paths and extensions
    files = ['fov1.tiff', 'fov2.tif', 'fov3.png', 'fov4.jpg']
    files2 = ['fov.1.tiff', 'fov.2.tiff', 'fov.3.png', 'fov.4']

    assert iou.remove_file_extensions(None) is None
    assert iou.remove_file_extensions([]) == []

    files_sans_ext = ['fov1', 'fov2', 'fov3', 'fov4']

    new_files = iou.remove_file_extensions(files)

    assert new_files == files_sans_ext

    with pytest.warns(UserWarning):
        new_files = iou.remove_file_extensions(['fov5.tar.gz', 'fov6.sample.csv'])
        assert new_files == ['fov5.tar', 'fov6.sample']

    with pytest.warns(UserWarning):
        new_files = iou.remove_file_extensions(files2)
        assert new_files == ['fov.1', 'fov.2', 'fov.3', 'fov.4']


def test_extract_delimited_names():
    filenames = [
        'fov1_restofname',
        'fov2',
    ]

    # test no files given (None/[])
    assert iou.extract_delimited_names(None) is None
    assert iou.extract_delimited_names([]) == []

    # non-optional delimiter warning
    with pytest.warns(UserWarning):
        iou.extract_delimited_names(['fov2'], delimiter='_', delimiter_optional=False)

    # test regular files list
    assert ['fov1', 'fov2'] == iou.extract_delimited_names(filenames, delimiter='_')


def test_list_folders():
    # Tests "Fuzzy Substring Matching",`exact_match` = False
    with tempfile.TemporaryDirectory() as temp_dir:
        # set up temp_dir subdirs
        dirnames = [
            'tf_txt',
            'othertf_txt',
            'test_csv',
            'test_out',
            'test_csv1',
            'test_csv2',
            'Ntest_csv',
            '.hidden_dir'
        ]

        dirnames.sort()
        for dirname in dirnames:
            os.mkdir(os.path.join(temp_dir, dirname))

        # add extra file
        pathlib.Path(os.path.join(temp_dir, 'test_badfile.txt')).touch()

        # test substrs is None (default)
        get_all = iou.list_folders(temp_dir, exact_match=False)
        assert sorted(get_all) == sorted(['tf_txt', 'othertf_txt', 'test_csv', 'test_out',
                                          'test_csv1', 'test_csv2', 'Ntest_csv'])

        # test substrs is not list (single string)
        get_txt = iou.list_folders(temp_dir, substrs='_txt', exact_match=False)
        assert sorted(get_txt) == sorted(['othertf_txt', 'tf_txt'])

        # test substrs is list
        get_test_and_other = iou.list_folders(
            temp_dir, substrs=['test_', 'other'], exact_match=False
        )
        assert sorted(get_test_and_other) == sorted(
            ['Ntest_csv', 'test_csv', 'test_csv1', 'test_csv2', 'test_out', 'othertf_txt']
        )

        # Test hidden files
        get_hidden_dirs = iou.list_folders(
            temp_dir, substrs="hidden", exact_match=False, ignore_hidden=False)
        assert get_hidden_dirs == [".hidden_dir"]

        # Tests "Exact Substring Matching", `exact_match` = True

        # Test substrs is None (default)
        get_all = iou.list_folders(temp_dir, exact_match=True)
        assert sorted(get_all) == sorted(['tf_txt', 'othertf_txt', 'test_csv', 'test_out',
                                          'test_csv1', 'test_csv2', 'Ntest_csv'])
        # Test exact substr is not list (single string)
        get_othertf_txt = iou.list_folders(temp_dir, substrs='othertf_txt', exact_match=True)
        assert get_othertf_txt == [dirnames[2]]

        # Test substrs, querying two folders (exactly)
        get_exact_n_substrs = iou.list_folders(
            temp_dir, substrs=['tf_txt', 'othertf_txt'], exact_match=True
        )
        assert sorted(get_exact_n_substrs) == ['othertf_txt', 'tf_txt']

        # Test the substr that the user specifies which is contained within multiple folders,
        # and only the folder that exactly matches the substring, not the one that contains it,
        # is returned when `exact_match=True`
        get_test_o = iou.list_folders(temp_dir, substrs='test_csv', exact_match=True)
        assert get_test_o == ["test_csv"]

        # Test hidden files
        get_hidden_dirs = iou.list_folders(
            temp_dir, substrs=".hidden_dir", exact_match=True, ignore_hidden=False)
        assert get_hidden_dirs == [".hidden_dir"]
