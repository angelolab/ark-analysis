import os
import tempfile
import pytest

from segmentation.utils import io_utils as iou


def test_list_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # set up temp_dir files
        filenames = [
            'tf.txt',
            'othertf.txt',
            'test.out',
            'test.csv',
        ]
        for filename in filenames:
            file = open(os.path.join(temp_dir, filename), 'w+')
            file.write('test')
            file.close()

        # test substrs is None (default)
        get_all = iou.list_files(temp_dir)
        assert get_all.sort() == filenames.sort()

        # test substrs is not list (single string)
        get_txt = iou.list_files(temp_dir, substrs='.txt')
        assert get_txt.sort() == filenames[0:2].sort()

        # test substrs is list
        get_test_and_other = iou.list_files(temp_dir, substrs=['test', 'other'])
        assert get_test_and_other.sort() == filenames[1:].sort()


def test_extract_delimited_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        filenames = [
            'Point1_restofname.txt',
            'Point2.txt',
        ]
        for filename in filenames:
            file = open(os.path.join(temp_dir, filename), 'w+')
            file.write('test')
            file.close()

        # test no files given (None/[])
        assert iou.extract_delimited_names(None) is None
        assert iou.extract_delimited_names([]) == []

        # non-optional delimiter warning
        with pytest.warns(UserWarning):
            iou.extract_delimited_names(['Point2.txt'], delimiter_optional=False)

        # test regular files list
        assert ['Point1', 'Point2'] == iou.extract_delimited_names(filenames)

        # test fullpath list
        fullpaths = [
            os.path.join(temp_dir, filename)
            for filename in filenames
        ]
        assert ['Point1', 'Point2'] == iou.extract_delimited_names(fullpaths)

        # test mixed
        assert ['Point1', 'Point2'] == iou.extract_delimited_names([fullpaths[0], filenames[1]])

    return


def test_list_folders():
    with tempfile.TemporaryDirectory() as temp_dir:
        # set up temp_dir subdirs
        dirnames = [
            'tf_txt',
            'othertf_txt',
            'test_csv',
            'test_out',
        ]
        for dirname in dirnames:
            os.mkdir(os.path.join(temp_dir, dirname))

        # test substrs is None (default)
        get_all = iou.list_folders(temp_dir)
        assert get_all.sort() == dirnames.sort()

        # test substrs is not list (single string)
        get_txt = iou.list_folders(temp_dir, substrs='_txt')
        assert get_txt.sort() == dirnames[0:2].sort()

        # test substrs is list
        get_test_and_other = iou.list_folders(temp_dir, substrs=['test_', 'other'])
        assert get_test_and_other.sort() == dirnames[1:].sort()
