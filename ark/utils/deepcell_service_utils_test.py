import pathlib
import os
import tempfile
from zipfile import ZipFile
import pytest

from ark.utils.deepcell_service_utils import create_deepcell_output


def mocked_run_deepcell(input_dir, output_dir, host, job_type):
    pathlib.Path(os.path.join(output_dir, 'fov1_feature_0.tif')).touch()
    pathlib.Path(os.path.join(output_dir, 'fov2_feature_0.tif')).touch()

    zip_path = os.path.join(output_dir, 'example_output.zip')
    with ZipFile(zip_path, 'w') as zipObj:
        for i in [1, 2]:
            filename = os.path.join(output_dir, f'fov{i}_feature_0.tif')
            zipObj.write(filename, os.path.basename(filename))
            os.remove(filename)


def test_create_deepcell_output(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        mocker.patch('ark.utils.deepcell_service_utils.run_deepcell_task', mocked_run_deepcell)

        input_dir = os.path.join(temp_dir, 'input_dir')
        os.makedirs(input_dir)

        output_dir = os.path.join(temp_dir, 'output_dir')
        os.makedirs(output_dir)
        pathlib.Path(os.path.join(input_dir, 'fov1.tif')).touch()
        pathlib.Path(os.path.join(input_dir, 'fov2.tif')).touch()

        create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                               fovs=['fov1', 'fov2'])

        # make sure DeepCell (.zip) output exists
        assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))

        # DeepCell output .zip file should be extracted
        assert os.path.exists(os.path.join(output_dir, 'fov1_feature_0.tif'))
        assert os.path.exists(os.path.join(output_dir, 'fov2_feature_0.tif'))

        # if fovs is None, all .tif files in input dir should be taken
        os.remove(os.path.join(output_dir, 'fov1_feature_0.tif'))
        os.remove(os.path.join(output_dir, 'fov2_feature_0.tif'))
        create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, 'fov1_feature_0.tif'))
        assert os.path.exists(os.path.join(output_dir, 'fov2_feature_0.tif'))

        # /fovs.zip file should be removed from input folder after completion
        assert not os.path.isfile(os.path.join(input_dir, 'fovs.zip'))

        pathlib.Path(os.path.join(input_dir, 'fovs.zip')).touch()
        # Warning should be displayed if /fovs.zip file exists (will be overwritten)
        with pytest.warns(UserWarning):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1'])

        # DeepCell output .tif file does not exist for some fov
        with pytest.warns(UserWarning):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   suffix='_other_suffix', fovs=['fov1'])
        pathlib.Path(os.path.join(input_dir, 'fov3.tif')).touch()
        with pytest.warns(UserWarning):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov2', 'fov3'])

        # ValueError should be raised if .tif file does not exists for some fov in fovs
        with pytest.raises(ValueError):
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov4'])
