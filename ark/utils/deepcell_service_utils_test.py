import pathlib
import os
import tempfile
from zipfile import ZipFile


from ark.utils.deepcell_service_utils import create_deepcell_output


def mocked_run_deepcell(input_dir, output_dir, host, job_type):
    pathlib.Path(os.path.join(output_dir, 'example_output.tiff')).touch()

    zip_path = os.path.join(output_dir, 'example_output.zip')
    with ZipFile(zip_path, 'w') as zipObj:
        filename = os.path.join(output_dir, 'example_output.tiff')
        zipObj.write(filename, os.path.basename(filename))


def test_create_deepcell_output(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        mocker.patch('ark.utils.deepcell_service_utils.run_deepcell_task', mocked_run_deepcell)

        input_dir = os.path.join(temp_dir, 'input_dir')
        os.makedirs(input_dir)

        output_dir = os.path.join(temp_dir, 'output_dir')
        os.makedirs(output_dir)
        pathlib.Path(os.path.join(input_dir, 'example_point.tif')).touch()

        create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                               points=['example_point'])

        assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))
