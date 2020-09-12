import warnings
from twisted.internet import reactor
from kiosk_client import manager
import os
from pathlib import Path
import shutil


def create_deepcell_output(deepcell_input_dir, points, deepcell_output_dir,
                           host='https://deepcell.org', job_type='multiplex'):
    """ Handles all of the necessary data manipulation for running deepcell tasks.

        Creates .zip files (to be used as input for DeepCell),
        calls run_deepcell_task method,
        and extracts zipped output files to the specified output location

        Args:
            points (list):              List of points in preprocessing pipeline

            deepcell_input_dir (str):   Location of preprocessed files
                                        (assume deepcell_input_dir contains <point>.tif
                                        for each point in points list)

            deepcell_output_dir (str):  Location to save DeepCell output (as .tif)

            host:                       Hostname and port for the kiosk-frontend API server
                                        Default: 'https://deepcell.org'

            job_type:                   Name of job workflow (multiplex, segmentation, tracking)
                                        Default: 'multiplex'

        Output:
            Writes DeepCell service .tif output to output_dir

        Raises:
            UserWarning:                Raised if DeepCell output .zip file
                                        was not created for some point in points list.

        """

    # create /zip folders if they do not exist, and make sure they are empty
    input_zip = os.path.join(deepcell_input_dir, 'zip')
    output_zip = os.path.join(deepcell_output_dir, 'zip')
    for directory in [input_zip, output_zip]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        [f.unlink() for f in Path(directory).glob("*") if f.is_file()]

    # zip the preprocessed .tif files
    for i, point in enumerate(points):
        shutil.make_archive(os.path.join(input_zip, point), 'zip',
                            deepcell_input_dir, point + '.tif')

    run_deepcell_task(input_zip, output_zip, host, job_type)

    # make sure DeepCell output files were created
    _, _, files = next(os.walk(output_zip))
    if len(files) != len(points):
        warnings.warn(f'failed to create {len(points) - len(files)} files')

    # finally extract the .zip files
    for f in files:
        shutil.unpack_archive(os.path.join(output_zip, f),
                              deepcell_output_dir, 'zip')


def run_deepcell_task(input_dir, output_dir, host='https://deepcell.org',
                      job_type='multiplex'):
    """Uses kiosk-client to run DeepCell task and saves output to output_dir.
        (https://github.com/vanvalenlab/kiosk-client)

        Args:
            input_dir: location of .zip files

            output_dir: location to save deepcell output (as .zip)

            host: Hostname and port for the kiosk-frontend API server.
                  Default: 'https://deepcell.org'

            job_type: Name of job workflow (multiplex, segmentation, tracking).
                      Default: 'multiplex'
        Output:
            Writes DeepCell service .zip output to output_dir

        """

    # more configuration parameters can be set. https://github.com/vanvalenlab/kiosk-client
    mgr_kwargs = {
        'host': host,
        'job_type': job_type,
        'download_results': True,
        'output_dir': output_dir
    }

    mgr = manager.BatchProcessingJobManager(**mgr_kwargs)
    mgr.run(filepath=input_dir)
    reactor.run()
