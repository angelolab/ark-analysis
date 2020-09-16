from twisted.internet import reactor
from kiosk_client import manager
import os
import glob
from zipfile import ZipFile


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

    # first remove all .zip files from output directory
    list_of_files = glob.glob(os.path.join(deepcell_output_dir, '*.zip'))
    for f in list_of_files:
        os.remove(f)

    zip_path = os.path.join(deepcell_input_dir, 'points.zip')
    with ZipFile(zip_path, 'w') as zipObj:
        for point in points:
            filename = os.path.join(deepcell_input_dir, point + '.tif')
            zipObj.write(filename, os.path.basename(filename))

    run_deepcell_task(zip_path, deepcell_output_dir, host, job_type)

    # extract the .tif output
    zf = glob.glob(os.path.join(deepcell_output_dir, '*.zip'))[0]
    with ZipFile(zf, 'r') as zipObj:
        zipObj.extractall(deepcell_output_dir)


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
