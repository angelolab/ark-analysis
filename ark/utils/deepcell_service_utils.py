from twisted.internet import reactor
from kiosk_client import manager
import os
import glob
from zipfile import ZipFile
import warnings


def create_deepcell_output(deepcell_input_dir, points, deepcell_output_dir,
                           suffix='_feature_0', host='https://deepcell.org', job_type='multiplex'):
    """ Handles all of the necessary data manipulation for running deepcell tasks.

        Creates .zip files (to be used as input for DeepCell),
        calls run_deepcell_task method,
        and extracts zipped output files to the specified output location

        Args:
            points (list):              List of points in preprocessing pipeline

            deepcell_input_dir (str):   Location of preprocessed files
                                        (assume deepcell_input_dir contains <point>.tif
                                        for each point in points list)

            suffix (str):               Suffix for DeepCell output filename. e.g. for pointX,
                                        DeepCell output should be <pointX>+suffix.tif.
                                        Default: '_feature_0'

            deepcell_output_dir (str):  Location to save DeepCell output (as .tif)

            host (str):                 Hostname and port for the kiosk-frontend API server
                                        Default: 'https://deepcell.org'

            job_type (str):             Name of job workflow (multiplex, segmentation, tracking)
                                        Default: 'multiplex'

        Output:
            Writes DeepCell service .tif output to output_dir

        Raises:
            ValueError:                 Raised if there is some point X (from points list) s.t.
                                        the file <deepcell_input_dir>/PointX.tif does not exist

        """

    zip_path = os.path.join(deepcell_input_dir, 'points.zip')
    if os.path.isfile(zip_path):
        warnings.warn(f'{zip_path} will be overwritten')

    with ZipFile(zip_path, 'w') as zipObj:
        for point in points:
            filename = os.path.join(deepcell_input_dir, point + '.tif')
            if not os.path.isfile(filename):
                raise ValueError('Could not find .tif file for %s. '
                                 'Invalid value for %s' % (point, filename))
            zipObj.write(filename, os.path.basename(filename))

    run_deepcell_task(zip_path, deepcell_output_dir, host, job_type)
    os.remove(zip_path)

    # extract the .tif output
    zip_files = glob.glob(os.path.join(deepcell_output_dir, '*.zip'))
    zip_files.sort(key=os.path.getmtime)
    with ZipFile(zip_files[-1], 'r') as zipObj:
        zipObj.extractall(deepcell_output_dir)
        for point in points:
            if point + suffix + '.tif' not in zipObj.namelist():
                warnings.warn(f'Deep Cell output file was not found for {point}.')


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
