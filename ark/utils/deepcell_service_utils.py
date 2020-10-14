from ark.utils import io_utils
from twisted.internet import reactor
from kiosk_client import manager
import os
import glob
from zipfile import ZipFile
import warnings


def create_deepcell_output(deepcell_input_dir, deepcell_output_dir, fovs=None,
                           suffix='_feature_0', host='https://deepcell.org', job_type='multiplex'):
    """ Handles all of the necessary data manipulation for running deepcell tasks.

        Creates .zip files (to be used as input for DeepCell),
        calls run_deepcell_task method,
        and extracts zipped output files to the specified output location

        Args:
            deepcell_input_dir (str):
                Location of preprocessed files (assume deepcell_input_dir contains <fov>.tif
                for each fov in fovs list)
            deepcell_output_dir (str):
                Location to save DeepCell output (as .tif)
            fovs (list):
                List of fovs in preprocessing pipeline. if None, all .tif files
                in deepcell_input_dir will be considered as input fovs. Default: None
            suffix (str):
                Suffix for DeepCell output filename. e.g. for fovX, DeepCell output
                should be <fovX>+suffix.tif. Default: '_feature_0'
            host (str):
                Hostname and port for the kiosk-frontend API server
                Default: 'https://deepcell.org'
            job_type (str):
                Name of job workflow (multiplex, segmentation, tracking)
                Default: 'multiplex'
        Raises:
            ValueError:
                Raised if there is some fov X (from fovs list) s.t.
                the file <deepcell_input_dir>/fovX.tif does not exist
        """
    if fovs is None:
        tifs = io_utils.list_files(deepcell_input_dir, substrs='.tif')
        fovs = io_utils.extract_delimited_names(tifs, delimiter='.')

    zip_path = os.path.join(deepcell_input_dir, 'fovs.zip')
    if os.path.isfile(zip_path):
        warnings.warn(f'{zip_path} will be overwritten.')

    with ZipFile(zip_path, 'w') as zipObj:
        for fov in fovs:
            filename = os.path.join(deepcell_input_dir, fov + '.tif')
            if not os.path.isfile(filename):
                raise ValueError('Could not find .tif file for %s. '
                                 'Invalid value for %s' % (fov, filename))
            zipObj.write(filename, os.path.basename(filename))

    run_deepcell_task(zip_path, deepcell_output_dir, host, job_type)
    os.remove(zip_path)

    # extract the .tif output
    zip_files = glob.glob(os.path.join(deepcell_output_dir, '*.zip'))
    zip_files.sort(key=os.path.getmtime)
    with ZipFile(zip_files[-1], 'r') as zipObj:
        zipObj.extractall(deepcell_output_dir)
        for fov in fovs:
            if fov + suffix + '.tif' not in zipObj.namelist():
                warnings.warn(f'Deep Cell output file was not found for {fov}.')


def run_deepcell_task(input_dir, output_dir, host='https://deepcell.org',
                      job_type='multiplex'):
    """Uses kiosk-client to run DeepCell task and saves output to output_dir.
        More configuration parameters can be set than those currently used.
        (https://github.com/vanvalenlab/kiosk-client)

        Args:
            input_dir (str):
                location of .zip files
            output_dir (str):
                location to save deepcell output (as .zip)
            host (str):
                Hostname and port for the kiosk-frontend API server.
                Default: 'https://deepcell.org'
            job_type (str):
                Name of job workflow (multiplex, segmentation, tracking).
                Default: 'multiplex'
        """

    mgr_kwargs = {
        'host': host,
        'job_type': job_type,
        'download_results': True,
        'output_dir': output_dir
    }

    mgr = manager.BatchProcessingJobManager(**mgr_kwargs)
    mgr.run(filepath=input_dir)
    reactor.run()
