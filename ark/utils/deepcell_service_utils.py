from twisted.internet import reactor
from kiosk_client import manager, utils
import os
import sys
import glob
import zipfile


def run_deepcell_task(input_dir, output_dir, point, host='https://deepcell.org', job_type='multiplex'):
    """Uses kiosk-client to run DeepCell task and extracts zip to output folder.
        (https://github.com/vanvalenlab/kiosk-client)

        Args:
            input_path: path to .tif image
            output_path: location to save deepcell output (as .tif)
            host: Hostname and port for the kiosk-frontend API server
            job_type: Name of job workflow (multiplex, segmentation, tracking)
        Output:
            Writes DeepCell service .zip output to /downloads folder and extract the .tif file to output_path

        """

    # more configuration parameters can be set. https://github.com/vanvalenlab/kiosk-client
    mgr_kwargs = {
        'host': host,
        'job_type': job_type,
        'download_results': True
    }

    # will only work for the current project structure
    input_path = input_dir[3:] if input_dir.startswith('../') else input_dir

    # build absolute path
    filename = os.path.join(os.path.dirname(__file__), os.path.join(sys.path[1], input_path) + point + '.tif.zip')

    mgr = manager.BatchProcessingJobManager(**mgr_kwargs)
    mgr.run(filepath=filename)
    reactor.run()

    # extract .zip file from /downloads to output_path

    dw_path = os.path.join(utils.get_download_path(), '*.zip')
    list_of_files = glob.glob(dw_path)

    # TODO get the relevant filename instead of simply taking the last
    last_modified_file = max(list_of_files, key=os.path.getctime)

    with zipfile.ZipFile(last_modified_file, "r") as z:
        z.extractall(output_dir)

