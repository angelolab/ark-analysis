from ark.utils import io_utils
import requests
from pathlib import Path
import time
from tqdm.notebook import tqdm
from urllib.parse import unquote_plus
import os
from zipfile import ZipFile, ZIP_DEFLATED
import warnings
from concurrent.futures import ThreadPoolExecutor

from ark.utils import misc_utils


def create_deepcell_output(deepcell_input_dir, deepcell_output_dir, fovs=None,
                           suffix='_feature_0', host='https://deepcell.org', job_type='mesmer',
                           scale=1.0, timeout=3600, zip_size=10, parallel=False):
    """Handles all of the necessary data manipulation for running deepcell tasks.
    Creates .zip files (to be used as input for DeepCell),
    calls run_deepcell_task method,
    and extracts zipped output files to the specified output location

    Args:
        deepcell_input_dir (str):
            Location of preprocessed files (assume deepcell_input_dir contains <fov>.tif
            for each fov in fovs list).  This should not be a GoogleDrivePath.
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
        scale (float):
            Value to rescale data by
            Default: 1.0
        timeout (int):
            Approximate seconds until timeout.
            Default: 1 hour (3600)
        zip_size (int):
            Maximum number of files to include in zip.
            Default: 100
        parallel (bool):
            Tries to zip, upload, and extract zip files in parallel
            Default: False
    Raises:
        ValueError:
            Raised if there is some fov X (from fovs list) s.t.
            the file <deepcell_input_dir>/fovX.tif does not exist
    """

    # check that scale arg can be converted to a float
    try:
        scale = float(scale)
    except ValueError:
        raise ValueError("Scale argument must be a number")

    # extract all the files from deepcell_input_dir
    input_files = io_utils.list_files(deepcell_input_dir, substrs=['.tif'])

    # set fovs equal to input_files it not already set
    if fovs is None:
        fovs = input_files

    # now extract only the names of the fovs without the file extension
    fovs = io_utils.remove_file_extensions(fovs)

    # make sure that all fovs actually exist in the list of input_files
    misc_utils.verify_in_list(
        fovs=fovs,
        deepcell_input_files=io_utils.remove_file_extensions(input_files))

    # partition fovs for smaller zip file batching
    fov_groups = [
        fovs[zip_size * i:zip_size * (i + 1)]
        for i in range(((len(fovs) + zip_size - 1) // zip_size))
    ]

    print(f'Processing tiffs in {len(fov_groups)} batches...')

    # yes this is function, don't worry about it
    # long story short, too many args to pass if function not in local scope
    # i.e easier to map fov_groups
    def _zip_run_extract(fov_group, group_index):
        # define the location of the zip file for our fovs
        zip_path = os.path.join(deepcell_input_dir, f'fovs_batch_{group_index + 1}.zip')

        if os.path.isfile(zip_path):
            warnings.warn(f'{zip_path} will be overwritten')

        # write all files to the zip file
        print('Zipping preprocessed tif files.')

        def zip_write(zip_path):
            with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zipObj:
                for fov in fov_group:
                    # file has .tif extension
                    basename = fov + '.tif'
                    if basename not in input_files:
                        basename = basename + 'f'

                    filename = os.path.join(deepcell_input_dir, basename)
                    zipObj.write(filename, basename)

        zip_write(zip_path)

        # pass the zip file to deepcell.org
        print('Uploading files to DeepCell server.')
        run_deepcell_direct(
            zip_path, deepcell_output_dir, host, job_type, scale, timeout
        )

        # extract the .tif output
        print("Extracting tif files from DeepCell response.")
        zip_names = io_utils.list_files(deepcell_output_dir, substrs=[".zip"])

        zip_files = [os.path.join(deepcell_output_dir, name) for name in zip_names]

        # sort by newest added
        zip_files.sort(key=os.path.getmtime)

        with ZipFile(zip_files[-1], "r") as zipObj:
            for name in zipObj.namelist():
                with open(os.path.join(deepcell_output_dir, name), mode='wb') as f:
                    f.write(zipObj.read(name))
            for fov in fov_group:
                if fov + suffix + '.tif' not in zipObj.namelist():
                    warnings.warn(f'Deep Cell output file was not found for {fov}.')

    # make calls in parallel
    if parallel:
        with ThreadPoolExecutor() as executor:
            executor.map(_zip_run_extract, fov_groups, range(len(fov_groups)))
            executor.shutdown(wait=True)
    else:
        list(map(_zip_run_extract, fov_groups, range(len(fov_groups))))


def run_deepcell_direct(input_dir, output_dir, host='https://deepcell.org',
                        job_type='mesmer', scale=1.0, timeout=3600):
    """Uses direct calls to DeepCell API and saves output to output_dir.

    Args:
        input_dir (str):
            location of .zip files
        output_dir (str):
            location to save deepcell output (as .zip)
        host (str):
            Hostname and port for the kiosk-frontend API server.
            Default: 'https://deepcell.org'
        job_type (str):
            Name of job workflow (mesmer, segmentation, tracking).
        scale (float):
            Value to rescale data by
            Default: 1.0
        timeout (int):
            Approximate seconds until timeout.
            Default: 1 hour (3600)
    """

    # upload zip file
    upload_url = host + "/api/upload"
    filename = Path(input_dir).name

    with open(input_dir, mode='rb') as f:
        upload_fields = {
            'file': (filename, f.read(), 'application/zip'),
        }
        f.seek(0)

    upload_response = requests.post(
        upload_url,
        timeout=timeout,
        files=upload_fields
    ).json()

    # call prediction
    predict_url = host + '/api/predict'

    predict_response = requests.post(
        predict_url,
        json={
            'dataRescale': scale,
            'imageName': filename,
            'imageUrl': upload_response['imageURL'],
            'jobType': job_type,
            'uploadedName': upload_response['uploadedName']
        }
    ).json()

    predict_hash = predict_response['hash']

    # check redis every 3 seconds
    redis_url = host + '/api/redis'

    print('Segmentation progress:')
    progress_bar = tqdm(total=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    pbar_last = 0
    total_time = 0
    redis_response = None

    while total_time < timeout:
        redis_response = requests.post(
            redis_url,
            json={
                'hash': predict_hash,
                'key': ["status", "progress", "output_url", "reason", "failures"]
            }
        ).json()
        if redis_response['value'][0] == 'done':
            break

        # update progress bar here
        if redis_response['value'][0] == 'waiting':
            pbar_next = int(redis_response['value'][1])
            progress_bar.update(max(pbar_next - pbar_last, 0))
            pbar_last = pbar_next

        if redis_response['value'][0] not in ['done', 'waiting', 'new']:
            print(redis_response['value'])

        time.sleep(3.0)
        total_time += 3
    progress_bar.close()

    # when done, download result or examine errors
    if len(redis_response['value'][4]) > 0:
        # error happened
        print(f"Encountered Failure(s): {unquote_plus(redis_response['value'][4])}")

    deepcell_output = requests.get(redis_response['value'][2], allow_redirects=True)

    with open(os.path.join(output_dir, "deepcell_response.zip"), mode="wb") as f:
        f.write(deepcell_output.content)

    # being kind and sending an expire signal to deepcell
    expire_url = redis_url + '/expire'
    requests.post(
        expire_url,
        json={
            'hash': predict_hash,
            'expireIn': 3600,
        }
    )

    return
