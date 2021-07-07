import os
import json
import io
import shutil
import warnings

import pandas as pd

import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet, InvalidToken

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

import mimetypes

_SCOPES = ['https://www.googleapis.com/auth/drive']

_FOLDER_MIME_CHECK = "mimeType = 'application/vnd.google-apps.folder'"
_FILE_MIME_CHECK = "mimeType != 'application/vnd.google-apps.folder'"
_SHORTCUT_MIME_CHECK = "mimeType = 'application/vnd.google-apps.shortcut'"

_FOLDER_MIME = "application/vnd.google-apps.folder"

SERVICE = None


def _gen_enckey(pw):
    pw = pw.encode()
    with open('/home/.toks/.s.txt', 'rb') as f:
        s = f.read()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=s,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(pw))


def _decrypt_cred_data(data, pw):
    fernet = Fernet(_gen_enckey(pw))
    data_out = None

    try:
        data_out = fernet.decrypt(data)
    except InvalidToken:
        raise ValueError("Invalid Key - Could not decrypt credentials...")

    if data_out is not None:
        data_out = json.loads(data_out)

    return data_out


def _get_creds(auth_pw):
    with open('/home/.toks/.creds.enc', 'rb') as f:
        data = f.read()

    # decrypt via user provided key
    client_config = _decrypt_cred_data(data, auth_pw)
    if client_config is None:
        return None

    # generate oauth2 session
    flow = InstalledAppFlow.from_client_config(client_config, _SCOPES)
    creds = flow.run_local_server(port=0)

    return creds


def init_google_drive_api(auth_pw):
    """Initializes the google drive api service

    Args:
        auth_pw (str): Encryption pw used to generate key for google drive api usage
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('/home/.toks/.token.json'):
        creds = Credentials.from_authorized_user_file('/home/.toks/.token.json', _SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                creds = _get_creds(auth_pw)
                if creds is None:
                    return
        else:
            creds = _get_creds(auth_pw)
            if creds is None:
                return

        # Save the credentials for the next run
        with open('/home/.toks/.token.json', 'w') as token:
            token.write(creds.to_json())

    global SERVICE
    SERVICE = build('drive', 'v3', credentials=creds)


def _validate(path_string):
    global SERVICE
    if path_string[0] != '/':
        raise ValueError('Invalid path provided.  Please use the format: /path/to/folder')

    parents = path_string.split('/')[1:-1]
    ids = ['root']

    # validate parent existence
    for i, parent in enumerate(parents):
        if parent == '':
            continue
        response = SERVICE.files().list(
            q=f"((('{ids[-1]}' in parents) and (name = '{parent}')) " +
              f"and (({_FOLDER_MIME_CHECK}) or ({_SHORTCUT_MIME_CHECK})))",
            spaces='drive',
            fields='files(id, shortcutDetails(targetId))',
        ).execute()

        files = response.get('files', [])
        if len(files) == 0:
            raise FileNotFoundError(f'Could not find the folder {parent} in parent folder ' +
                                    f'{parents[i - 1]}...')

        # if shortcut, get target id
        if files[0].get('shortcutDetails', None) is not None:
            ids.append(files[0].get('shortcutDetails').get('targetId'))
        else:
            ids.append(files[0].get('id'))

    # validate file existence
    response = SERVICE.files().list(
        q=f"(('{ids[-1]}' in parents) and (name = '{path_string.split('/')[-1]}'))",
        spaces='drive',
        fields='files(id, shortcutDetails(targetId))'
    ).execute()
    files = response.get('files', [])

    # transform ids into map
    ids = dict(zip(['root'] + parents, ids))

    # return appropriate result
    if len(files) == 0:
        return ids, None

    # if shortcut, get target id
    if files[0].get('shortcutDetails', None) is not None:
        return ids, files[0].get('shortcutDetails').get('targetId')
    else:
        return ids, files[0].get('id')


class GoogleDrivePath(object):
    def __init__(self, path_string):
        self.bad_service = False
        global SERVICE
        if self._service_check():
            warnings.warn(
                """Please call `init_google_drive_api` with the appropriate password, in
                   order to use GoogleDrivePath objects...""",
                UserWarning
            )
            self.bad_service = True
            self.path_string = None
            self.parent_id_map = {}
            self.fileID = None
            return super().__init__()

        self.path_string = path_string
        self.parent_id_map, self.fileID = _validate(path_string)
        return super().__init__()

    def __truediv__(self, more_path):
        new_path_string = self.path_string
        if more_path != "":
            new_path_string = self.path_string + '/' + more_path

        return GoogleDrivePath(new_path_string)

    def _service_check(self):
        global SERVICE
        return SERVICE is None or self.bad_service

    def filename(self):
        """ Get filename given in path_string
        """
        return self.path_string.split('/')[-1]

    def path_parents(self):
        """ Get list of parent directories listed in path_string
        """
        return self.path_string.split('/')[1:-1]

    def get_name_and_data(self):
        """ Get name and media from Drive object

        Returns:
            tuple:
                name and media (media is None if mimeType is a folder)
        """
        global SERVICE
        response = SERVICE.files().get(fileId=self.fileID).execute()

        return (
            response.get('name'),
            None if response.get('mimeType') == _FOLDER_MIME else self.read()
        )

    def mkdir(self):
        """ Creates specified directory on Drive

        Must have valid parents, and filename must not contain a file extension

        Returns:
            bool:
                If a folder was able to be created.
        """

        if self.parent_id_map is not None and self.fileID is None and '.' not in self.filename():
            global SERVICE
            folder_metadata = {
                'name': self.filename(),
                'mimeType': _FOLDER_MIME,
                'parents': [self.parent_id_map[(['root'] + self.path_parents())[-1]]]
            }
            response = SERVICE.files().create(body=folder_metadata, fields='id').execute()
            self.fileID = response.get('id')
            return True

        return False

    def clone(self, dest, overwrite=False, clear_dest=False):
        """ Clones directory structure into provided destination

        For a given path_string, e.g '/root/folderA/folderB', all files and subdirectories of
        'folderB' are cloned into the destination.

        Args:
            dest (str):
                A path location to contain cloned directory structure.  Must be a directory.
            overwrite (bool):
                If files are already present in dest and overwrite is False, no cloning is
                performed.  Otherwise, the directory is overwriten with the cloned data.
            clear_dest (bool):
                If clear_dest is true, all contents of dest are removed before cloning. Default is
                False.
        """
        if self._service_check() or self.fileID is None:
            raise FileNotFoundError(f"The path '{self.path_string}' does not exist, so it cannot" +
                                    " be cloned")

        if clear_dest and os.path.exists(dest):
            shutil.rmtree(dest)

        if not os.path.exists(dest):
            new_foldername = os.path.basename(dest)
            dest = os.path.abspath(os.path.join(dest, os.pardir))
            if not os.path.exists(dest):
                raise FileNotFoundError(f"Could not resolve the path {dest}")

            dest = os.path.join(dest, new_foldername)
            os.mkdir(dest)

        elif not overwrite:
            raise FileExistsError(
                f"The path '{dest}' already exists. If you wish to overwrite/update files"
                + " here, please pass the argument `overwrite=True`"
            )

        fname, fdata = self.get_name_and_data()
        if fdata is None:
            for filename in self.lsfiles():
                filepath = self / filename
                filepath.clone(dest, overwrite=True)
            for dirname in self.lsdirs():
                dirpath = self / dirname
                dirpath.clone(os.path.join(dest, dirname), overwrite=overwrite)
        else:
            with open(os.path.join(dest, fname), 'wb') as f:
                f.write(fdata.read())

        return

    def upload(self, src, overwrite=False):
        """ Uploads contents and file structure of local directory to Drive.

        For a given path_string, e.g '/root/folderA/folderB', the provided source file, as well as
        all of its contents, will be added into 'folderB'.

        Args:
            src (str):
                A path location to copy a directory structure/content from.  Doesn't need to be a
                directory.
            overwrite (bool):
                If similarly named content exists on Drive, overwrite is False, no upload will take
                place.  Otherwise, the content on the Drive folder is updated/overwritten.
        """
        if (
            self._service_check()
            or set(list(self.parent_id_map.keys())[1:]) != set(self.path_parents())
        ):
            raise FileNotFoundError(f"The path {self.path_string} contains folders which do not " +
                                    "exist...")

        if not os.path.exists(src):
            raise FileNotFoundError(f"The path '{src}' does not exist...")

        if self.fileID is None:
            if not self.mkdir():
                self.write(src)
                return
        elif not overwrite:
            raise FileExistsError(
                f"The Drive path '{self.path_string}' already exists. If you wish to"
                + " overwrite/update files, please pass the argument `overwrite=True`"
            )

        if not os.path.isdir(src):
            print(f"\x1b[1K\rUploading {src} to Drive...", end='')
            (self / os.path.basename(src)).write(src, overwrite=overwrite)
        else:
            for filename in os.listdir(src):
                if not os.path.isdir(os.path.join(src, filename)):
                    print(
                        f"\x1b[1K\rUploading {os.path.join(src, filename)} to Drive...",
                        end=''
                    )
                    (self / filename).write(os.path.join(src, filename), overwrite=True)
                else:
                    (self / filename).upload(os.path.join(src, filename), overwrite=overwrite)

        return

    def read(self):
        """ Returns given file data if path leads to a file

        Returns:
            BytesIO: file data bytes.
        """
        if self._service_check() or self.fileID is None:
            raise FileNotFoundError(f"The path '{self.path_string}' does not exist, so it cannot" +
                                    " be read")

        global SERVICE
        request = SERVICE.files().get_media(fileId=self.fileID)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(
                f'\x1b[1K\rDownloading {self.path_string} - {int(status.progress() * 100)}% ...',
                end=''
            )
        fh.seek(0)
        return fh

    def read_csv(self, **kwargs):
        """ Reads csv from Drive and returns a pandas dataframe

        Args:
            **kwargs (dict): Arguments passed to `pandas.read_csv()` function.

        Returns:
            DataFrame: pandas dataframe from read csv
        """
        if self.path_string.split('/')[-1].endswith('.csv'):
            data = self.read()
            if data is not None:
                return pd.read_csv(data, **kwargs)
        else:
            raise ValueError(f"{self.path_string} is not a csv file...")

    def write(self, data, overwrite=False):
        """ Writes file data or data from file path to provided Drive file.

        Args:
            data (str or BytesIO):
                If BytesIO, data is directly uploaded.  If str, data is assumed to be a filepath
                and data is attempted to be read from the local file.
            overwrite (bool):
                If a file already exists at the path_string and overwrite is False, no data is
                written.  Otherwise, the existing file on Drive is updated/overwritten.
        """
        if (
            self._service_check()
            or set(list(self.parent_id_map.keys())[1:]) != set(self.path_parents())
        ):
            raise FileNotFoundError(f"The path {self.path_string} contains folders which do not " +
                                    "exist...")

        if self.fileID is not None and not overwrite:
            raise FileExistsError(
                f"The Drive path '{self.path_string}' already exists. If you wish to"
                + " overwrite/update files, please pass the argument `overwrite=True`"
            )

        mtype = mimetypes.types_map.get(
            '.' + self.filename().split('.')[-1],
            'application/octet-stream'
        )
        if type(data) is str:
            # warn if extension causes mismatch
            data_mtype = mimetypes.types_map.get(
                '.' + data.split('.')[-1],
                'application/octet-stream'
            )
            if data_mtype != mtype:
                warnings.warn(
                    f'Warning: local MimeType {data_mtype} does not match with Drive file of '
                    + f'MimeType: {mtype} ...',
                    UserWarning
                )

        global SERVICE
        if self.fileID is None:
            # create upload structure
            if type(data) is str:
                media = MediaFileUpload(data, mimetype=mtype, resumable=True)
            else:
                media = MediaIoBaseUpload(data, mimetype=mtype, resumable=True)

            file_metadata = {
                'name': self.filename(),
                'parents': [self.parent_id_map[self.path_parents()[-1]]]
            }

            response = SERVICE.files().create(body=file_metadata,
                                              media_body=media,
                                              fields='id').execute()

            self.fileID = response.get('id')
        else:
            response = SERVICE.files().get(fileId=self.fileID).execute()
            if response.get('mimeType') == _FOLDER_MIME:
                raise IsADirectoryError(
                    f'This path {self.path_string} points to a folder...\n'
                    + 'Please create a new path to a new file, for example:\n'
                    + '    new_filepath = this_filepath / "example.txt"'
                )

            if response.get('mimeType') != mtype:
                warnings.warn(
                    f"Warning: extension inferred mimeType '{mtype}' doesn't match Drive's "
                    + f"'{response.get('mimeType')}'",
                    UserWarning
                )

            if type(data) is str:
                media = MediaFileUpload(data, mimetype=mtype, resumable=True)
            else:
                media = MediaIoBaseUpload(data, mimetype=mtype, resumable=True)

            response = SERVICE.files().update(fileId=self.fileID,
                                              media_body=media,
                                              media_mime_type=mtype).execute()
        return

    def lsfiles(self):
        """ List all non-directory files at path_string in Drive
        """
        if self._service_check() or self.fileID is None:
            return None

        filenames = []

        global SERVICE
        page_token = None
        while True:
            response = SERVICE.files().list(
                q=f"(('{self.fileID}' in parents) and ({_FILE_MIME_CHECK}))",
                spaces='drive',
                fields='nextPageToken, files(name, mimeType, shortcutDetails(targetMimeType))',
                pageToken=page_token).execute()

            for file in response.get('files', []):
                if file.get('shortcutDetails', None) is not None:
                    mimeType = file.get('shortcutDetails').get('targetMimeType')
                    if mimeType != _FOLDER_MIME:
                        filenames.append(file.get('name'))
                else:
                    filenames.append(file.get('name'))

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        return filenames

    def lsdirs(self):
        """ List all directories at path_string in Drive
        """
        if self._service_check() or self.fileID is None:
            return None

        dirnames = []

        global SERVICE
        page_token = None
        while True:
            response = SERVICE.files().list(
                q=f"(('{self.fileID}' in parents) " +
                  f"and (({_FOLDER_MIME_CHECK}) or ({_SHORTCUT_MIME_CHECK})))",
                spaces='drive',
                fields='nextPageToken, files(name, shortcutDetails(targetMimeType))',
                pageToken=page_token).execute()

            for file in response.get('files', []):
                if file.get('shortcutDetails', None) is not None:
                    mimeType = file.get('shortcutDetails').get('targetMimeType')
                    if mimeType == _FOLDER_MIME:
                        dirnames.append(file.get('name'))

                dirnames.append(file.get('name'))

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        return dirnames


def path_join(*path_parts, get_filehandle=False):
    """ Generalization of os.path.join for GoogleDrivePaths and strings

    Args:
        path_parts (tuple):
            Tuple of GoogleDrivePath+strings or strings
        get_filehandle (bool):
            If true and path_parts contains a GoogleDrivePath, file handles are returned instead
            of filepaths/GoogleDrivePath

    Returns:
        str or GoogleDrivePath or BytesIO:
            Filepath, GoogleDrivePath, or the filehandle
    """
    google_drive_path = type(path_parts[0]) is GoogleDrivePath

    if not google_drive_path:
        return os.path.join(path_parts[0], *path_parts[1:])

    path_parts_filt = [
        pp
        for pp in path_parts[1:]
        if pp != ""
    ]

    path_out = path_parts[0] / '/'.join(path_parts_filt)
    if get_filehandle:
        return path_out.read()

    return path_out
