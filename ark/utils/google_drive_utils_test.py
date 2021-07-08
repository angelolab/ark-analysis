from copy import deepcopy
import os
import re
import mimetypes
import io
import functools
from collections import namedtuple

import tempfile

from ark.utils import google_drive_utils

from pytest_mock import MockerFixture


def _op_in(qt, v):
    def check_parent(path):
        return qt.replace("'", "") == os.path.dirname(path)

    return check_parent


def _op_eq(qt, v):

    def check_name(path):
        return v.replace("'", "") == os.path.basename(path)

    def check_mimetype(path):
        if v.replace("'", "") == google_drive_utils._FOLDER_MIME:
            return os.path.isdir(path)
        elif v == google_drive_utils._SHORTCUT_MIME_CHECK.split(' ')[-1]:
            return False
        else:
            return True

    return check_name if qt == 'name' else check_mimetype


def _op_neq(qt, v):
    def check_name(path):
        return v.replace("'", "") != os.path.basename(path)

    def check_mimetype(path):
        if v.replace("'", "") == google_drive_utils._FOLDER_MIME:
            return not os.path.isdir(path)
        elif v == google_drive_utils._SHORTCUT_MIME_CHECK.split(' ')[-1]:
            return True
        else:
            return False

    return check_name if qt == 'name' else check_mimetype


def _op_and(cond1, cond2):
    def check_and(path):
        return cond1(path) and cond2(path)

    return check_and


def _op_or(cond1, cond2):
    def check_or(path):
        return cond1(path) or cond2(path)

    return check_or


OPERATOR_KEY = {
    'in': _op_in,
    '=': _op_eq,
    '!=': _op_neq,
    'and': _op_and,
    'or': _op_or,
}


def _parse_base_query(bq, funcs=None, base_dir=None):
    query_term, operator, values = tuple(bq.split(' '))

    if query_term[0] == '$' and values[0] == '$':
        return OPERATOR_KEY[operator](funcs[query_term], funcs[values])
    else:
        if query_term == "'root'":
            query_term = base_dir

        return OPERATOR_KEY[operator](query_term, values)


def _parse_full_query(q: str, funcs=None, base_dir=None, var_count=0):
    base_queries = re.findall('\\([^\\(\\)]*\\)', q)

    # trim parentheses
    base_queries_trimmed = [bq[1:-1] for bq in base_queries]

    results = dict(zip(
        ['$' + str(i + var_count) for i in range(len(base_queries))],
        [_parse_base_query(bq, funcs, base_dir=base_dir) for bq in base_queries_trimmed]
    ))

    if funcs is not None:
        results.update(funcs)

    q_reduced = q
    for i, bq in enumerate(base_queries):
        q_reduced = q_reduced.replace(bq, '$' + str(i + var_count))

    q_reduced_split = [term for term in q_reduced.split(' ') if term != '']

    if len(q_reduced_split) == 1:
        return results['$' + str(i + var_count)]
    else:
        return _parse_full_query(q_reduced, funcs=results, base_dir=base_dir,
                                 var_count=len(base_queries) + var_count)


def _parse_base_field(bf, dicts=None):
    name = bf[0]
    new_keys = [nk.replace(' ', '') for nk in bf[1].split(',')]

    out_dict = {name: {}}
    for nk in new_keys:
        if nk[0] == '$':
            out_dict[name].update(dicts[nk])
        else:
            out_dict[name][nk] = ''

    return out_dict


def _parse_fields(fields: str, dicts=None):
    base_fields = re.findall('([^\\(\\)\\s]+)\\(([^\\(\\)]*)\\)', fields)

    next_dicts = dict(zip(
        ['$' + str(i) for i in range(len(base_fields))],
        [_parse_base_field(bf, dicts) for bf in base_fields]
    ))

    fields_reduced = fields
    for i, bf in enumerate(base_fields):
        fields_reduced = fields_reduced.replace(f'{bf[0]}({bf[1]})', '$' + str(i))

    if fields_reduced[0] == '$':
        return next_dicts[fields_reduced]
    else:
        return _parse_fields(fields_reduced, dicts=next_dicts)


def _fill_fields(path: str, _fields: dict):
    # always fix next page token to none
    fields = deepcopy(_fields)
    fields['nextPageToken'] = None

    files = fields.get('files', None)
    if files is not None:
        if files.get('id', None) is not None:
            files['id'] = path
        if files.get('name', None) is not None:
            files['name'] = os.path.basename(path)
        if files.get('mimeType', None) is not None:
            if os.path.isdir(path):
                files['mimeType'] = google_drive_utils._FOLDER_MIME
            else:
                files['mimeType'] = mimetypes.types_map.get(
                    os.path.splitext(path)[1],
                    'application/octet-stream'
                )
        # never need to worry about shortcuts
        files['shortcutDetails'] = None

    return fields


class _MockedExecute:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class _MockedFiles:
    def __init__(self, mock_drive_dir):
        self.mock_drive_dir = mock_drive_dir
        return

    def list(self, q, spaces, fields, pageToken=None):
        assert(spaces == 'drive')

        query_function = _parse_full_query(q, funcs=None, base_dir=self.mock_drive_dir)

        matches = []
        for root, dirs, files in os.walk(self.mock_drive_dir):
            for name in files:
                if query_function(os.path.join(root, name)):
                    matches.append(os.path.join(root, name))
            for name in dirs:
                if query_function(os.path.join(root, name)):
                    matches.append(os.path.join(root, name))

        fields = fields.split('nextPageToken, ')[-1]
        parsed_fields = _parse_fields(fields)

        filled_fields = [_fill_fields(match, parsed_fields) for match in matches]

        response = {
            'nextPageToken': None,
            'files': [ff['files'] for ff in filled_fields],
        }

        return _MockedExecute(response)

    def create(self, body, media_body=None, fields=None):
        assert(fields == 'id' or fields is None)

        path = ''
        for parent in body.get('parents', []):
            if parent == 'root':
                parent = self.mock_drive_dir
            path = os.path.join(parent, body.get('name', 'default_name'))
            if media_body is None:
                os.mkdir(path)
            else:
                with open(path, mode='wb') as f:
                    f.write(media_body.read())
            break

        response = {}
        if fields == 'id':
            response['id'] = path

        return _MockedExecute(response)

    def get(self, fileId):
        return _MockedExecute({
            'name': os.path.basename(fileId),
            'mimeType':
                google_drive_utils._FOLDER_MIME if os.path.isdir(fileId) else (
                    mimetypes.types_map.get(
                        os.path.splitext(fileId)[1],
                        'application/octet-stream'
                    )
                )
        })

    def get_media(self, fileId):
        return fileId

    def update(self, fileId, media_body, media_mime_type):
        with open(fileId, 'wb') as f:
            f.write(media_body.read())
        return _MockedExecute(None)


class _MockedService:
    def __init__(self, mock_drive_dir):
        self._files = _MockedFiles(mock_drive_dir)

    def files(self):
        return self._files


def _mocked_init(auth_pw, mock_drive_dir):
    google_drive_utils.SERVICE = _MockedService(mock_drive_dir)


class _MockUploadFile:
    def __init__(self, data, mimetype, resumable):
        with open(data, mode='rb') as f:
            self.fh = io.BytesIO(f.read())
        self.fh.seek(0)

    def read(self):
        return self.fh.read()


class _MockUploadBytes():
    def __init__(self, data, mimetype, resumable):
        self.fh = io.BytesIO(data.read())
        self.fh.seek(0)

    def read(self):
        return self.fh.read()


class _MockDownload:
    def __init__(self, fh, request):
        self.path = request
        self.fh = fh

    def next_chunk(self):
        with open(self.path, mode='rb') as f:
            self.fh.write(f.read())

        return namedtuple('status', 'progress')(progress=lambda: 1), True


def _offline_mocker(mocker: MockerFixture, mock_drive_dir):
    mocker.patch(
        'ark.utils.google_drive_utils.init_google_drive_api',
        lambda x: _mocked_init(x, mock_drive_dir)
    )

    mocker.patch.multiple(
        'googleapiclient.http.MediaFileUpload',
        create=True,
        __init__=_MockUploadFile.__init__,
        read=_MockUploadFile.read
    )

    mocker.patch.multiple(
        'googleapiclient.http.MediaIoBaseUpload',
        create=True,
        __init__=_MockUploadBytes.__init__,
        read=_MockUploadBytes.read
    )

    mocker.patch.multiple(
        'googleapiclient.http.MediaIoBaseDownload',
        __init__=_MockDownload.__init__,
        next_chunk=_MockDownload.next_chunk
    )


def _fill_gdrive(gdrive, dir_structure=None):
    if dir_structure is None:
        dir_structure = {
            'folderA': {
                'fileA.txt': 'folderA/fileA.txt content!',
                'fileB.txt': 'folderA/fileB.txt content!',
            },
            'folderB': {
                'fileA.txt': 'folderB/fileA.txt content!',
                'fileB.txt': 'folderB/fileB.txt content!',
                'fileC.txt': 'folderB/fileC.txt content!',
            },
            'fileA.txt': '/fileA.txt content!',
            'data_folder': {
                'example.csv':
                ''.join([w for w in """ name, age,
                                        bill, 21,
                                        tom, 24,
                                        thom, 33,
                                        beuregard, 93,""".split(' ') if w != ''])
            }
        }

    for dirc in dir_structure.keys():
        if type(dir_structure[dirc]) is dict:
            _fill_gdrive(os.path.join(gdrive, dirc), dir_structure[dirc])
        else:
            os.makedirs(gdrive, exist_ok=True)
            with open(os.path.join(gdrive, dirc), mode='w') as f:
                f.write(dir_structure[dirc])


def local_gdrive(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as gdrive:
            _fill_gdrive(gdrive)
            _offline_mocker(kwargs['mocker'], gdrive)
            google_drive_utils.init_google_drive_api('')
            return func(*args, **kwargs)
    return wrapper


@local_gdrive
def test_validate(mocker: MockerFixture):
    fileA_path = google_drive_utils.GoogleDrivePath('/folderA/fileA.txt')

    assert(fileA_path.fileID.endswith('/folderA/fileA.txt'))


@local_gdrive
def test_get_name_and_data(mocker: MockerFixture):
    folderA_path = google_drive_utils.GoogleDrivePath('/folderA')

    print('')
    name, media = folderA_path.get_name_and_data()
    assert(name == 'folderA')
    assert(media is None)

    fileA_path = folderA_path / 'fileA.txt'

    name, media = fileA_path.get_name_and_data()
    assert(name == 'fileA.txt')
    assert(media.read() == b'folderA/fileA.txt content!')


@local_gdrive
def test_mkdir(mocker: MockerFixture):
    print('')
    folderA_path = google_drive_utils.GoogleDrivePath('/folderA')
    folderA_truepath = folderA_path.fileID
    nested_path = folderA_path / 'nested_folder'

    nested_path.mkdir()
    assert(nested_path.fileID == os.path.join(folderA_truepath, 'nested_folder'))


@local_gdrive
def test_read(mocker: MockerFixture):
    print('')
    fileA_path = google_drive_utils.GoogleDrivePath('/folderA/fileA.txt')

    assert(fileA_path.read().read() == b'folderA/fileA.txt content!')


@local_gdrive
def test_write(mocker: MockerFixture):
    print('')
    folderA_path = google_drive_utils.GoogleDrivePath('/folderA')
    newfileC_path = folderA_path / 'fileC.txt'

    assert(newfileC_path.fileID is None)

    newfileC_path.write(io.BytesIO(b'new text file!'))
    assert(newfileC_path.fileID == os.path.join(folderA_path.fileID, 'fileC.txt'))
    assert(newfileC_path.read().read() == b'new text file!')

    newfileC_path.write(io.BytesIO(b'overwritten text'), overwrite=True)
    assert(newfileC_path.read().read() == b'overwritten text')


@local_gdrive
def test_clone(mocker: MockerFixture):
    print('')
    folderA_path = google_drive_utils.GoogleDrivePath('/folderA')
    with tempfile.TemporaryDirectory() as local:

        # test base cloning
        clone_dir = os.path.join(local, 'folder_clone')
        folderA_path.clone(clone_dir)
        assert(os.path.exists(os.path.join(clone_dir, 'fileA.txt')))
        assert(os.path.exists(os.path.join(clone_dir, 'fileB.txt')))
        assert(not os.path.exists(os.path.join(clone_dir, 'fileC.txt')))

        # test overwrite cloning
        clone_overwrite = os.path.join(local, 'content')
        os.mkdir(clone_overwrite)
        with open(os.path.join(clone_overwrite, 'fileB.txt'), mode='w') as f:
            f.write('fileB content!')

        with open(os.path.join(clone_overwrite, 'fileC.txt'), mode='w') as f:
            f.write('fileC content!')

        folderA_path.clone(clone_overwrite, overwrite=True)
        assert(os.path.exists(os.path.join(clone_overwrite, 'fileB.txt')))
        assert(os.path.exists(os.path.join(clone_overwrite, 'fileA.txt')))
        with open(os.path.join(clone_overwrite, 'fileB.txt'), mode='r') as f:
            assert(f.read() == 'folderA/fileB.txt content!')
        with open(os.path.join(clone_overwrite, 'fileC.txt'), mode='r') as f:
            assert(f.read() == 'fileC content!')

        # test dir clearing
        folderA_path.clone(clone_overwrite, overwrite=True, clear_dest=True)
        assert(not os.path.exists(os.path.join(clone_overwrite, 'fileC.txt')))

    return


@local_gdrive
def test_upload(mocker: MockerFixture):
    print('')
    folderC_path = google_drive_utils.GoogleDrivePath('/folderC')
    with tempfile.TemporaryDirectory() as local:
        upload_dir = os.path.join(local, 'folder_upload')
        os.mkdir(upload_dir)
        with open(os.path.join(upload_dir, 'fileA.txt'), mode='w') as f:
            f.write('/folderC/fileA.txt content!')

        folderC_path.upload(upload_dir)
        assert((folderC_path / 'fileA.txt').read().read() == b'/folderC/fileA.txt content!')
