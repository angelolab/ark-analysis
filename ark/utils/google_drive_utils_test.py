import os
import re
import mimetypes

from ark.utils import google_drive_utils

from pytest_mock import MockerFixture


def _op_in(qt, v):
    def check_parent(path):
        return qt == os.path.dirname(path)

    return check_parent


def _op_eq(qt, v):
    def check_name(path):
        return v == os.path.basename(path)

    def check_mimetype(path):
        if v == google_drive_utils._FOLDER_MIME:
            return os.path.isdir(path)
        elif v == google_drive_utils._SHORTCUT_MIME_CHECK.split(' ')[-1]:
            return False
        else:
            return True

    return check_name if qt == 'name' else check_mimetype


def _op_neq(qt, v):
    def check_name(path):
        return v != os.path.basename(path)

    def check_mimetype(path):
        if v == google_drive_utils._FOLDER_MIME:
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


operator_key = {
    'in': _op_in,
    '=': _op_eq,
    '!=': _op_neq,
    'and': _op_and,
    'or': _op_or,
}


def _parse_base_query(bq, funcs=None):
    query_term, operator, values = tuple(bq.split(' '))

    if query_term[0] == '$' and values[0] == '$':
        return operator_key[operator](funcs[query_term], funcs[values])
    else:
        return operator_key[operator](query_term, values)


def _parse_full_query(q: str, funcs=None):
    base_queries = re.findall('\\([^\\(\\)]*\\)', q)

    # trim parentheses
    base_queries_trimmed = [bq[1:-1] for bq in base_queries]

    results = dict(zip(
        ['$' + str(i) for i in range(len(base_queries))],
        [_parse_base_query(bq, funcs) for bq in base_queries_trimmed]
    ))

    q_reduced = q
    for i, bq in enumerate(base_queries):
        q_reduced = q_reduced.replace(bq, '$' + str(i))

    print(q_reduced)

    q_reduced_split = [term for term in q_reduced.split('(') if term != '']

    if len(q_reduced_split) == 1:
        return results['$0']
    else:
        return _parse_full_query(q_reduced, funcs=results)


def _parse_base_field(bf, dicts=None):
    name = bf[0]
    new_keys = [nk.remove(' ') for nk in bf[1].split(',')]

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


def _fill_fields(path: str, fields: dict):
    # always fix next page token to none
    fields['nextPageToken'] = None

    files = fields.get('files', None)
    if files is not None:
        if files.get('id', None) is not None:
            files['id'] = path
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

    pass


class MockedExecute:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class MockedFiles:
    def __init__(self, mock_drive_dir):
        self.mock_drive_dir = mock_drive_dir
        return

    def list(self, q, spaces, fields):
        assert(spaces == 'drive')

        query_function = _parse_full_query(q)

        matches = []
        for root, dirs, files in os.walk(self.mock_drive_dir):
            for name in files:
                if query_function(os.path.join(root, name)):
                    matches.append(os.path.join(root, name))

        parsed_fields = _parse_fields(fields)

        filled_fields = [_fill_fields(match, parsed_fields) for match in matches]

        response = {
            'nextPageToken': None,
            'files': [ff['files'] for ff in filled_fields],
        }

        return MockedExecute(response)

    def create(self, body, media_body=None, fields=None):
        assert(fields == 'id' or fields is None)

        path = ''
        for parent in body.get('parents', []):
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

        return MockedExecute(response)

    def get(self, fileId):
        return MockedExecute({
            'name': os.path.basename(fileId),
            'mimeType': mimetypes.types_map.get(
                os.path.splitext(fileId)[1],
                'application/octed-stream'
            )
        })

    def get_media(self, fileId):
        return MockedExecute(fileId)

    def update(self, fileId, media_body, media_mime_type):
        with open(fileId, 'wb') as f:
            f.write(media_body.read())
        return MockedExecute(None)


class MockedService:
    def __init__(self, mock_drive_dir):
        self.files = MockedFiles(mock_drive_dir)

    def files(self):
        return self.files


def _mocked_init(auth_pw, mock_drive_dir):
    google_drive_utils.SERVICE = MockedService(mock_drive_dir)


def _offline_mocker(mocker: MockerFixture, mock_drive_dir):
    mocker.patch(
        'ark.utils.google_drive_utils.init_google_drive_api',
        lambda x: _mocked_init(x, mock_drive_dir)
    )

    # TODO: write mockers for these
    mocker.patch(
        'googleapiclient.http.MediaFileUpload', None
    )
    mocker.patch(
        'googleapiclient.http.MediaIoBaseUpload', None
    )
    mocker.patch(
        'googleapiclient.http.MediaIoBaseDownload', None
    )

    pass
