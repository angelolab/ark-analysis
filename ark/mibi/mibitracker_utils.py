"""Helper class for making and retrying requests to the MIBItracker.
Copyright (C) 2021 Ionpath, Inc.  All rights reserved.

Hehe copying from Ionpath go command-C
"""

import datetime
import io
import json
import os
import warnings

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from skimage import io as skio
from urllib3.util.retry import Retry

# The number of retries will be applied to these code and methods only.
MAX_RETRIES = 3
# Unavailable and Bad Gateway; we see this transiently on App Engine.
RETRY_STATUS_CODES = (502, 503)
# POST is added here as compared to the urllib3 defaults.
RETRY_METHOD_WHITELIST = (
    'HEAD', 'GET', 'PUT', 'DELETE', 'OPTIONS', 'TRACE', 'POST')
# Timeout for MIBItracker requests
SESSION_TIMEOUT = 10
# Timeout for data transfer requests
DATA_TRANSFER_TIMEOUT = 30


class MibiTrackerError(Exception):
    """Raise for exceptions where the response from the MibiTracker API is
        invalid or unexpected."""


class MibiRequests():
    """Helper class for making requests to the MIBItracker.
    This is an opinionated way of using ``requests.Session`` with the following
    features:
    - In the case of specified HTTP errors, requests are retried.
    - All responses call ``raise_for_status()``.
    - An instance requests an authorization token upon initialization, and
        includes it in a default header for all future requests. No token
        refresh capabilities are built in, so if the token expires a new
        instance needs to be initialized.
    Args:
        url: The string url to the backend of a MIBItracker instance, e.g.
            ``'https://backend-dot-mibitracker-demo.appspot.com'``.
        email: The string email address of your MIBItracker account.
        password: The string password of your MIBItracker account.
        token: A JSON Web Token (JWT) to validate a MIBItracker session.
        refresh: Number of seconds since previous refresh to automatically
            refresh token. Defaults to 5 minutes. Set to 0 or None in order
            to not attempt refreshes.
        retries: The max number of retries for HTTP status errors. Defaults
            to ``MAX_RETRIES`` which is set to 3.
        retry_methods: The HTTP methods to retry. Defaults to
            ``RETRY_METHOD_WHITELIST`` which is the defaults to ``urllib3``'s
            whitelist with the addition of POST.
        retry_codes: The HTTP status codes to retry. Defaults to ``(502, 503)``,
            which are associated with transient errors seen on app engine.
        session_timeout: Timeout for MIBItracker requests.
        data_transfer_timeout: Timeout for data transfer requests.
    Attributes:
        url: The string url to the backend of a MIBItracker instance.
        session: A ``StatusCheckedSession`` that includes an authorization
            header and automatically raises for HTTP status errors.
    Raises:
        HTTPError: Raised by ``requests.raise_for_status()``, i.e. if a
            response's status code is >= 400.
    """

    def __init__(self,
                 url,
                 email=None,
                 password=None,
                 token=None,
                 refresh=300,  # 5 minutes
                 retries=MAX_RETRIES,
                 retry_methods=RETRY_METHOD_WHITELIST,
                 retry_codes=RETRY_STATUS_CODES,
                 session_timeout=SESSION_TIMEOUT,
                 data_transfer_timeout=DATA_TRANSFER_TIMEOUT):

        self.url = url.rstrip('/')  # We add this as part of request params
        self.session = StatusCheckedSession(timeout=session_timeout)
        self._data_transfer_timeout = data_transfer_timeout
        self._refresh_seconds = refresh
        self._last_refresh = datetime.datetime.now()

        # Provide either an email and password, or a token
        # The token will be used in lieu of email and password if provided
        if token is not None:
            self.session.headers.update({
                'Authorization': 'JWT {}'.format(token)
            })
            self.session.options(self.url)
        elif email is not None and password is not None:
            self._auth(url, email, password)
        else:
            raise ValueError(
                'Provide either both an email and password or a token'
            )

        retry = Retry(status=retries, method_whitelist=retry_methods,
                      status_forcelist=retry_codes, backoff_factor=0.3)
        # Set this session to use these retry settings for all http[s] requests
        self.session.mount('http://', HTTPAdapter(max_retries=retry))
        self.session.mount('https://', HTTPAdapter(max_retries=retry))

    def _auth(self, url, email, password):
        """Adds an authorization token to the session's default header."""
        response = self.session.post(
            '{}/api-token-auth/'.format(url),
            headers={'content-type': 'application/json'},
            data=json.dumps({'email': email, 'password': password}))
        token = response.json()['token']
        self.session.headers.update({'Authorization': 'JWT {}'.format(token)})

    def refresh(self):
        """Refreshes the authorization token stored in the session header.
        Raises HTTP 400 if attempting to refresh an expired token."""
        token = self.session.post(  # use the session to avoid recursion
            '{}/api-token-refresh/'.format(self.url),
            data=json.dumps(
                {'token': self.session.headers['Authorization'][4:]}
            ),
            headers={'content-type': 'application/json'},
        ).json()['token']
        self.session.headers.update({'Authorization': 'JWT {}'.format(token)})

    def _check_refresh(self):
        current_time = datetime.datetime.now()
        if (self._refresh_seconds and
                (current_time - self._last_refresh).total_seconds() >
                self._refresh_seconds):
            self._last_refresh = current_time
            self.refresh()

    @staticmethod
    def _prepare_route(route):
        if not route.startswith('/'):
            return '/{}'.format(route)
        return route

    def get(self, route, *args, **kwargs):
        """Makes a GET request to the url using the session.
        Args:
            route: The route to add to the base url, such as ``'/images/'``
                or ``'/tissues/?organ=tonsil'``.
            *args: Passed to ``requests.Session.get``.
            **kwargs: Passes to ``requests.Session.get``.
        Returns:
            The response from ``requests.Session.get``.
        """
        self._check_refresh()
        return self.session.get('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def post(self, route, *args, **kwargs):
        """Makes a POST request to the url using the session.
        Args:
            route: The route to add to the base url, such as ``'/slides/1/'``.
            *args: Passed to ``requests.Session.post``.
            **kwargs: Passes to ``requests.Session.post``.
        Returns:
            The response from ``requests.Session.post``.
        """
        self._check_refresh()
        return self.session.post('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def put(self, route, *args, **kwargs):
        """Makes a PUT request to the url using the session.
        Args:
            route: The route to add to the base url, such as ``'/images/1/'``.
            *args: Passed to ``requests.Session.put``.
            **kwargs: Passes to ``requests.Session.put`.
        Returns:
            The response from ``requests.Session.put``.
        """
        self._check_refresh()
        return self.session.put('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def delete(self, route, *args, **kwargs):
        """Makes a DELETE request to the url using the session.
        Args:
            route: The route to add to the base url, such as ``'/images/1/'``.
            *args: Passed to ``requests.Session.delete``.
            **kwargs: Passes to ``requests.Session.delete``.
        Returns:
            The response from ``requests.Session.delete``.
        """
        self._check_refresh()
        return self.session.delete('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def download_file(self, path):
        """Downloads a file from MIBItracker storage.
        Args:
            path: The path to the file in storage. This usually can be
                constructed from the run and image folders.
        Returns: An open file object containing the downloaded file's data,
            rewound to the beginning of the file.
        """
        response = self.get('/download/', params={'path': path})
        url = requests.get(response.json()['url'],
                           timeout=self._data_transfer_timeout)
        buf = io.BytesIO()
        buf.write(url.content)
        buf.seek(0)
        return buf

    def search_runs(self, run_name, run_label=None):
        """Searches for runs which have the name and optionally label.
        Args:
            run_name: The name of the run the image belongs to.
            run_label: (optional) The label of the run.
        Returns: A list of JSON data for each run that matches the search.
            If only ``run_name`` is specified, this list could be of any
            length as a run's name is not necessarily unique. If
            ``run_label`` is also specified, this is guaranteed to be
            unique and the returned list could either be of length
            zero or one.
        """

        payload = {'name': run_name}

        if run_label:
            payload['label'] = run_label

        return self.get('/runs/', params=payload).json()

    def copy_run(self, old_label, new_label, **kwargs):
        """Creates a copy of the run corresponding to the label.
        Args:
            old_label: The label of the run to copy.
            new_label: The label to use for the copied run.
            kwargs: Any key-value pair that can be included in the json when
                creating the new run to differentiate it from the old run,
                such as a different project id or FOV size (such as if images
                were cropped from their originals).
        Returns:
            A tuple of the response JSON returned by old run and from creating
            its copy, respectively.
        """
        # We don't expect there to be many runs with the same label, so it is
        # safe to turn paging off.
        response = self.get('/runs/?label={}&paging=no'.format(old_label))
        data = response.json()
        try:
            assert len(data) == 1
        except AssertionError:
            raise MibiTrackerError('Expected 1 run with label {}, but {} were '
                                   'found'.format(old_label, len(data)))
        data = data[0]

        # Get the XMLs from the original runs. They may not have the date and
        # mass cal filled out; if those are needed they should be downloaded
        # from the bucket instead.
        xml_path = '/'.join((data['path'], data['xml']))
        buf = self.download_file(xml_path)

        run_data = {
            'instrument': data['instrument']['id'],  # required
            'slides': ','.join(str(d['id']) for d in data['slide_ids']),
            'label': new_label,
            'fov_size': data['fov_size'],
            'aperture': data['aperture'] and data['aperture']['id'],
            'project': data['project'] and data['project']['id'],  # optional
            'description': data['description'],
            'operator': data['operator'],  # Not yet used, but field exists
            'user_run_date': '{}T00:00:00'.format(data['run_date']),
        }
        # The old run date supercedes the possibly-missing date in the xml
        # A timestamp is temporarily added for the JSON encoded but will be
        # stripped by the application.

        run_data.update(kwargs)
        files = {'xml': (data['filename'], buf, 'application/xml')}

        response = self.post('/runs/', data=run_data, files=files)
        return data, response.json()

    def copy_run_image_metadata(self, old_run_label, new_run_label, **kwargs):
        """This updates image metadata but not the actual TIFF data.
        Each new image will remain unavailable until a TIFF is uploaded.
        Args:
            old_run_label: The label of the run whose images' metadata is to
                be copied.
            new_run_label: The label of the run whose images' metadata will be
                updated. This run must already exist.
            kwargs: Any key-value pair that can be included in the json when
                updating the the images to differentiate them from the old
                runs', such as a different project id or frame (such as if
                images were cropped from their originals).
        Returns:
            A dictionary with keys of the original image IDs, and values of the
            response JSON returned when updating the new images.
        """
        old_images = self.get(
            '/images/?run__label={}&paging=no'.format(old_run_label))

        image_map = {}
        for item in old_images.json():

            # Get image from copied run
            response = self.get(
                '/images/?run__label={}&folder={}&paging=no'.format(
                    new_run_label, item['folder']))
            assert len(response.json()) == 1
            new_image = response.json()[0]
            # Update the section and tissue of the copied image using
            # info from the original image
            updated_image = {
                'tissue': item['tissue'] and item['tissue']['id'],
                'fov_size': item['fov_size'],
            }
            if item['section']:
                updated_image.update({'section': item['section']['id']})

            # TODO (Stanley) all these checks are for version compatibility.
            # Should be removed for mibitracker >= V1.1.5
            if 'aperture' in item:
                updated_image.update({
                    'aperture': item['aperture'] and item['aperture']['id']
                    })
            if 'imaging_preset' in item:
                updated_image.update({'imaging_preset': item['imaging_preset']})
            if 'lens1_voltage' in item:
                updated_image.update({'lens1_voltage': item['lens1_voltage']})

            # Add optional updates to copy such as a different frame size due to
            # cropping, or fixed mass calibration, etc.
            updated_image.update(kwargs)

            # Add SED
            sed = item['sed']
            headers = {}
            if sed:
                sed_path = '/'.join((item['run']['path'], item['folder'], sed))
                _, ext = os.path.splitext(sed_path)
                # MIBItracker only accepts TIFF, PNG and BMP images so it has
                # to be one of these if it's already in there
                if ext.lower() == '.bmp':
                    content_type = 'image/bmp'
                elif ext.lower() in ('.tif', '.tiff'):
                    content_type = 'image/tiff'
                elif ext.lower() == '.png':
                    content_type = 'image/png'

                buf = self.download_file(sed_path)
                files = {'attachment': (sed, buf, content_type)}
                data = updated_image
            else:
                files = None
                updated_image['sed'] = None
                # When there's no file upload, PUT requires JSON
                data = json.dumps(updated_image)
                headers.update({'content-type': 'application/json'})

            response = self.put(
                '/images/{}/'.format(new_image['id']),
                files=files,
                data=data,
                headers=headers
            )
            image_map[item['id']] = response.json()
        return image_map

    def _upload_mibitiff(self, url, tiff_file):
        # This shouldn't send mibitracker credentials so don't use the session
        response = requests.put(
            url,
            data=tiff_file,
            headers={'content-type': 'image/tiff'},
            timeout=self._data_transfer_timeout
        )
        response.raise_for_status()
        return response

    def upload_mibitiff(self, tiff_file, run_id=None):
        """Uploads a single TIFF to the MIBItracker.
        This uses the 'run' and 'folder' fields in the MibiTiff's description to
        associate this with the correct image.
        Args:
            tiff_file: A string path to a TIFF file or an open file object
                containing the TIFF data.
            run_id: The ID of the run the image is from. This enables checking
                of the image metadata against expected values and may become
                mandatory in the future.
        Returns:
            The response from the MIBItracker after uploading the file and
            queuing it to be processed into viewable data.
        Raises:
            TypeError: Raised if tiff_file is not a string path or file object.
            ValueError: Raised if run_id is None.
        """
        if run_id is None:
            raise ValueError(
                'run_id is mandatory in MIBitracker >=v1.1 and cannot be None')
        response = self.get(
            f'/upload_mibitiff/sign_tiff_url/?run_id={run_id}').json()
        try:
            with open(tiff_file, 'rb') as fh:
                self._upload_mibitiff(response['url'], fh)
        except TypeError:
            try:
                tiff_file.seek(0)
            except:
                raise TypeError('tiff_file must be a string or file object')
            self._upload_mibitiff(response['url'], tiff_file)
        return self.post(
            '/upload_mibitiff/',
            data=json.dumps(
                {'location': response['location'], 'run_id': run_id}),
            headers={'content-type': 'application/json'}
        )

    def _upload_channel(self, image_id, image_file, filename):
        _, ext = os.path.splitext(filename)
        files = {
            'attachment': (filename,
                           image_file,
                           'image/{}'.format(ext[1:].lower()))
        }

        response = self.post(
            f'/images/{image_id}/upload_channel/',
            files=files
        )
        return response

    def upload_channel(self, image_id, image_file, filename=None):
        """Uploads a grayscale PNG or TIFF to the MIBItracker.
        Args:
            image_id: The integer id of the image to associate the channel with.
            image_file: A string path to a PNG or TIFF file, or an open file
                object of a PNG or TIFF file to upload.
            filename: The name to use for the file being uploaded. If
                image_file is a path, this can be omitted and that file's name
                will be used. It should be the name of the channel/layer with
                an extension to indicate format, such as "CD45.tiff" or
                "cell_boundaries.png".
        Returns:
            The response from the MIBItracker after uploading the file.
        Raises:
            TypeError: Raised if image_file is not a string path or file object.
        """
        try:
            with open(image_file, 'rb') as fh:
                if not filename:
                    filename = os.path.basename(image_file)
                return self._upload_channel(image_id, fh, filename)
        except TypeError:
            try:
                image_file.seek(0)
            except:
                raise TypeError('image_file must be a string or file object')
            if not filename:
                try:
                    filename = os.path.basename(image_file.name)
                except AttributeError:
                    raise ValueError('filename must be provided with a file '
                                     'object that does not have a name')
            return self._upload_channel(image_id, image_file, filename)

    def run_images(self, run_label):
        """Gets a JSON array of image metadata from a given run label.
        Args:
            run_label: The unique string label of a run.
        Returns:
            A list of dicts of of image metadata for the specified run.
        """
        return self.get(
            '/images/',
            params={'run__label': run_label, 'paging': 'no'}).json()

    def image_conjugates(self, image_id):
        """Gets a JSON array of panel conjugates from a given image id.
        Args:
            image_id: The integer id of an image.
        Returns:
            A list of dicts of antibody conjugate details from the image's
            panel. This will be an empty list of the image does not have a
            section assigned, or if its section does not have a panel assigned.
        """
        return self.get(
            '/images/{}/conjugates/'.format(image_id),
            params={'paging': 'no'}).json()

    def image_id(self, run_label, fov_id):
        """Gets the primary key of an image given the specified run and FOV.
        Args:
            run_label: The label of the run the image belongs to. If no images
                found using run label (which is the unique identifier of
                each run), run name is checked instead (run name is not
                guaranteed to be unique per run).
            fov_id: The FOV ID, in the format of ``FOV<n>`` or ``Point<n>``
                for data generated with MIBIcontrol and MiniSIMS, respectively.
        Returns:
            An int id corresponding to the primary key of the image.
        Raises:
            ValueError: Raised if no images match the specified run and FOV,
                or if more than one image matches the specified run and FOV.
        """
        results = self.get(
            '/images/',
            params={
                'run__label': run_label,
                'number': fov_id,
                'paging': 'no'}
        ).json()

        len_results = len(results)
        if len_results == 0:
            warnings.warn(f'No images found matching run label: {run_label}, '
                          f'fov_id: {fov_id}. Checking run name '
                          f'instead.')
            results = self.get(
                '/images/',
                params={
                    'run__name': run_label,
                    'number': fov_id,
                    'paging': 'no'}
            ).json()

        len_results = len(results)
        if len_results == 0:
            raise MibiTrackerError(
                f'No images found matching run {run_label} {fov_id}.')
        if len_results > 1:
            raise MibiTrackerError(
                f'Multiple images match run {run_label} {fov_id}.'
            )

        return results[0]['id']

    def get_channel_data(self, image_id, channel_name):
        """Gets a single channel from MIBItracker as a 2D numpy array.
        Args:
            image_id: The integer id of an image.
            channel_name: The name of the channel to download.
        Returns:
            A MxN numpy array of the channel data.
        """
        try:
            response = self.get(
                f'images/{image_id}/channel_url/',
                params={
                    'channel': channel_name
                })
            response.raise_for_status()
        except HTTPError as e:
            if e.response.status_code == 404:
                raise MibiTrackerError(
                    f'Channel \'{channel_name}\' not found in the image.')
            raise e

        png = requests.get(response.json()['url'])
        buf = io.BytesIO()
        buf.write(png.content)
        buf.seek(0)
        return skio.imread(buf)

    def create_imageset(self, image_ids, imageset_name, project_id,
                        imageset_description=None):
        """Creates a new imageset in a project with the specified images.
         Note that all images in the set must be either in the project
         specified or in other projects to which the user has access and
         sharing is enabled.
        Args:
            image_ids: A list of ints of ids of the images in MIBItracker
                corresponding to the images to be added to the new imageset.
            imageset_name: A string name for the new imageset.
            project_id: An integer id specifying the project in which to create
                the image set.
            imageset_description: (optional) A string description for the new
                imageset. Defaults to None.
        """
        self.post(
            '/imagesets/',
            data=json.dumps({
                'images': image_ids,
                'name': imageset_name,
                'project': project_id,
                'description': imageset_description
            }),
            headers={'content-type': 'application/json'})


class StatusCheckedSession(requests.Session):
    """Raises for HTTP errors and adds any response JSON to the message."""

    def __init__(self, timeout=SESSION_TIMEOUT):
        super(StatusCheckedSession, self).__init__()
        self.timeout = timeout

    @staticmethod
    def _check_status(response):
        try:
            response.raise_for_status()
        except HTTPError as e:
            try:
                response_json = response.json()
            except json.decoder.JSONDecodeError:
                response_json = None
            raise HTTPError(str(e), response_json, response=response)
        return response

    def _set_timeout(self, kwargs):
        if 'timeout' not in kwargs:
            kwargs.update({'timeout': self.timeout})

    def get(self, *args, **kwargs):
        self._set_timeout(kwargs)
        response = super().get(*args, **kwargs)
        return self._check_status(response)

    def options(self, *args, **kwargs):
        self._set_timeout(kwargs)
        response = super().options(*args, **kwargs)
        return self._check_status(response)

    def post(self, *args, **kwargs):
        self._set_timeout(kwargs)
        response = super().post(*args, **kwargs)
        return self._check_status(response)

    def put(self, *args, **kwargs):
        self._set_timeout(kwargs)
        response = super().put(*args, **kwargs)
        return self._check_status(response)

    def delete(self, *args, **kwargs):
        self._set_timeout(kwargs)
        response = super().delete(*args, **kwargs)
        return self._check_status(response)
