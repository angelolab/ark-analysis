"""Helper class for making and retrying requests to the MIBItracker.
Copyright (C) 2021 Ionpath, Inc.  All rights reserved.
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
    'HEAD', 'GET', 'PUT', 'DELETE', 'OPTIONS', 'TRACE', 'POST'
)

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
        url:
            The string url to the backend of a MIBItracker instance, e.g.
            ``'https://backend-dot-mibitracker-demo.appspot.com'``.
        email:
            The string email address of your MIBItracker account.
        password:
            The string password of your MIBItracker account.
        token:
            A JSON Web Token (JWT) to validate a MIBItracker session.
        refresh:
            Number of seconds since previous refresh to automatically
            refresh token. Defaults to 5 minutes. Set to 0 or None in order
            to not attempt refreshes.
        retries:
            The max number of retries for HTTP status errors. Defaults
            to ``MAX_RETRIES`` which is set to 3.
        retry_methods:
            The HTTP methods to retry. Defaults to
            ``RETRY_METHOD_WHITELIST`` which is the defaults to ``urllib3``'s
            whitelist with the addition of POST.
        retry_codes:
            The HTTP status codes to retry. Defaults to ``(502, 503)``,
            which are associated with transient errors seen on app engine.
        session_timeout:
            Timeout for MIBItracker requests.
            data_transfer_timeout: Timeout for data transfer requests.

    Attributes:
        url:
            The string url to the backend of a MIBItracker instance.
        session:
            A ``StatusCheckedSession`` that includes an authorization
            header and automatically raises for HTTP status errors.

    Raises:
        HTTPError:
            Raised by ``requests.raise_for_status()``, i.e. if a
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
        """Adds an authorization token to the session's default header.
        """
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
            route:
                The route to add to the base url, such as ``'/images/'``
                or ``'/tissues/?organ=tonsil'``.
            *args:
                Passed to ``requests.Session.get``.
            **kwargs:
                Passes to ``requests.Session.get``.
        Returns:
            requests.Session.get:
                The response from ``requests.Session.get``.
        """
        self._check_refresh()
        return self.session.get('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def post(self, route, *args, **kwargs):
        """Makes a POST request to the url using the session.

        Args:
            route:
                The route to add to the base url, such as ``'/slides/1/'``.
            *args:
                Passed to ``requests.Session.post``.
            **kwargs:
                Passes to ``requests.Session.post``.
        Returns:
            requests.Session.post:
                The response from ``requests.Session.post``.
        """
        self._check_refresh()
        return self.session.post('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def put(self, route, *args, **kwargs):
        """Makes a PUT request to the url using the session.

        Args:
            route:
                The route to add to the base url, such as ``'/images/1/'``.
            *args:
                Passed to ``requests.Session.put``.
            **kwargs:
                Passes to ``requests.Session.put``.
        Returns:
            requests.Session.put:
                The response from ``requests.Session.put``.
        """
        self._check_refresh()
        return self.session.put('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def delete(self, route, *args, **kwargs):
        """Makes a DELETE request to the url using the session.

        Args:
            route:
                The route to add to the base url, such as ``'/images/1/'``.
            *args:
                Passed to ``requests.Session.delete``.
            **kwargs:
                Passes to ``requests.Session.delete``.
        Returns:
            requests.Session.delete:
                The response from ``requests.Session.delete``.
        """
        self._check_refresh()
        return self.session.delete('{}{}'.format(
            self.url, self._prepare_route(route)), *args, **kwargs)

    def download_file(self, path):
        """Downloads a file from MIBItracker storage.

        Args:
            path:
                The path to the file in storage. This usually can be
                constructed from the run and image folders.
        Returns:
            io.BytesIO:
                An open file object containing the downloaded file's data,
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
        Returns:
            list:
                A list of JSON data for each run that matches the search.
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

    def get_channel_data(self, image_id, channel_name):
        """Gets a single channel from MIBItracker as a 2D numpy array.

        Args:
            image_id:
                The integer id of an image.
            channel_name:
                The name of the channel to download.
        Returns:
            numpy.ndarray:
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


class StatusCheckedSession(requests.Session):
    """Raises for HTTP errors and adds any response JSON to the message.
    """

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
