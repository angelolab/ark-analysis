import asyncio
from time import time


def throttle(wait):
    """Second order decorator for rate-limiting a function within an asyncio concurrent app

    - The first call will always happen *without* delay.
    - Subsequent calls, within wait seconds, are dropped, even if argurments differ.
    - The final call will always execute, sometimes with a delay. This guarentees
      that that final value passed to will be applied.

    Example usage::

        @throttle(.5)
        def update_a_gui_element(e):
            do_stuff(e.name)
            but_not_too_often()

    Args:
        wait (float):
            minimum time between subsequent calls, in seconds
    Returns:
        function:
            Decorator for throttling by *wait* seconds
    """
    def decorator(fn):
        time_of_last_call = 0
        timer = _NullTimer()

        def current_wait_time():
            time_since_last_call = time() - time_of_last_call
            return max(0, wait - time_since_last_call)

        def throttled(*args, **kwargs):
            nonlocal time_of_last_call, timer

            def call_it():
                nonlocal time_of_last_call
                time_of_last_call = time()
                fn(*args, **kwargs)

            timer.cancel()
            timer = _Timer(current_wait_time(), call_it)
            timer.start()
        return throttled
    return decorator


class _NullTimer:
    def cancel(self):
        pass


class _Timer:
    def __init__(self, timeout, callback):
        self._task = None
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        if self._task is not None:
            self._task.cancel()
