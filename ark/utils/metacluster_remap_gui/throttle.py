import asyncio
from time import time


class NullTimer:
    def cancel(self):
        pass


class Timer:
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


def throttle(wait):
    def decorator(fn):
        time_of_last_call = 0
        timer = NullTimer()

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
            timer = Timer(current_wait_time(), call_it)
            timer.start()
        return throttled
    return decorator
