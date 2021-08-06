import asyncio
from time import time


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()


def throttle(wait):
    def decorator(fn):
        time_of_last_call = 0
        fn_latest = fn
        scheduled, timer = False, None
        new_args, new_kwargs = None, None

        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer, fn_latest

            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer, fn_latest
                time_of_last_call = time()
                fn_latest(*new_args, **new_kwargs)
                scheduled = False

            time_since_last_call = time() - time_of_last_call
            new_args, new_kwargs = args, kwargs
            if not scheduled:
                scheduled = True
                new_wait = max(0, wait - time_since_last_call)
                timer = Timer(new_wait, call_it)
                timer.start()
        return throttled
    return decorator
