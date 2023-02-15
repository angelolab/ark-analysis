import asyncio

import pytest

from ark.utils.metacluster_remap_gui.throttle import throttle


@pytest.mark.asyncio
async def test_can_decorate_function_with_throttle():
    c = 0

    @throttle(.01)
    def inc():
        nonlocal c
        c = c + 1

    assert c == 0
    inc()
    await asyncio.sleep(.02)
    assert c == 1


@pytest.mark.asyncio
async def test_throttle_actually_throttles():
    c = 0

    @throttle(.01)
    def inc():
        nonlocal c
        c = c + 1

    inc()
    inc()
    inc()
    inc()
    await asyncio.sleep(.02)
    assert c == 1  # only ran once
    inc()
    await asyncio.sleep(.02)
    assert c == 2  # runs again after timeout


@pytest.mark.asyncio
async def test_throttle_makes_final_call_take_precedence():
    c = "z"

    @throttle(.01)
    def go(char):
        nonlocal c
        c = c + char

    go('a')
    await asyncio.sleep(.001)  # just allowing for overhead
    go('e')
    go('q')
    go('k')
    await asyncio.sleep(.02)
    assert c == "zak"
