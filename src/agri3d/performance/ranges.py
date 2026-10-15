"""Profiler/NVTX ranges are inactive by default and never change tensors."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_ENABLED = ContextVar("pointnet2_profiling", default=False)
_NVTX = ContextVar("pointnet2_nvtx", default=False)


@contextmanager
def profiling_ranges(nvtx=False):
    token, nvtoken = _ENABLED.set(True), _NVTX.set(nvtx)
    try:
        yield
    finally:
        _NVTX.reset(nvtoken)
        _ENABLED.reset(token)


@contextmanager
def range_scope(name):
    if not _ENABLED.get():
        yield
        return
    with ExitStack() as stack:
        stack.enter_context(torch.profiler.record_function(name))
        if _NVTX.get() and torch.cuda.is_available():
            stack.enter_context(torch.cuda.nvtx.range(name))
        yield


def profile_range(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not _ENABLED.get():
                return function(*args, **kwargs)
            with range_scope(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate
