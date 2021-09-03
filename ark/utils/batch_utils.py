import functools
import inspect
from types import FunctionType, CodeType
from ark.utils import io_utils


def batch_over_fovs(listing_strategy, loading_strategy, append_strategies, dirname='label_dir'):
    def decorator_batch(func):
        def wrapper_batch(*args, batch_size=5, **kwargs):
            filenames = listing_strategy(args[0])

            included_fovs = kwargs.get('included_fovs', None)
            suffix = kwargs.get('suffix', '_feature_0')
            if included_fovs:
                found_fovs = io_utils.extract_delimited_names(filenames, delimiter=suffix)
                filenames = \
                    [filenames[i] for i, fov in enumerate(found_fovs) if fov in included_fovs]

            batching_strategy = \
                [filenames[i:i + batch_size] for i in range(0, len(filenames), batch_size)]

            # batched return containers
            vals = tuple([] for i in range(len(append_strategies)))

            for batch_names in batching_strategy:
                loaded_data = loading_strategy(args[0], batch_names)

                val = func(loaded_data, *args[1:], **kwargs)
                for batch_val, container in zip(val, vals):
                    container.append(batch_val)

            return tuple(
                append_strat(container)
                for append_strat, container in zip(append_strategies, vals)
            )
        wrapper_batch.replaces = [(0, dirname)]
        return wrapper_batch
    return decorator_batch


def _blank(): pass


def make_batch_function(core_func, batching_strategy):
    def decorator_batch(func):

        replaces = None
        operational_func = batching_strategy(core_func)
        if hasattr(operational_func, 'replaces'):
            replaces = operational_func.replaces

        @functools.wraps(func)
        def wrapper_batch(*args, batch_size=5, **kwargs):
            return batching_strategy(core_func)(*args, batch_size=batch_size, **kwargs)

        wrapper_batch.__doc__ = func.__doc__
        wrapper_batch.__name__ = func.__name__
        wrapper_batch.__defaults__ = core_func.__defaults__ + (5,)

        # there be demon magic afoot here
        core_spec = inspect.getfullargspec(core_func)

        new_args = core_spec.args.copy()
        if replaces is not None:
            for ind, new_name in replaces:
                new_args[ind] = new_name

        all_args = tuple(new_args + ['batch_size'] + core_spec.kwonlyargs)

        passer_args = [
            len(core_spec.args) + 1,
            len(core_spec.kwonlyargs),
            _blank.__code__.co_nlocals,
            _blank.__code__.co_stacksize,
            core_func.__code__.co_flags,
            _blank.__code__.co_code, (), (),
            all_args, wrapper_batch.__code__.co_filename,
            wrapper_batch.__code__.co_name,
            wrapper_batch.__code__.co_firstlineno,
            wrapper_batch.__code__.co_lnotab
        ]

        passer_code = CodeType(*passer_args)
        passer = FunctionType(passer_code, globals())
        passer.__defaults__ = core_func.__defaults__ + (5,)
        wrapper_batch.__wrapped__ = passer

        wrapper_args = [
            len(core_spec.args) + 1,
            len(core_spec.kwonlyargs),
            wrapper_batch.__code__.co_nlocals,
            wrapper_batch.__code__.co_stacksize,
            wrapper_batch.__code__.co_flags,
            wrapper_batch.__code__.co_code, (), (),
            all_args + ('args', 'kwargs'),
            wrapper_batch.__code__.co_filename,
            wrapper_batch.__code__.co_name,
            wrapper_batch.__code__.co_firstlineno,
            wrapper_batch.__code__.co_lnotab,
            wrapper_batch.__code__.co_freevars,
            wrapper_batch.__code__.co_cellvars
        ]

        wrapper_batch.__code__ = CodeType(*wrapper_args)

        return wrapper_batch
    return decorator_batch
