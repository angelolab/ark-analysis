import inspect
from types import FunctionType, CodeType
from ark.utils import io_utils

# here be dragons


def batch_over_fovs(listing_strategy, loading_strategy, append_strategies, dirname='label_dir'):
    """Generates batching strategy over field of view images.

    The first argument of a function passed to the resulting strategy will be replaced with
    `dirname`. For compatability, functions must take all 'per fov' data in as its first argument.

    Args:
        listing_strategy (Callable):
            function which takes a directory and gets all relevant filenames
        loading_strategy (Callable):
            function which takes a directory and loads batched filenames from said directory
        append_strategy (tuple):
            tuple of callables for how to 'rejoin' returned values after batch processing
        dirname (str):
            name of replaced function argument/parameter

    Returns:
        Callable:
            Batching strategy which can add batch processing to a function
    """

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


def _exec_format(val):
    return f"'{val}'" if isinstance(val, str) else val


def make_batch_function(core_func, batching_strategy):
    """ Creates a batch function from a core funcion and a given batching strategy

    For example, a core function can be `ark.spatial_analysis.calculate_cluster_spatial_enrichment`
    whose batching strategy can be generated via `batch_over_fovs`.

    Args:
        core_func (Callable):
            Function which will be made batch processing compatable.  The function is not
            overwritten
        batching_strategy (Callable):
            Implementation details on how to make the function batch processable.

    Returns:
        Callable:
            Batch processable version of `core_func`
    """

    def decorator_batch(func):

        # extract argument name replacements from batching strategy
        replaces = None
        operational_func = batching_strategy(core_func)
        if hasattr(operational_func, 'replaces'):
            replaces = operational_func.replaces

        # replace arguments
        core_spec = inspect.getfullargspec(core_func)

        new_args = core_spec.args.copy()
        if replaces is not None:
            for ind, new_name in replaces:
                new_args[ind] = new_name

        # create argument definition string used in exec call
        wrapper_parameter_def = [
            f'{new_arg}={_exec_format(core_spec.defaults[::-1][i])}'
            if i < len(core_spec.defaults) else new_arg
            for i, new_arg in enumerate(new_args[::-1])
        ][::-1]

        # again, here be dragons. not wyverns. straight up dragons

        # set scope for exec call
        scope = {
            'batching_strategy': batching_strategy,
            'core_func': core_func
        }

        # fear
        exec(f"def _wrapper_batch({', '.join(wrapper_parameter_def)}, batch_size=5):\n"
             + f"\treturn batching_strategy(core_func)({', '.join(new_args)}, "
             + "batch_size=batch_size)", scope)

        # extract wrapper from exec call
        wrapper_batch = scope['_wrapper_batch']

        # add relevant dummy function metadata
        wrapper_batch.__doc__ = func.__doc__
        wrapper_batch.__name__ = func.__name__
        wrapper_batch.__module__ = func.__module__

        # make compatable with 'help()' builtin
        all_args = tuple(new_args + core_spec.kwonlyargs)

        passer_args = [
            len(core_spec.args) + 1,
            len(core_spec.kwonlyargs),
            _blank.__code__.co_nlocals,
            _blank.__code__.co_stacksize,
            core_func.__code__.co_flags,
            _blank.__code__.co_code, (), (),
            all_args + ('batch_size',), wrapper_batch.__code__.co_filename,
            wrapper_batch.__code__.co_name,
            wrapper_batch.__code__.co_firstlineno,
            wrapper_batch.__code__.co_lnotab
        ]

        passer_code = CodeType(*passer_args)
        passer = FunctionType(passer_code, globals())
        passer.__defaults__ = core_func.__defaults__ + (5,)
        wrapper_batch.__wrapped__ = passer

        return wrapper_batch
    return decorator_batch
