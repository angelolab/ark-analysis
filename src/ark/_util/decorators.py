import functools
import warnings
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

import modguard

P = ParamSpec("P")
T = TypeVar("T")


# TODO: change to paramspec as soon as we drop support for python 3.9, see https://stackoverflow.com/a/68290080
def deprecation_alias(version: str, **aliases: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorate a function to warn user of use of arguments set for deprecation.

    Parameters
    ----------
    aliases
        Deprecation argument aliases to be mapped to the new arguments.

    Returns
    -------
    A decorator that can be used to mark an argument for deprecation and substituting it with the new argument.

    Raises
    ------
    TypeError
        If the provided aliases are not of string type.

    Example
    -------
    Assuming we have an argument 'table' set for deprecation and we want to warn the user and substitute with 'tables':

    ```python
    @deprecation_alias(table="tables")
    def my_function(tables: AnnData | dict[str, AnnData]):
        pass
    ```
    """

    @modguard.public()
    def deprecation_decorator(f: Callable[P, T]) -> Callable[Concatenate[str, P], T]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[Concatenate[str, P], T]:
            class_name = f.__qualname__
            _rename_kwargs(f.__name__, kwargs, aliases, class_name, dep_version=version)
            return f(*args, **kwargs)

        return wrapper

    return deprecation_decorator


def _rename_kwargs(func_name: str, kwargs: dict[str, Any], aliases: dict[str, str], class_name: None | str, dep_version: str) -> None:
    """Rename function arguments set for deprecation and gives warning in case of usage of these arguments."""
    for alias, new in aliases.items():
        if alias in kwargs:
            class_name = class_name + "." if class_name else ""
            if new in kwargs:
                raise TypeError(
                    f"{class_name}{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is being deprecated in Ark version {dep_version}, only use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is being deprecated as an argument to `{class_name}{func_name}` in Ark "
                    f"version {dep_version}, switch to `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)
