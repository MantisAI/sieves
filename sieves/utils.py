import functools
from collections.abc import Callable, Iterable
from typing import Any

from frozendict import frozendict


def make_cacheable(func: Callable[..., Any]) -> Callable[..., Any]:
    """Utility to convert mutable into immutable args so that a function call is cacheable.
    Useful to be compatible with `functools.lrucache`.

    :param func: Callable to process args for.
    :return Callable: Callable with pre-processed arguments.
    """

    def _make_cacheable(value: Any) -> Any:
        """Goes through values recursively and makes them cacheable."
        :param value: Value to make cacheable.
        :return Any: Cacheable value.
        """
        try:
            hash(value)
            return value
        except TypeError:
            pass

        if isinstance(value, dict):
            return frozendict({k: _make_cacheable(v) for k, v in value.items()})
        elif isinstance(value, Iterable):
            return tuple(_make_cacheable(v) for v in value)

        return value

    @functools.wraps(func)
    def wrapped(*args: Iterable[Any], **kwargs: dict[str, Any]) -> Any:
        return func(*_make_cacheable(args), **_make_cacheable(kwargs))

    return wrapped
