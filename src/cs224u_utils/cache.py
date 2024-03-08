"""This module provides decorators and utils for caching calls to functions."""

import functools
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union, overload

from pydantic import BaseModel, Field

# Define a type variable for the function
F = TypeVar("F", bound=Callable[..., Any])


def deterministic_hash(string: str) -> str:
    """A hash for strings that always returns the same output for a given input."""
    return hashlib.sha256(string.encode()).hexdigest()


def safe_to_string(obj: object) -> str:
    """Convert an object into its string representation.

    If the object string repr is of the form <path.ClassName object at 0xmemory_address>,
    instead just use the classname of the object.

    Args:
        obj: Any object.

    Returns:
        A str repr of that object.
    """
    as_str = str(obj)
    if "<" in as_str:
        # The string repr of the object is ClassName<memory address>
        # This is non-deterministic (at least, the memory part is)
        # So just use its class name
        return obj.__class__.__name__
    return as_str


class DiskCacheConfig(BaseModel):
    """Cache configuration."""

    cache_root: Path = Field(default_factory=lambda: Path.home() / ".disk_cache")


class DiskCacheWrapper(Generic[F]):
    """Stateful wrapper around a disk_cache decorated function."""

    def __init__(
        self,
        callback_fn: F,
        config: DiskCacheConfig,
    ) -> None:
        self._fn = callback_fn
        self.config = config
        if not self.config.cache_root.exists():
            self.config.cache_root.mkdir(exist_ok=True, parents=True)

        self.cache_dir = self.config.cache_root / (self._fn.__name__ if self._fn.__name__ else "lambda")

    def loc(self, *args: Any, **kwargs: Any) -> Path:
        """Locate the cache file for a given set of args and kwargs."""
        key = ",".join(
            [safe_to_string(arg) for arg in args] + [f"{k}:{safe_to_string(v)}" for k, v in kwargs.items()],
        )
        key = deterministic_hash(key)

        return self.cache_dir / key

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke self._fn with the given args/kwargs."""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        cache_file = self.loc(*args, **kwargs)
        if cache_file.exists():
            with cache_file.open("rb") as f:
                return pickle.load(f)

        else:
            result = self._fn(*args, **kwargs)
            with cache_file.open("wb") as f:
                pickle.dump(result, f)
            return result


# Overload definitions for the decorator
@overload
def disk_cache(
    fn: F,
    _config_object: Optional[DiskCacheConfig] = None,
    **raw_config: None,
) -> DiskCacheWrapper:
    """Case for:

    Examples::
        @disk_cache
        def my_func(): ...
    """
    ...


@overload
def disk_cache(
    fn: None = None,
    _config_object: None = None,
    **raw_config: Any,
) -> Callable[[F], DiskCacheWrapper]:
    """Case for:

    Examples::
        @disk_cache(some=config)
        def my_func(): ...

    """
    ...


def disk_cache(
    fn: Optional[Callable] = None,
    _config_object: Optional[DiskCacheConfig] = None,
    **raw_config: Any,
) -> Union[DiskCacheWrapper, functools.partial[DiskCacheWrapper]]:
    """A very basic disk caching decorator.

    Will cache function call results on disk indexed by the argument values to
    said function. Intended to perform identically to @lru_cache(None), but persists
    between runs.
    """
    if raw_config:
        # parse it and reinit this decorator with the configured config object
        parsed_config = DiskCacheConfig.parse_obj(raw_config)
        return functools.partial(disk_cache, _config_object=parsed_config)

    assert fn is not None, ValueError("Decorated function cannot be None!")
    config = _config_object or DiskCacheConfig()
    return DiskCacheWrapper(callback_fn=fn, config=config)
