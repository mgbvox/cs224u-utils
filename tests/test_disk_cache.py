import pickle
import tempfile
from pathlib import Path

from cs224u_utils.cache import DiskCacheConfig, disk_cache


def test_root_is_home():
    assert DiskCacheConfig().cache_root.parent == Path.home()


def test_get_cache_location():
    @disk_cache
    def my_func(foo: str) -> str:
        return foo

    res = my_func("foo")
    assert res == "foo"
    cache_loc = my_func.loc("foo")
    assert pickle.loads(cache_loc.read_bytes()) == "foo"
    assert my_func("foo") == "foo" == res


def test_cache_specify_alt_cache_location():
    with tempfile.TemporaryDirectory() as td:
        alt_root = Path(td) / ".disk_cache"

        @disk_cache(cache_root=alt_root)
        def my_func(foo: str) -> str:
            return foo

        assert my_func.cache_dir.parent == alt_root

        res = my_func("asdf")
        assert res == "asdf"
        assert my_func.loc("asdf").is_relative_to(alt_root)
