from importlib.metadata import version

import polycomb


def test_version_is_exported_from_package_metadata():
    assert polycomb.__version__ == version("polycomb")
