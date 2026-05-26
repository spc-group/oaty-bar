import pytest
from prefect.testing.utilities import prefect_test_harness
from tiled.client import Context, from_context
from tiled.server.app import build_app_from_config

from .tiled_trees import build_tree


@pytest.fixture(scope="package")
def xafs_run():
    with build_tree() as run:
        yield run


@pytest.fixture(autouse=True, scope="session")
def prefect_server():
    with prefect_test_harness() as server:
        yield server


@pytest.fixture()
def results_catalog(tmp_path):
    "Test 'mounting' sub-trees of a catalog."
    uri = f"sqlite:///{tmp_path}/tiled.sqlite"
    one_tree_config = {
        "trees": [
            {
                "path": "/",
                "tree": "catalog",
                "args": {
                    "uri": uri,
                    "init_if_not_exists": True,
                    "writable_storage": [tmp_path / "data"],
                },
            },
        ]
    }
    with Context.from_app(build_app_from_config(one_tree_config)) as context:
        client = from_context(context)
        yield client
