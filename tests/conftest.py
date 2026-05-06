import pytest
from tiled.client import Context, from_context
from tiled.server.app import build_app_from_config


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
