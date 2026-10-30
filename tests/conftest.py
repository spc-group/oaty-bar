import socket

import pytest
from prefect.testing.utilities import prefect_test_harness
from pytest import MonkeyPatch
from pytest_postgresql.janitor import DatabaseJanitor
from tiled.client import Context, from_context, from_uri
from tiled.server.app import build_app_from_config

from .tiled_server import TiledServerInfo
from .tiled_trees import build_tree


## Add an option for default skipping slow tests
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def modify_runslow(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_collection_modifyitems(config, items):
    modify_runslow(config, items)



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


TILED_PROFILES = """
tiled_read_only:
  uri: {uri}
tiled_writable:
  uri: {uri}?api_key={api_key}
"""


@pytest.fixture(scope="session")
def tiled_server(tmp_path_factory, session_mocker, postgresql_proc):
    db_proc = postgresql_proc
    monkeypatch = MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("tiled")
    # Get a random open port
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    port = 8719
    dbjanitor = DatabaseJanitor(
        user=db_proc.user,
        host=db_proc.host,
        port=db_proc.port,
        dbname=db_proc.dbname,
        template_dbname=db_proc.template_dbname,
        maintenance_dbname=db_proc.maintenance_dbname,
        password=db_proc.password,
        isolation_level=None,
    )
    catalog_uri = f"postgresql://{db_proc.user}:{db_proc.password}@{db_proc.host}:{db_proc.port}/{db_proc.dbname}"
    # catalog_uri = f"sqlite:///{tmp_path / 'catalog.db'}"
    server_info = TiledServerInfo(tmp_path, port=str(port), catalog_uri=catalog_uri)
    monkeypatch.setenv("TILED_CONFIG", str(tmp_path / "server_config.yml"))
    # Start the tiled server
    # Set up the profiles corresponding to the server
    profile_dir = tmp_path / "tiled" / "profiles"
    profile_dir.mkdir(parents=True)
    session_mocker.patch("tiled.profiles.paths", [profile_dir])
    with open(profile_dir / "default.yml", mode="w") as fp:
        fp.write(
            TILED_PROFILES.format(uri=server_info.uri, api_key=server_info.api_key)
        )
    with dbjanitor:
        try:
            server_info.start()
            # Execute tests
            yield server_info
        finally:
            # Make sure the server gets cleaned up when we're done
            monkeypatch.undo()
            server_info.stop()


@pytest.fixture()
def tiled_catalog(tiled_server):
    catalog = from_uri(tiled_server.uri, api_key=tiled_server.api_key)
    existing_keys = list(catalog.keys())
    try:
        yield catalog
    finally:
        # Remove any transiently created datasets
        new_keys = [key for key in catalog.keys() if key not in existing_keys]
        for key in new_keys:
            catalog[key].delete(recursive=True, external_only=False)
        catalog.context.close()
        del catalog
