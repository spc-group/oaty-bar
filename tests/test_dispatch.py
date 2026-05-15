import asyncio
import datetime as dt
import json
import tomllib
import urllib

import pytest
import pytest_asyncio
from dmax.data_storage import Experiment
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from oaty_bar._dispatch import OatyBarConfig, dispatch_new_runs, load_dm_apis
from oaty_bar.workflows import Workflow

WS_HOST = "127.0.0.1"
WS_PORT = 4832
WS_URL = f"ws://{WS_HOST}:{WS_PORT}"


# Borrowed from websockets tests
async def handler(ws):
    path = urllib.parse.urlparse(ws.request.path).path
    if path == "/":
        # The default path is an eval shell.
        async for expr in ws:
            value = eval(expr)
            await ws.send(str(value))
    elif path == "/crash":
        raise RuntimeError
    elif path == "/no-op":
        pass
    elif path == "/delay":
        delay = float(await ws.recv())
        await ws.close()
        await asyncio.sleep(delay)
    else:
        raise AssertionError(f"unexpected path: {path}")


@pytest_asyncio.fixture()
async def websocket():
    async with serve(handler, WS_HOST, WS_PORT) as server:
        async with connect(WS_URL) as client:
            yield client, server


@pytest.mark.asyncio
async def test_handle_message(websocket, mocker):
    client, server = websocket
    api = mocker.AsyncMock()
    api.username = "s255idzuser"
    api.experiment.return_value = Experiment(
        name="cabana-2026-C3",
        id="12345",
        primaryStorage={"name": "", "id": 0},
        experimentStation={"name": "", "id": 0},
        experimentType={"name": "", "id": 0},
        createDate=dt.datetime.now(),
        updateDate=dt.datetime.now(),
        startDate=dt.datetime.now(),
        endDate=dt.datetime.now(),
        dataDirectory="/tmp",
    )
    (connection,) = server.connections
    dispatched = asyncio.create_task(
        dispatch_new_runs(websocket=client, dm_apis={"255IDZ": api})
    )
    # First check that we don't do anything unless there's a stop document
    await connection.send(
        json.dumps(
            {
                "type": "container-child-metadata-updated",
                "key": "ABC123",
                "specs": [],
                "metadata": {"start": {}},
            }
        )
    )

    await asyncio.sleep(0.01)
    assert not api.send.called
    # Now check that we start a new workflow of some sort
    await connection.send(
        json.dumps(
            {
                "type": "container-child-metadata-updated",
                "key": "ABC123",
                "specs": [],
                "metadata": {
                    "start": {
                        "uid": "54321",
                        "dm_exp": "cabana-2026-C3",
                        "dm_station_name": "255IDZ",
                    },
                    "stop": {"exit_status": "success"},
                },
            }
        )
    )
    await asyncio.sleep(0.01)
    assert not api.submit_processing_job.assert_called_once_with(
        workflow="simple", run_uid="54321", target_folder="/tmp", filePath="/dev/null"
    )


config_toml = """
[tiled]
websocket_uri = "ws://localhost:8020"

[ data_management_stations.255IDZ ]
username = "cleese"
password = "secret"
station_name = "255IDZ"
"""


async def test_load_dm_apis(tmp_path):
    toml_file = tmp_path / "oaty_bar_config.toml"
    with open(toml_file, mode="a+") as fd:
        fd.write(config_toml)
    with open(toml_file, mode="rb") as fd:
        cfg_dict = tomllib.load(fd)
    config = OatyBarConfig(**cfg_dict)
    dm_apis = load_dm_apis(config)
    assert "255IDZ" in dm_apis


@pytest.mark.asyncio
async def test_add_workflow(websocket, mocker):
    """Adds a new workflow if the requested workflow doesn't exist."""
    client, server = websocket
    api = mocker.AsyncMock()
    api.username = "s255idzuser"
    api.experiment.return_value = Experiment(
        name="cabana-2026-C3",
        id="12345",
        primaryStorage={"name": "", "id": 0},
        experimentStation={"name": "", "id": 0},
        experimentType={"name": "", "id": 0},
        createDate=dt.datetime.now(),
        updateDate=dt.datetime.now(),
        startDate=dt.datetime.now(),
        endDate=dt.datetime.now(),
        dataDirectory="/tmp",
    )
    api.workflows.return_value = []
    (connection,) = server.connections
    dispatched = asyncio.create_task(
        dispatch_new_runs(websocket=client, dm_apis={"255IDZ": api})
    )
    # Now check that we start a new workflow of some sort
    await connection.send(
        json.dumps(
            {
                "type": "container-child-metadata-updated",
                "key": "ABC123",
                "specs": [],
                "metadata": {
                    "start": {
                        "uid": "54321",
                        "dm_exp": "cabana-2026-C3",
                        "dm_station_name": "255IDZ",
                    },
                    "stop": {"exit_status": "success"},
                },
            }
        )
    )
    await asyncio.sleep(0.01)
    api.add_workflow.assert_called_once()


@pytest.mark.asyncio
async def test_update_workflow(websocket, mocker):
    """Updates an existing workflow if the version number is higher."""
    client, server = websocket
    api = mocker.AsyncMock()
    api.username = "s255idzuser"
    api.experiment.return_value = Experiment(
        name="cabana-2026-C3",
        id="12345",
        primaryStorage={"name": "", "id": 0},
        experimentStation={"name": "", "id": 0},
        experimentType={"name": "", "id": 0},
        createDate=dt.datetime.now(),
        updateDate=dt.datetime.now(),
        startDate=dt.datetime.now(),
        endDate=dt.datetime.now(),
        dataDirectory="/tmp",
    )
    api.workflows.return_value = [
        Workflow(
            name="simple",
            version=0,
            owner="s255idzuser",
            userAccount="s255idzuser",
            description="",
            id="12345",
            stages={},
        ),
    ]
    (connection,) = server.connections
    dispatched = asyncio.create_task(
        dispatch_new_runs(websocket=client, dm_apis={"255IDZ": api})
    )
    # Now check that we start a new workflow of some sort
    await connection.send(
        json.dumps(
            {
                "type": "container-child-metadata-updated",
                "key": "ABC123",
                "specs": [],
                "metadata": {
                    "start": {
                        "uid": "54321",
                        "dm_exp": "cabana-2026-C3",
                        "dm_station_name": "255IDZ",
                    },
                    "stop": {"exit_status": "success"},
                },
            }
        )
    )
    await asyncio.sleep(0.01)
    assert not api.add_workflow.called
    api.set_workflow.assert_called_once()
