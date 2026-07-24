import asyncio
import json
import urllib

import pytest
import pytest_asyncio
from prefect.testing.fixtures import asserting_events_worker
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from oaty_bar._dispatch import dispatch_new_runs, process_msg

# Make sure isort doesn't remove the fixture from imports
assert asserting_events_worker

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
async def test_ignores_new_run():
    """No events should be emitted without a stop document."""
    # Check that we don't do anything unless there's a stop document
    events = process_msg(
        json.dumps(
            {
                "type": "container-child-metadata-updated",
                "key": "ABC123",
                "specs": [],
                "metadata": {"start": {}},
            }
        ),
        instance_uuid="123456",
    )
    assert len(list(events)) == 0


@pytest.mark.asyncio
async def test_emits_resource():
    events = process_msg(
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
                        "beamline_id": "255-ID-Z",
                    },
                    "stop": {"exit_status": "success"},
                },
            }
        ),
        instance_uuid="123456",
    )
    start_event, success_event = events
    assert start_event.resource == {
        "prefect.resource.id": "oaty-bar.dispatcher.123456",
        "bluesky.run.uid": "54321",
        "aps.beamline.id": "255-ID-Z",
    }
    assert start_event.payload == {
        "type": "container-child-metadata-updated",
        "key": "ABC123",
        "specs": [],
        "metadata": {
            "start": {
                "uid": "54321",
                "dm_exp": "cabana-2026-C3",
                "dm_station_name": "255IDZ",
                "beamline_id": "255-ID-Z",
            },
            "stop": {"exit_status": "success"},
        },
    }


@pytest.mark.asyncio
async def test_dispatch_emits_events(
    websocket, mocker, prefect_server, asserting_events_worker
):
    client, server = websocket
    (connection,) = server.connections
    dispatched = asyncio.create_task(dispatch_new_runs(websocket=client))
    # Check that we emit the correct event when created
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
    asserting_events_worker.drain()
    events = asserting_events_worker._client.events
    event_names = {ev.event for ev in asserting_events_worker._client.events}
    assert event_names == {"bluesky.run.stopped", "bluesky.run.success"}
