"""Tools for reading data events from somewhere.

Data sources should be usable like…

```python
async def process_data(…):
    # Start of the pipeline
    events = my_data_source(…)
    # Operations in the pipeline
    events = apply_correction(events)
    events = some_other_operation(events)
    # Terminal of the pipeline
    await write_data(events)
```

"""

import asyncio
import json
from asyncio import Queue
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import xarray as xr
import zarr.api.asynchronous as zarraio
from tiled.client.base import BaseClient
from tiled.client.container import Container
from websockets import ConnectionClosed
from websockets.asyncio.client import connect


def websocket_uri(node: BaseClient):
    base_uri = node.uri.split("/api/")[0]
    node_path = "/".join(node.path_parts)
    ws_uri = f"{base_uri}/api/v1/stream/single/{node_path}"
    scheme, rest = ws_uri.split("://")
    ws_uri = f"{scheme.replace('http', 'ws')}://{rest}"
    return ws_uri


async def node_websocket(run: Container) -> AsyncGenerator:
    base_uri = run.uri.split("/api/")[0]
    run_path = "/".join(run.path_parts)
    ws_uri = f"{base_uri}/api/v1/stream/single/{run_path}"
    scheme, rest = ws_uri.split("://")
    ws_uri = f"{scheme.replace('http', 'ws')}://{rest}"
    async for websocket in connect(ws_uri):
        try:
            async for message in websocket:
                yield message
        except ConnectionClosed:
            continue
        except GeneratorExit:
            break


async def listen_to_container(container: Container, work_queue: Queue):
    # for name, stream in container.items():

    # Listens for additional data to be written
    async for message in node_websocket(container):
        print(message)


async def wait_for_run_close(run, work_queue: Queue):
    """Check for a stop document, and close all the things."""
    websocket_messages = node_websocket(run.parent)
    # ws_uri = websocket_uri(run.parent)
    # async with connect(ws_uri) as websocket:
    #     async for message_json in websocket:
    async for message_json in websocket_messages:
        message = json.loads(message_json)
        if message["type"] != "container-child-metadata-updated":
            continue
        md = message["metadata"]
        if "stop" in md.keys():
            work_queue.shutdown()
            break
    await websocket_messages.aclose()


async def bluesky_run_events(run: Container):
    work_queue: Queue = Queue()
    # Background task that puts new
    listener_task = asyncio.create_task(wait_for_run_close(run, work_queue=work_queue))
    while True:
        try:
            next_data = await work_queue.get()
        except asyncio.QueueShutDown:
            break
        yield next_data
    await listener_task


async def bluesky_stream_events_deprecated(
    stream: Container, signals: Sequence[str] | None = None
) -> AsyncGenerator[xr.Dataset, Any]:
    if signals is None:
        signals = stream.keys()
    base_uri = stream.uri.split("/api/")[0]
    stream_path = "/".join(stream.path_parts)
    zarr_uri = f"{base_uri}/zarr/v3/{stream_path}"
    group = await zarraio.open_group(
        store=zarr_uri,
        mode="r",
    )
    # Sort out which arrays we want to fetch
    data_keys = stream.metadata["data_keys"]
    data_keys.setdefault("seq_num", "internal/seq_num")
    if signals is not None:
        data_keys = {
            sig: data_key for sig, data_key in data_keys.items() if sig in signals
        }
    paths = {
        sig: (f"internal/{sig}" if "external" not in data_key else sig)
        for sig, data_key in data_keys.items()
    }

    # Do the fetching
    async def data_arrays(signal, path):
        zarray = await group.getitem(path)
        arr = await zarray.getitem(slice(None))
        return (signal, arr)

    zarray_items = await asyncio.gather(
        *[data_arrays(sig, path) for sig, path in paths.items()]
    )
    zarrays = {sig: arr for sig, arr in zarray_items}
    if zarrays != {}:
        ds = xr.Dataset(
            data_vars={
                sig: ("seq_num", arr)
                for sig, arr in zarrays.items()
                if sig != "seq_num"
            },
            coords={"seq_num": zarrays["seq_num"]} if "seq_num" in zarrays else {},
        )
        yield ds
