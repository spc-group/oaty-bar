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
import io
from asyncio import Queue
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any
from urllib.parse import urlencode

import msgpack
import pyarrow
import pyarrow.feather
import xarray as xr
import zarr.api.asynchronous as zarraio
from tiled.client.array import ArrayClient
from tiled.client.base import BaseClient
from tiled.client.composite import CompositeClient
from tiled.client.container import Container
from tiled.client.dataframe import DataFrameClient
from websockets import ConnectionClosed
from websockets.asyncio.client import connect


@dataclass()
class FetchItem:
    client: BaseClient
    path: PurePath
    which: slice | tuple[int]


@dataclass()
class WorkItem:
    """An item of work that should be done when possible.

    Parameters
    ==========
    data
      Known data that has already been fetched.
    to_fetch
      New data that should be retrieved.

    """

    data: xr.Dataset | None = None
    to_fetch: Mapping[str:FetchItem] = field(default_factory=dict)


async def node_websocket(run: Container) -> AsyncGenerator:
    base_uri = run.uri.split("/api/")[0]
    run_path = "/".join(run.path_parts)
    ws_uri = f"{base_uri}/api/v1/stream/single/{run_path}"
    scheme, rest = ws_uri.split("://")
    ws_uri = f"{scheme.replace('http', 'ws')}://{rest}"
    params = {"envelope_format": "msgpack"}
    ws_uri = f"{ws_uri}?{urlencode(params)}"
    async for websocket in connect(ws_uri):
        try:
            async for message in websocket:
                unpacked = msgpack.unpackb(message)
                yield unpacked
        except ConnectionClosed:
            continue
        except GeneratorExit:
            break


async def listen_to_array(client: ArrayClient, work_queue: Queue[WorkItem]) -> None:
    tasks = []
    schema = None
    data_name = client.path_parts[-1]
    path = PurePath("/".join(client.path_parts))
    async with asyncio.TaskGroup() as tg:
        async for message in node_websocket(client):
            if message["type"] == "array-ref":
                start = message["patch"]["offset"][0]
                num = message["patch"]["shape"][0]
                fetch_item = FetchItem(
                    client=client, path=path, which=slice(start, start + num)
                )
                work_item = WorkItem(data=None, to_fetch={data_name: fetch_item})
                await work_queue.put(work_item)


async def listen_to_dataframe(
    client: DataFrameClient, work_queue: Queue[WorkItem]
) -> None:
    tasks = []
    async with asyncio.TaskGroup() as tg:
        async for message in node_websocket(client):
            if message["type"] == "table-data":
                data = pyarrow.feather.read_feather(io.BytesIO(message["payload"]))
                data = xr.Dataset.from_dataframe(data)
                work_item = WorkItem(data=data, to_fetch={})
                await work_queue.put(work_item)
            # elif message['type'] == 'table-schema':
            #     schema_encoded = message['arrow_schema']
            #     b64_prefix = "data:application/vnd.apache.arrow.file;base64,"
            #     schema = pyarrow.ipc.read_schema(io.BytesIO(base64.b64decode(schema_encoded[len(b64_prefix):])))


async def listen_to_composite(client: CompositeClient, work_queue: Queue[WorkItem]):
    tasks = []
    # We need to intercept the work_queue so we can merge composite datasets
    sub_queue = Queue()

    async def link_queues(work_queue, sub_queue):
        while True:
            datum = await sub_queue.get()
            # print("Intercepted:", datum)
            await work_queue.put(datum)

    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(link_queues(work_queue, sub_queue)),
            tg.create_task(listen_to_container(client, sub_queue)),
        ]


async def listen_to_container(client: Container, work_queue: Queue):
    tasks = []
    async with asyncio.TaskGroup() as tg:
        async for message in node_websocket(client):
            if message["type"] == "container-child-created":
                # New child, defer to a specific listener to handle updates
                child = client.base[message["key"]]
                listeners = [
                    listener
                    for Klass, listener in LISTENERS.items()
                    if isinstance(child, Klass)
                ]
                try:
                    listener = listeners[0]
                except IndexError as exc:
                    raise KeyError(child.__class__.__mro__) from exc
                task = tg.create_task(listener(child, work_queue=work_queue))
                tasks.append(task)


LISTENERS = {
    DataFrameClient: listen_to_dataframe,
    ArrayClient: listen_to_array,
    CompositeClient: listen_to_composite,
    Container: listen_to_container,
}


async def wait_for_run_close(run, work_queue: Queue):
    """Check for a stop document, and close all the things."""
    websocket_messages = node_websocket(run.parent)
    async for message in websocket_messages:
        if message["type"] != "container-child-metadata-updated":
            continue
        md = message["metadata"]
        if "stop" in md.keys():
            work_queue.shutdown()
            break
    await websocket_messages.aclose()


async def fetch_items(
    uri: str, work_queue: Queue[WorkItem], data_queue: Queue[xr.Dataset]
) -> None:
    base_uri = uri.split("/api/")[0]
    zarr_uri = f"{base_uri}/zarr/v3/"
    group = await zarraio.open_group(
        store=zarr_uri,
        mode="r",
    )
    while True:
        try:
            work = await work_queue.get()
        except asyncio.QueueShutDown:
            break
        fetched_vars = {}
        fetched_coords = {}
        # Fetch the data
        for name, item in work.to_fetch.items():
            zarray = await group.getitem(str(item.path))
            arr = await zarray.getitem(item.which)
            if name in ["seq_num", "time"]:
                fetched_coords[name] = arr
            else:
                dims = ("seq_num", *[f"{name}-dim{idx}" for idx in range(1, arr.ndim)])
                fetched_vars[name] = (dims, arr)
        # Build a new xarray that combines the old and new data
        new_xarr = xr.Dataset(
            data_vars={
                **getattr(work.data, "data_vars", {}),
                **fetched_vars,
            },
            coords={
                **getattr(work.data, "coords", {}),
                **fetched_coords,
            },
            attrs=getattr(work.data, "attrs", {}),
        )
        await data_queue.put(new_xarr)
    data_queue.shutdown()


async def bluesky_run_events(run: Container):
    work_queue: Queue = Queue()
    data_queue: Queue = Queue()
    # Background task that puts new
    async with asyncio.TaskGroup() as tg:
        closing_task = tg.create_task(wait_for_run_close(run, work_queue=work_queue))
        listener_task = tg.create_task(listen_to_container(run, work_queue=work_queue))
        fetcher_task = tg.create_task(
            fetch_items(uri=run.uri, work_queue=work_queue, data_queue=data_queue)
        )
        while True:
            try:
                next_data = await data_queue.get()
            except asyncio.QueueShutDown:
                break
            yield next_data
        listener_task.cancel()
        fetcher_task.cancel()


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
