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

import io
from asyncio import Queue, Task
from collections.abc import AsyncGenerator
from pathlib import PurePath
from urllib.parse import urlencode

import anyio
import msgpack
import pyarrow
import pyarrow.feather
import xarray as xr
import zarr.api.asynchronous as zarraio
from anyio import TaskHandle, create_task_group
from anyio.abc import ObjectSendStream, TaskGroup
from tiled.client.array import ArrayClient
from tiled.client.composite import CompositeClient
from tiled.client.container import Container
from tiled.client.dataframe import DataFrameClient
from websockets import ConnectionClosed
from websockets.asyncio.client import connect

# @dataclass()
# class FetchItem:
#     client: BaseClient
#     path: PurePath
#     which: slice | tuple[int]


# @dataclass()
# class WorkItem:
#     """An item of work that should be done when possible.

#     Parameters
#     ==========
#     data
#       Known data that has already been fetched.
#     to_fetch
#       New data that should be retrieved.

#     """

#     data: xr.Dataset | None = None
#     to_fetch: Mapping[str:FetchItem] = field(default_factory=dict)


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


async def fetch_array(zarray, selection: slice | tuple[int]):
    name = zarray.name.strip("/")
    # Get the array from the API
    arr = await zarray.getitem(selection)
    # Turn it into the standard xarray structure
    dims = ("seq_num", *[f"{name}-dim{idx}" for idx in range(1, arr.ndim)])
    xarr = xr.Dataset(
        data_vars={
            name: (dims, arr),
        },
        coords={},
        attrs={},
    )
    return xarr


async def zarr_array(client):
    path = PurePath("/".join(client.path_parts))
    base_uri = client.uri.split("/api/")[0]
    zarr_uri = f"{base_uri}/zarr/v3/{path.parent}"
    zgroup = await zarraio.open_group(
        store=zarr_uri,
        mode="r",
    )
    arr = await zgroup.getitem(client.path_parts[-1])
    return arr


async def listen_to_array(
    client: ArrayClient, sender: ObjectSendStream[TaskHandle], tg: TaskGroup
) -> None:
    tasks = []
    schema = None
    data_name = client.path_parts[-1]
    zarr = await zarr_array(client)
    async for message in node_websocket(client):
        if message["type"] == "array-ref":
            start = message["patch"]["offset"][0]
            num = message["patch"]["shape"][0]
            task = tg.create_task(fetch_array(zarr, slice(start, start + num)))
            await sender.send(task)


async def existing_data(data):
    """A dummy coroutine that just returns existing data.

    This lets us create tasks that actually don't have to do any real
    awaiting.

    """
    return data


async def listen_to_dataframe(
    client: DataFrameClient, sender: ObjectSendStream[TaskHandle], tg: TaskGroup
) -> None:
    tasks = []
    async for message in node_websocket(client):
        if message["type"] == "table-data":
            data = pyarrow.feather.read_feather(io.BytesIO(message["payload"]))
            data = xr.Dataset.from_dataframe(data)
            new_task = tg.create_task(existing_data(data))
            await sender.send(new_task)
        # elif message['type'] == 'table-schema':
        #     schema_encoded = message['arrow_schema']
        #     b64_prefix = "data:application/vnd.apache.arrow.file;base64,"
        #     schema = pyarrow.ipc.read_schema(io.BytesIO(base64.b64decode(schema_encoded[len(b64_prefix):])))


async def listen_to_composite(
    client: CompositeClient, sender: ObjectSendStream[TaskHandle], tg: TaskGroup
):
    tasks = []
    # We need to intercept the work_queue so we can merge composite datasets
    expected_keys = client.metadata["data_keys"].keys()
    sub_queue: Queue[Task] = Queue()

    async def link_queues(sender, sub_queue):
        while True:
            datum = await sub_queue.get()
            await sender.send(datum)

    tg.create_task(link_queues(sender, sub_queue)),
    tg.create_task(listen_to_container(client, sender=sender, tg=tg)),


async def listen_to_container(
    client: Container, sender: ObjectSendStream[TaskHandle], tg: TaskGroup
):
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
            tg.start_soon(listener, child, sender, tg)


LISTENERS = {
    DataFrameClient: listen_to_dataframe,
    ArrayClient: listen_to_array,
    CompositeClient: listen_to_composite,
    Container: listen_to_container,
}


async def wait_for_run_close(run, sender: ObjectSendStream[TaskHandle]):
    """Check for a stop document, and close all the things."""
    websocket_messages = node_websocket(run.parent)
    async for message in websocket_messages:
        if message["type"] != "container-child-metadata-updated":
            continue
        md = message["metadata"]
        if "stop" in md.keys():
            sender.close()
            break
    await websocket_messages.aclose()


# async def fetch_items(
#     uri: str, work_queue: Queue[Task], data_queue: Queue[xr.Dataset]
# ) -> None:
#     while True:
#         try:
#             work = await work_queue.get()
#         except asyncio.QueueShutDown:
#             break
#         fetched_vars = {}
#         fetched_coords = {}
#         # Fetch the data
#         # for name, item in work.to_fetch.items():
#         #     zarray = await group.getitem(str(item.path))
#         #     arr = await zarray.getitem(item.which)
#         #     if name in ["seq_num", "time"]:
#         #         fetched_coords[name] = arr
#         #     else:
#         #         dims = ("seq_num", *[f"{name}-dim{idx}" for idx in range(1, arr.ndim)])
#         #         fetched_vars[name] = (dims, arr)
#         # Build a new xarray that combines the old and new data
#         # new_xarr = xr.Dataset(
#         #     data_vars={
#         #         **getattr(work.data, "data_vars", {}),
#         #         **fetched_vars,
#         #     },
#         #     coords={
#         #         **getattr(work.data, "coords", {}),
#         #         **fetched_coords,
#         #     },
#         #     attrs=getattr(work.data, "attrs", {}),
#         # )
#         await data_queue.put(work)
#     data_queue.shutdown()


async def bluesky_run_events(run: Container, max_tasks: int = 100):
    # Queues hold fetch-tasks and another for the results of that work
    sender, receiver = anyio.create_memory_object_stream(max_buffer_size=max_tasks)
    async with create_task_group() as tg, receiver, sender:
        closing_task = tg.start_soon(wait_for_run_close, run, sender)
        listener_task = tg.start_soon(listen_to_container, run, sender, tg)
        # Wait for new data to be available so we can generate it
        async for next_task in receiver:
            data = await next_task
            yield data
        tg.cancel()


# async def bluesky_stream_events_deprecated(
#     stream: Container, signals: Sequence[str] | None = None
# ) -> AsyncGenerator[xr.Dataset, Any]:
#     if signals is None:
#         signals = stream.keys()
#     base_uri = stream.uri.split("/api/")[0]
#     stream_path = "/".join(stream.path_parts)
#     zarr_uri = f"{base_uri}/zarr/v3/{stream_path}"
#     group = await zarraio.open_group(
#         store=zarr_uri,
#         mode="r",
#     )
#     # Sort out which arrays we want to fetch
#     data_keys = stream.metadata["data_keys"]
#     data_keys.setdefault("seq_num", "internal/seq_num")
#     if signals is not None:
#         data_keys = {
#             sig: data_key for sig, data_key in data_keys.items() if sig in signals
#         }
#     paths = {
#         sig: (f"internal/{sig}" if "external" not in data_key else sig)
#         for sig, data_key in data_keys.items()
#     }

#     # Do the fetching
#     async def data_arrays(signal, path):
#         zarray = await group.getitem(path)
#         arr = await zarray.getitem(slice(None))
#         return (signal, arr)

#     zarray_items = await asyncio.gather(
#         *[data_arrays(sig, path) for sig, path in paths.items()]
#     )
#     zarrays = {sig: arr for sig, arr in zarray_items}
#     if zarrays != {}:
#         ds = xr.Dataset(
#             data_vars={
#                 sig: ("seq_num", arr)
#                 for sig, arr in zarrays.items()
#                 if sig != "seq_num"
#             },
#             coords={"seq_num": zarrays["seq_num"]} if "seq_num" in zarrays else {},
#         )
#         yield ds
