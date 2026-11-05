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
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from functools import wraps
from pathlib import PurePath
from urllib.parse import urlencode

import anyio
import msgpack
import numpy as np
import pyarrow
import pyarrow.feather
import xarray as xr
import zarr.api.asynchronous as zarraio
from anyio import TaskHandle, create_task_group
from anyio.abc import ObjectSendStream
from tiled.client.array import ArrayClient
from tiled.client.composite import CompositeClient
from tiled.client.container import Container
from tiled.client.dataframe import DataFrameClient
from websockets.asyncio.client import connect


@dataclass()
class DataPipe:
    """A wrapper around an operation in a pipeline that makes it awaitable.

    Instances of this class can be awaited to consume the asynchronous
    iterator they wrap. Instances can also be iterated (e.g. ``async
    for data in data_pipe: ...``), and will return the wrapper
    iterator.

    """

    iterator: AsyncIterator
    """The wrapped item that will be consumed if the pipe is awaited
    or iterated."""

    def __await__(self):
        async def inner():
            async for _ in self.iterator:
                pass

        yield from inner().__await__()

    def __aiter__(self):
        return self.iterator.__aiter__()

    def __anext__(self):
        return self.iterator.__anext__()

    def __repr__(self):
        class_name = self.__class__.__name__
        generator_name = self.iterator.__name__
        return f"<{class_name}: {generator_name}>"


def datapipe(func):
    """A decorator that turns a generator function into a data pipe.

    ```python
    @datapipe
    async def my_pipe(data):
        yield from data

    input_data = data_source(…)
    await my_pipe(input_data)  # ← causes pipe to run
    ```

    …is equivalent to…

    ```python
    async def my_pipe(data):
        yield from data

    input_data = data_source(…)
    input_data = my_pipe(input_data)
    pipe = DataPipe(input_data)
    await pipe  # ← causes pipe to run
    """

    @wraps(func)
    def inner(*args, **kwargs):
        return DataPipe(func(*args, **kwargs))

    return inner


def node_websocket(run: Container) -> AsyncGenerator:
    base_uri = run.uri.split("/api/")[0]
    run_path = "/".join(run.path_parts)
    ws_uri = f"{base_uri}/api/v1/stream/single/{run_path}"
    scheme, rest = ws_uri.split("://")
    ws_uri = f"{scheme.replace('http', 'ws')}://{rest}"
    params = {"envelope_format": "msgpack"}
    ws_uri = f"{ws_uri}?{urlencode(params)}"
    return connect(ws_uri)


async def node_messages(websocket):
    try:
        async for message in websocket:
            unpacked = msgpack.unpackb(message)
            yield unpacked
    except GeneratorExit:
        pass


async def fetch_array(client, selection: slice | None = None):
    zarray = await zarr_array(client)
    name = zarray.name.strip("/")
    if selection is None:
        selection = slice(zarray.shape[0])
    # Get the array from the API
    arr = await zarray.getitem(selection)
    if selection is None:
        index = np.arange(arr.shape[0])
    else:
        index = np.mgrid[selection]
    # Turn it into the standard xarray structure
    dims = ("index", *[f"{name}-dim{idx}" for idx in range(1, arr.ndim)])
    xarr = xr.Dataset(
        data_vars={
            name: (dims, arr),
        },
        coords={"index": index},
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


async def listen_to_array(client: ArrayClient, queue: FetchQueue) -> None:
    tasks = []
    schema = None
    data_name = client.path_parts[-1]
    async with node_websocket(client) as websocket:
        # Fetch any data that's already on disk
        await queue.start_rest_task(fetch_array, client)
        # Start streaming new data
        async for message in node_messages(websocket):
            if message["type"] == "array-ref":
                start = message["patch"]["offset"][0]
                num = message["patch"]["shape"][0]
                await queue.start_live_task(
                    fetch_array, client, slice(start, start + num)
                )


async def as_is(data):
    """A dummy coroutine that just returns existing data.

    This lets us create tasks that actually don't have to do any real
    awaiting.

    """
    return data


async def listen_to_dataframe(client: DataFrameClient, queue: FetchQueue) -> None:
    starting_index = 0

    async def fetch_initial_data(client):
        df = await anyio.to_thread.run_sync(client.read)
        data = xr.Dataset.from_dataframe(df)
        return data

    async with node_websocket(client) as websocket:
        # Fetch any array that was already written
        await queue.start_rest_task(fetch_initial_data, client)
        # Wait for new data
        async for message in node_messages(websocket):
            if message["type"] == "table-data":
                data = pyarrow.feather.read_feather(io.BytesIO(message["payload"]))
                data = xr.Dataset.from_dataframe(data)
                data["index"] = [
                    starting_index + message["sequence"] - 1
                ]  # 1-indexed to 0-indexed
                await queue.start_live_task(as_is, data)


async def listen_to_composite(client: CompositeClient, queue: FetchQueue):
    expected_keys = client.metadata["data_keys"].keys()
    # We need to intercept the work_queue so we can merge composite datasets
    sub_queue = FetchQueue(max_tasks=queue.max_tasks)
    queue.start_soon(listen_to_container, client, sub_queue)
    dataset = xr.Dataset()
    have_seen = {}
    next_index = 0
    async with sub_queue:
        async for task in sub_queue:
            new_data = await task
            t0 = time.perf_counter()
            # Combine the new data with the previous cached data
            dataset = xr.combine_by_coords(
                [dataset, new_data],
                coords=["index"],
                compat="no_conflicts",
                join="outer",
            )
            awaiting_sigs = [
                have_seen.setdefault(int(idx), set(expected_keys))
                for idx in new_data["index"].data
            ]
            # Determine if we have any new complete arrays to produce
            for sigs in awaiting_sigs:
                sigs -= set(new_data.keys())
            ready_indices = [
                idx
                for idx, awaiting_sigs in have_seen.items()
                if len(awaiting_sigs) == 0
            ]
            merge_indices = []
            while next_index in ready_indices:
                merge_indices.append(next_index)
                del have_seen[next_index]
                next_index += 1
            if not merge_indices:
                continue
            merged_data = dataset.sel(index=merge_indices)
            dataset = dataset.drop_sel(index=merge_indices)
            await queue.start_live_task(as_is, merged_data)


async def listen_to_container(container: Container, queue: FetchQueue):
    async with node_websocket(container) as websocket:
        async for message in node_messages(websocket):
            if message["type"] == "container-child-created":
                # New child, defer to a specific listener to handle updates
                child = container.base[message["key"]]
                listeners = [
                    listener
                    for Klass, listener in LISTENERS.items()
                    if isinstance(child, Klass)
                ]
                try:
                    listener = listeners[0]
                except IndexError as exc:
                    raise KeyError(child.__class__.__mro__) from exc
                queue.start_soon(listener, child, queue)


LISTENERS = {
    DataFrameClient: listen_to_dataframe,
    ArrayClient: listen_to_array,
    CompositeClient: listen_to_composite,
    Container: listen_to_container,
}


async def wait_for_run_close(run, sender: ObjectSendStream[TaskHandle]):
    """Check for a stop document, and close all the things."""
    async with node_websocket(run.parent) as websocket:
        async for message in node_messages(websocket):
            if message["type"] != "container-child-metadata-updated":
                continue
            md = message["metadata"]
            if "stop" in md.keys():
                # Let other tasks finish up, then shut it all down
                await anyio.sleep(10)
                sender.close()
                break


@dataclass
class FetchQueue:
    max_tasks: int

    async def __aenter__(self):
        self._rest_sender, self._rest_receiver = anyio.create_memory_object_stream(
            max_buffer_size=self.max_tasks
        )
        self._live_sender, self._live_receiver = anyio.create_memory_object_stream(
            max_buffer_size=self.max_tasks
        )
        self.tg = create_task_group()
        async with create_task_group() as tg:
            tg.start_soon(self._rest_sender.__aenter__)
            tg.start_soon(self._rest_receiver.__aenter__)
            tg.start_soon(self._live_sender.__aenter__)
            tg.start_soon(self._live_receiver.__aenter__)
        await self.tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.tg.cancel()
        await self.tg.__aexit__(exc_type, exc_val, exc_tb)
        async with create_task_group() as tg:
            tg.start_soon(self._live_receiver.__aexit__, exc_type, exc_val, exc_tb)
            tg.start_soon(self._live_sender.__aexit__, exc_type, exc_val, exc_tb)
            tg.start_soon(self._rest_receiver.__aexit__, exc_type, exc_val, exc_tb)
            tg.start_soon(self._rest_sender.__aexit__, exc_type, exc_val, exc_tb)

    async def __aiter__(self):
        async for task in self._live_receiver:
            yield task

    async def start_live_task(self, func, *args, name=None):
        task = self.start_soon(func, *args, name=None)
        await self._live_sender.send(task)
        return task

    async def start_rest_task(self, func, *args, name=None):
        task = self.start_soon(func, *args, name=None)
        await self._live_sender.send(task)
        return task

    def start_soon(self, func, *args, name=None):
        return self.tg.start_soon(func, *args, name=name)


@datapipe
async def bluesky_run_events(run: Container, max_tasks: int = 100) -> AsyncIterator:
    """Generates xarrays with events added to a single Bluesky run.

    The generator will yield xarrays of new data until the stop
    document is written, then it will exit.

    Parameters
    ==========
    run
      The run to stream data from.
    max_tasks
      How many concurrent fetch tasks can be running at a time.

    """
    # Queues hold fetch-tasks
    # async with create_task_group() as tg, receiver, sender:
    async with FetchQueue(max_tasks) as queue:
        closing_task = queue.tg.start_soon(wait_for_run_close, run, queue._live_sender)
        listener_task = queue.tg.start_soon(listen_to_container, run, queue)
        # Wait for new data to be available so we can generate it
        async for next_task in queue:
            data = await next_task
            # To-do: make sure we avoid cancel scope stack corruption:
            # https://anyio.readthedocs.io/en/stable/cancellation.html#avoiding-cancel-scope-stack-corruption
            yield data
