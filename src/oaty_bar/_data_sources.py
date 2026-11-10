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
from anyio import ClosedResourceError, EndOfStream, fail_after
from anyio.abc import TaskGroup
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
        await queue.start_soon(fetch_array, client, None)
        # Start streaming new data
        async for message in node_messages(websocket):
            if message["type"] == "array-ref":
                start = message["patch"]["offset"][0]
                num = message["patch"]["shape"][0]
                print(message)
                await queue.start_soon(fetch_array, client, slice(start, start + num))


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
        await queue.start_soon(fetch_initial_data, client)
        # Wait for new data
        async for message in node_messages(websocket):
            if message["type"] == "table-data":
                data = pyarrow.feather.read_feather(io.BytesIO(message["payload"]))
                data = xr.Dataset.from_dataframe(data)
                data["index"] = [
                    starting_index + message["sequence"] - 1
                ]  # 1-indexed to 0-indexed
                await queue.start_soon(as_is, data)


async def listen_to_composite(client: CompositeClient, queue: FetchQueue):
    expected_keys = client.metadata["data_keys"].keys()
    # We need to intercept the work_queue so we can merge composite datasets
    sub_queue = FetchQueue(max_tasks=queue.max_tasks, task_group=queue.task_group)
    queue.task_group.start_soon(listen_to_container, client, sub_queue)
    dataset = xr.Dataset()
    have_seen = {}
    next_index = 0
    async with sub_queue:
        async for task in sub_queue:
            new_data = await task
            t0 = time.perf_counter()
            # Combine the new data with the previous cached data
            print("^^^")
            print(dataset)
            print(new_data)
            print("$$$")
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
            await queue.start_soon(as_is, merged_data)


async def listen_to_container(container: Container, queue: FetchQueue):
    def start_child_listener(child):
        listeners = [
            listener
            for Klass, listener in LISTENERS.items()
            if isinstance(child, Klass)
        ]
        try:
            listener = listeners[0]
        except IndexError as exc:
            raise KeyError(child.__class__.__mro__) from exc
        queue.task_group.start_soon(listener, child, queue)

    async with node_websocket(container) as websocket:
        for child in container.base.values():
            start_child_listener(child)
        async for message in node_messages(websocket):
            if message["type"] == "container-child-created":
                # New child, defer to a specific listener to handle updates
                child = container.base[message["key"]]
                start_child_listener(child)


LISTENERS = {
    DataFrameClient: listen_to_dataframe,
    ArrayClient: listen_to_array,
    CompositeClient: listen_to_composite,
    Container: listen_to_container,
}


async def wait_for_run_close(run, queue: FetchQueue):
    """Check for a stop document, and then stop live streaming."""
    if "stop" in run.metadata.keys():
        # The live_sender won't get any data
        queue.close_soon()
        return
    async with node_websocket(run.parent) as websocket:
        async for message in node_messages(websocket):
            if message["type"] != "container-child-metadata-updated":
                continue
            md = message["metadata"]
            if "stop" in md.keys():
                print("STOPPING")
                # Let other tasks finish up, then shut it all down
                await anyio.sleep(4)
                queue.close_soon()
                break


class FetchQueue:
    """A two-part queue for passing asynchronous tasks.

    The **rest queue** is expected to be closed by listeners when they
    have submitted all the tasks for data that are currently at-rest.

    The **live queue** should be closed once there is no longer an
    expectation that any more data will be written (e.g. after a
    Bluesky run writes a "stop" document).

    Iterating over this queue will produce first the *rest* tasks,
    then the *live* tasks.

    """

    max_tasks: int
    _will_close_soon: bool = False
    task_group: TaskGroup

    def __init__(self, max_tasks: int, task_group: TaskGroup = None):
        self.max_tasks = max_tasks
        self._task_group = task_group

    async def __aenter__(self):
        # Create the ends of the queue
        self.sender, self.receiver = anyio.create_memory_object_stream(
            max_buffer_size=self.max_tasks
        )
        await self.sender.__aenter__()
        await self.receiver.__aenter__()
        # Create a task group to use for managing submitted tasks
        if self._task_group is None:
            self.task_group = anyio.create_task_group()
            await self.task_group.__aenter__()
        else:
            self.task_group = self._task_group
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._task_group is None:
            self.task_group.cancel()
            await self.task_group.__aexit__(exc_type, exc_val, exc_tb)
        await self.sender.__aexit__(exc_type, exc_val, exc_tb)
        await self.receiver.__aexit__(exc_type, exc_val, exc_tb)

    async def __aiter__(self):
        while True:
            try:
                with fail_after(1) as scope:
                    task = await self.receiver.receive()
            except ClosedResourceError, EndOfStream:
                print("BREAKING")
                # The queue was closed, guess we're done
                break
            except TimeoutError:
                qstats = self.receiver.statistics()
                print(self._will_close_soon, self)
                # print(qstats)
                if (
                    self._will_close_soon
                    and qstats.current_buffer_used == 0
                    and qstats.tasks_waiting_receive == 0
                ):
                    # We're most likely done, so clean up
                    self.sender.close()
                    break
                continue
            yield task
        print("CLOSED", self)

    def close_soon(self):
        print("CLOSING", self)
        self._will_close_soon = True

    async def start_soon(self, func, *args, name=None):
        task = self.task_group.start_soon(func, *args, name=None)
        await self.sender.send(task)
        return task


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
    # Outer stream avoids cancel-scope stack corruption:
    # https://anyio.readthedocs.io/en/stable/cancellation.html#avoiding-cancel-scope-stac
    send_stream, receive_stream = anyio.create_memory_object_stream(
        max_buffer_size=max_tasks
    )
    async with receive_stream:
        async with anyio.create_task_group() as tg:

            async def _produce():
                async with send_stream, FetchQueue(max_tasks, task_group=tg) as queue:
                    closing_task = tg.start_soon(wait_for_run_close, run, queue)
                    listener_task = tg.start_soon(listen_to_container, run, queue)
                    async for next_task in queue:
                        data = await next_task
                        await send_stream.send(data)

            tg.start_soon(_produce)
            async for data in receive_stream:
                yield data
            tg.cancel()
