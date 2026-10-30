import asyncio
import json
from pathlib import Path, PurePath

import h5py
import numpy as np
import pytest
from bluesky_tiled_plugins import TiledWriter

from oaty_bar._data_sources import DataPipe, bluesky_run_events, datapipe

data_dir = Path(__file__).parent / "data"


def scan_docs(storage_path):
    scan_jsonl_file = Path(__file__).parent / "data" / "bluesky_scan_docs.jsonl"
    docs = []
    with open(scan_jsonl_file) as fd:
        for line in fd:
            name, doc = json.loads(line.strip())
            if name == "stream_resource":
                doc["uri"] = doc["uri"].format(storage_path=storage_path)
            docs.append([name, doc])
    return docs


def append_to_dataset(fd, path: PurePath, data):
    parent = path.parent
    group = fd.require_group(str(parent))
    if path.name in group:
        ds = group[path.name]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = data
    else:
        ds = group.create_dataset(
            path.name, data=data, maxshape=(None, *data.shape[1:])
        )
    return ds


async def write_docs(new_docs, writer):
    """A background task to write documents like it was a scan.

    Also copies data files in real-time to simulate data getting
    written.

    """
    resources = {}
    TICK = 0.1
    await asyncio.sleep(TICK * 10)
    for name, doc in new_docs:
        await asyncio.sleep(TICK)
        if name == "stream_resource":
            resources[doc["uid"]] = doc
        elif name == "stream_datum":
            # Copy this source frame to the destination
            resource = resources[doc["stream_resource"]]
            fp = Path(resource["uri"][16:])
            entry = PurePath(resource["parameters"]["dataset"])
            with h5py.File(data_dir / "bdet_data.h5", mode="r") as fd:
                # Read next source frame
                arr = fd[str(entry)]
                frame = arr[slice(doc["indices"]["start"], doc["indices"]["stop"])]
            with h5py.File(fp, mode="a") as fd:
                # Write to temporary HDF5 file
                append_to_dataset(fd, entry, frame)
        await asyncio.to_thread(writer, name, doc)


@pytest.mark.slow
@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_reads_websockets_from_tiled(tiled_catalog, tiled_server):
    catalog = tiled_catalog
    writer = TiledWriter(catalog, batch_size=1)
    docs = scan_docs(tiled_server.storage_path)
    await write_docs(docs[:1], writer)  # write 'start' doc to make the run
    start_doc_uid = docs[0][1]["uid"]
    run = catalog[start_doc_uid]
    pipe = bluesky_run_events(run=run)
    write_task = asyncio.create_task(write_docs(docs[1:], writer))
    try:
        xarrs = [xarr async for xarr in pipe]
    finally:
        # Make sure the task completed so we don't deadlock
        await write_task
    print(run["primary"].read())
    assert len(xarrs) == 9
    xarr = xarrs[0]
    assert "bdet" in xarr.keys()
    indices = [xarr.index.data for xarr in xarrs]
    print(indices)
    np.testing.assert_equal(indices, [0, 1, 2, 3, 4, 5, 6, 7, 8])


# Build a simple generator to use for the pipe tests
async def generate_numbers(numbers: list[int] = None):
    for num in range(10):
        if numbers is not None:
            numbers.append(num)
        yield num


@pytest.mark.asyncio
async def test_pipe_wrapper_is_iterable():
    # See if it's iterable
    numbers = []
    [_ async for _ in DataPipe(generate_numbers(numbers))]
    assert numbers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_pipe_wrapper_is_iterator():
    # See if it's iterable
    numbers = []
    iterator = DataPipe(generate_numbers(numbers))
    while True:
        try:
            await anext(iterator)
        except StopAsyncIteration:
            break
    assert numbers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_pipe_wrapper_is_awaitable():
    # See if the pipe can be awaited to consume the contents
    numbers = []
    await DataPipe(generate_numbers(numbers))
    assert numbers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_pipe_decorator_is_iterable():
    numbers = []
    [_ async for _ in datapipe(generate_numbers)(numbers)]
    assert numbers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_pipe_decorator_is_iterator():
    numbers = []
    iterator = datapipe(generate_numbers)(numbers)
    while True:
        try:
            await anext(iterator)
        except StopAsyncIteration:
            break
    assert numbers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_pipe_decorator_is_awaitable():
    # See if the pipe can be awaited to consume the contents
    numbers = []
    await datapipe(generate_numbers)(numbers)
    assert numbers == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_pipe_repr():
    data = DataPipe(generate_numbers())
    assert repr(data) == "<DataPipe: generate_numbers>"
