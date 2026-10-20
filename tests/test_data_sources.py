import asyncio
import json
from pathlib import Path

import pytest
from bluesky_tiled_plugins import TiledWriter

from oaty_bar._data_sources import bluesky_run_events


def scan_docs():
    scan_jsonl_file = Path(__file__).parent / "data" / "bluesky_scan_docs.jsonl"
    docs = []
    with open(scan_jsonl_file) as fd:
        for line in fd:
            name, doc = json.loads(line.strip())
            docs.append([name, doc])
    return docs


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_reads_websockets_from_tiled(tiled_catalog):
    catalog = tiled_catalog
    writer = TiledWriter(catalog)
    docs = scan_docs()

    # Define a background task to write documents like it was a scan
    async def write_docs():
        await asyncio.sleep(0.1)
        for name, doc in docs[1:]:
            await asyncio.sleep(0.1)
            writer(name, doc)
            print(f"Wrote {name} doc")

    writer(*(docs[0]))
    run = catalog["07d47cff-ff4d-48b9-8855-e82a090e149d"]
    pipe = bluesky_run_events(run=run)
    write_task = asyncio.create_task(write_docs())
    try:
        xarrs = [xarr async for xarr in pipe]
    finally:
        # Make sure the task completed so we don't deadlock
        await write_task
    # assert len(xarrs) > 0
