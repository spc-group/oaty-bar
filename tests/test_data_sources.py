import json
from pathlib import Path

from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri

from oaty_bar._data_sources import bluesky_stream_events


def add_sim_scan(catalog):
    writer = TiledWriter(catalog)
    scan_jsonl_file = Path(__file__).parent / "data" / "bluesky_scan_docs.jsonl"
    with open(scan_jsonl_file) as fd:
        for line in fd:
            name, doc = json.loads(line.strip())
            writer(name, doc)


async def test_reads_data_from_tiled(tiled_server):
    catalog = from_uri(tiled_server.uri, api_key=tiled_server.api_key)
    add_sim_scan(catalog)
    run = catalog["07d47cff-ff4d-48b9-8855-e82a090e149d"]
    events = bluesky_stream_events(stream=run["primary"])
    events = [ev async for ev in events]
    assert events[0]["stage-x"].shape == (9,)
