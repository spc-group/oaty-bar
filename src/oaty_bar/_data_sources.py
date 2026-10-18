import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import xarray as xr
import zarr.api.asynchronous as zarraio
from tiled.client.container import Container


async def bluesky_stream_events(
    stream: Container, signals: Sequence[str] | None = None
) -> AsyncGenerator[xr.Dataset, Any]:
    if signals is None:
        signals = stream.keys()
    base_uri = stream.uri.split("/api/")[0]
    zarr_uri = f"{base_uri}/zarr/v3/{'/'.join(stream.path_parts)}"
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
    ds = xr.Dataset(
        data_vars={
            sig: ("seq_num", arr) for sig, arr in zarrays.items() if sig != "seq_num"
        },
        coords={"seq_num": zarrays["seq_num"]},
    )
    yield ds
