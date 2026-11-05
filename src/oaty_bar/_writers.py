from collections.abc import AsyncIterator
from functools import partial
from uuid import uuid7 as new_uuid

import pyarrow
import xarray as xr
from anyio import to_thread
from tiled.client.container import Container
from tiled.structures.core import Spec

from ._data_sources import datapipe


@datapipe
async def write_results(data: AsyncIterator[xr.Dataset], catalog: Container):
    """Write Xarray result data to a Tiled results catalog."""
    containers = {}
    async for dataset in data:
        source_client = dataset.attrs["source_client"]
        source_path = "/".join(source_client.path_parts)
        # Create a new composite container to hold the results if needed
        if source_path not in containers:
            metadata = {
                "uid": new_uuid(),
                "run_start": source_client.metadata["run_start"],
            }
            create_container = partial(
                catalog.create_container,
                metadata["uid"],
                metadata=metadata,
                specs=[Spec("composite")],
            )
            containers[source_path] = await to_thread.run_sync(create_container)
        results = containers[source_path]
        # Figure out what kinds of signals we need to store
        array_keys = [key for key, arr in dataset.data_vars.items() if arr.ndim > 1]
        internal_keys = [
            key for key in dataset.data_vars.keys() if key not in array_keys
        ]
        # Write arrays to the server
        index = int(dataset["index"].data[0])
        for key in array_keys:
            array = dataset[key].data
            if key not in results:
                # New array
                write_array = partial(results.write_array, array, key=key)
                await to_thread.run_sync(write_array)
            else:
                # Add to existing array
                patch_array = partial(
                    results[key].patch, array, offset=index, extend=True
                )
                await to_thread.run_sync(patch_array)
        # Write tabular data to the server
        table = pyarrow.Table.from_pandas(dataset[internal_keys].to_dataframe())
        if "internal" not in results.base.keys():
            create_table = partial(
                results.create_appendable_table, schema=table.schema, key="internal"
            )
            await to_thread.run_sync(create_table)
        append_partition = partial(results.base["internal"].append_partition, 0, table)
        await to_thread.run_sync(append_partition)
        # All done with the writing
        dataset.attrs["results_client"] = results
        yield dataset
