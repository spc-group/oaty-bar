import asyncio
import datetime as dt
import json
import logging
import re
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import IO, Any

import h5py
import numpy as np
from prefect import flow, task
from tiled.client import from_profile
from tiled.client.container import Container
from tiled.client.dataframe import DataFrameClient
from tiled.queries import Eq
from tiled.server.schemas import DataSource
from tiled.utils import SerializationError
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger

log = logging.getLogger("oaty-bar")


def nxgroup(
    parent: h5py.Group, name: str, nx_class: str | None = None, exist_ok: bool = False
) -> h5py.Group:
    if name in parent and exist_ok:
        return parent[name]
    # Need to create a new group
    group = parent.create_group(name)
    if nx_class is not None:
        group.attrs["NX_class"] = nx_class
    return group


def nxentry(parent: h5py.Group, name: str) -> h5py.Group:
    return nxgroup(parent=parent, name=name, nx_class="NXentry")


def nxdata(parent: h5py.Group, name: str) -> h5py.Group:
    return nxgroup(parent=parent, name=name, nx_class="NXdata")


def nxinstrument(parent: h5py.Group, name: str) -> h5py.Group:
    return nxgroup(parent=parent, name=name, nx_class="NXinstrument")


def nxnote(parent: h5py.Group, name: str, exist_ok: bool = False) -> h5py.Group:
    return nxgroup(parent=parent, name=name, nx_class="NXnote", exist_ok=exist_ok)


def nxfield(
    parent: h5py.Group,
    name: str,
    value=None,
    shape=None,
    dtype=None,
    compression=None,
    chunks=None,
) -> h5py.Dataset:
    field = parent.create_dataset(
        name,
        data=value,
        shape=shape,
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )
    return field


def nxlink(parent: h5py.Group, name: str, target: h5py.Group | str, soft=False):
    """Create a link between datasets within the same file."""
    if soft:
        target_name = target
        link = h5py.SoftLink(target_name)
    else:
        target_name = getattr(target, "name", target)
        link = target
    parent[name] = link
    # Add metadata attrs
    try:
        parent[name].attrs["target"] = target_name
    except KeyError:
        # Most likely this is a soft link to an open dataset, but in
        # case it's not…
        if not soft:
            raise


def nxexternallink(
    parent: h5py.Group, name: str, target: str | Sequence[str], filepath: Path
):
    """Create a link between a dataset in an external file."""
    other_file = str(filepath.resolve().expanduser())
    if not isinstance(target, (str, bytes)):
        # Must be a list of keys (e.g. `['entry', 'data']` instead of
        # `"entry/data"`)
        target = "/".join(["", *target])
    link = h5py.ExternalLink(other_file, target)
    parent[name] = link


async def write_run(
    nxfile: h5py.File,
    run: Container,
    force: bool,
    semaphore: asyncio.Semaphore,
):
    """Write a run to the HDF file as a nexus-compatiable entry.

    *node* should be the container for this run. E.g.

    """
    name = run.metadata["start"]["uid"]
    nxfile.attrs["default"] = name
    entry = nxentry(nxfile, name)
    # Create bluesky groups
    nxdata(entry, "data")
    instrument = nxinstrument(entry, "instrument")
    bluesky = nxnote(instrument, "bluesky")
    write_metadata(run.metadata, entry=entry)
    # Write stream data
    nxnote(bluesky, "streams")
    if "streams" in run.keys():
        # Older Tiled versions use an additional "streams" branch in the tree
        streams = run["streams"]
    else:
        streams = run
    async with asyncio.TaskGroup() as tg:
        coros = [
            write_event_stream(
                name=stream_name, node=stream_node, entry=entry, semaphore=semaphore
            )
            for stream_name, stream_node in streams.items()
        ]
        tasks = [tg.create_task(coro) for coro in coros]
    # Write attributes
    return entry


async def write_results(
    entry: h5py.Group,
    run: Container,
    force: bool,
    semaphore: asyncio.Semaphore,
):
    """Write a run of results to the HDF file as a nexus-compatiable
    entry.

    Similar to `write_run()` except expects the basic structure to already exist.

    Parameters
    ==========
    entry
      The base NXEntry HDF5 group to which we should write.
    run
      The Tiled container with the results to write.

    """
    tasks = []
    writers = {
        DataFrameClient: write_table,
    }
    results_group = nxnote(entry, "results", exist_ok=True)
    async with asyncio.TaskGroup() as tg:
        for node_name, node in list(run.items()):
            # Different node structures need to be written differently
            write = writers[type(node)]
            tasks.append(
                tg.create_task(
                    write(
                        name=node_name,
                        node=node,
                        parent_group=results_group,
                        semaphore=semaphore,
                    )
                )
            )
    return entry


def to_hdf_type(value):
    """Some objects cannot be stored as HDF5 types.

    For example, a datetime should be converted to a string.

    Complex structures, like dictionaries, are converted to JSON.

    """
    type_conversions = [
        # (old => new)
        (dt.datetime, str),
        (dict, json.dumps),
        (list, json.dumps),
    ]
    new_types = [new for old, new in type_conversions if isinstance(value, old)]
    new_type = [*new_types, lambda x: x][0]
    return new_type(value)


def write_metadata(metadata: dict[str, Any], entry: h5py.Group):
    """Write run-level metadata to the Nexus file."""
    bluesky_group = entry["instrument/bluesky"]
    md_group = nxnote(bluesky_group, "metadata")
    flattened = {
        f"{doc_name}.{key}": value
        for doc_name, doc in metadata.items()
        for key, value in doc.items()
    }
    items = [(key, value) for key, value in flattened.items() if value is not None]
    for key, value in items:
        value = to_hdf_type(value)
        nxfield(md_group, key, value)
    # Create additional convenient links
    if "start.sample_name" in md_group.keys():
        nxlink(parent=entry, name="sample_name", target=md_group["start.sample_name"])
    if "start.scan_name" in md_group.keys():
        nxlink(parent=entry, name="scan_name", target=md_group["start.scan_name"])
    if "start.plan_name" in md_group.keys():
        nxlink(parent=entry, name="plan_name", target=md_group["start.plan_name"])
        nxlink(
            parent=bluesky_group, name="plan_name", target=md_group["start.plan_name"]
        )
    if "start.uid" in md_group.keys():
        nxlink(parent=entry, name="entry_identifier", target=md_group["start.uid"])
        nxlink(parent=bluesky_group, name="uid", target=md_group["start.uid"])
    for phase in ["start", "stop"]:
        if f"{phase}.time" in flattened.keys():
            timestamp = dt.datetime.fromtimestamp(flattened[f"{phase}.time"])
            nxfield(
                parent=entry,
                name=f"{phase}_time",
                value=timestamp.astimezone().isoformat(),
            )
    if "start.time" in flattened.keys() and "stop.time" in flattened.keys():
        nxfield(
            parent=entry,
            name="duration",
            value=flattened["stop.time"] - flattened["start.time"],
        )


def insert_data_source(parent: h5py.Group, source: DataSource):
    """Insert data into the HDF group by reading from another HDF5
    file.

    """
    # Create an empty array to hold the copied sources
    dtype_kind = source.structure["data_type"]["kind"]
    dtype_size = source.structure["data_type"]["itemsize"]
    dtype = f"{dtype_kind}{dtype_size}"
    if "value" in parent.keys():
        del parent["value"]
    ds = parent.create_dataset(
        "value",
        shape=source.structure["shape"],
        dtype=dtype,
        compression="gzip",
    )
    # Open and copy data from source files
    source_path = source.parameters["dataset"]
    assets = sorted(source.assets, key=lambda asset: asset.num)
    start = 0
    for asset in assets:
        uri = asset.data_uri
        if not uri.startswith("file://localhost"):
            raise ValueError(f"Cannot process data source uri: {uri}")
        source_file = uri.removeprefix("file://localhost")
        with h5py.File(source_file, mode="r") as src_fd:
            src_ds = src_fd[source_path]
            stop = start + src_ds.shape[0]
            try:
                ds[start:stop] = src_ds
            except Exception as exc:
                breakpoint()
        start = stop


async def write_array_slice(source, slc, dest, semaphore: asyncio.Semaphore):
    async with semaphore:
        arr = await asyncio.to_thread(source.read, slc)
    dest[slc] = arr


async def write_data_key(
    col_name: str,
    data_key: Mapping[str, Any],
    stream_node: Container,
    stream_group: h5py.Group,
    semaphore: asyncio.Semaphore,
) -> h5py.Group:
    """Load the data from the API and write it to an HDF5 group."""
    stream_name = stream_node.path_parts[-1]
    data_group = nxdata(stream_group, col_name)
    loop = asyncio.get_running_loop()
    ndims = len(data_key.get("shape", []))
    if ndims < 3:
        # Simple array, easier to load all at once
        async with semaphore:
            xarr = await loop.run_in_executor(
                None, stream_node.read, [col_name, f"ts_{col_name}"]
            )
        nxfield(data_group, "value", xarr[col_name])
    else:
        # Create an empty array to hold the data
        dtype = data_key.get("dtype_numpy", None)
        shape = data_key["shape"]
        if data_key.get("dtype", None) == "array":
            compression, chunks = ("gzip", (1, *shape[1:]))
        else:
            compression, chunks = (None, None)
        array_group = nxfield(
            data_group,
            "value",
            shape=shape,
            dtype=dtype,
            compression=compression,
            chunks=chunks,
        )
        # Load slices in parallel
        array_node = stream_node[col_name]
        coros = [
            write_array_slice(array_node, slc, array_group, semaphore=semaphore)
            for slc in range(shape[0])
        ]
        log.info(f"Loading {ndims}-D array '{stream_name}/{col_name}'")
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(coro) for coro in coros]
        # Read just the timestamp for setting later
        async with semaphore:
            xarr = await loop.run_in_executor(
                None, stream_node.read, [f"ts_{col_name}"]
            )
    # Set timestamps if we can, but not every array has timestamp information
    timestamps = xarr.get(f"ts_{col_name}")
    if timestamps is not None:
        nxfield(data_group, "EPOCH", timestamps)
        nxfield(data_group, "time", timestamps - np.min(timestamps))
        data_group["time"].attrs["units"] = "s"
        data_group.attrs["axes"] = "time"
    log.info(f"Could not find timestamps for dataset '{stream_name}/{col_name}'")

    # Extra parameters
    data_group.attrs["signal"] = "value"
    if "units" in data_key.keys():
        data_group["value"].attrs["units"] = data_key["units"]


async def write_table(
    name: str, node, parent_group: h5py.Group, semaphore: asyncio.Semaphore
) -> h5py.Group:
    """Write a Tiled table to the HDF file."""
    table_group = nxnote(parent_group, name)
    async with semaphore:
        df = await asyncio.to_thread(node.read)
    for series_name, series in df.items():
        data_group = nxdata(table_group, series_name)
        nxfield(data_group, "value", series.values)


async def write_event_stream(
    name: str, node, entry: h5py.Group, semaphore: asyncio.Semaphore
) -> h5py.Group:
    """Write a stream to the HDF file as a nexus-compatiable entry.

    *node* should be the container for this stream. E.g.

    .. code-block:: python

        write_stream(name="primary", node=run["primary"])

    Parameters
    ==========
    name
      The name for the new HDF5 NXdata group.
    node
      The tiled container for this stream.
    entry
      The HDF5 group/file to add this stream's group to.

    Returns
    =======
    grp
      The HDF5 group used to hold this stream's data.

    """
    metadata = node.metadata
    stream_group = nxnote(entry["instrument/bluesky/streams"], name, exist_ok=False)
    # Make sure we have access to these data
    # try:
    #     internal = node["internal"].read()
    # except KeyError:
    #     # We don't have an internal dataset for some reason
    #     internal = None
    # Write all the children to disk concurrently
    data_keys = metadata["data_keys"]
    async with asyncio.TaskGroup() as tg:
        coros = [
            write_data_key(
                col_name,
                data_key,
                stream_node=node,
                stream_group=stream_group,
                semaphore=semaphore,
            )
            for col_name, data_key in data_keys.items()
        ]
        tasks = [tg.create_task(coro) for coro in coros]
    # for col_name, desc in metadata["data_keys"].items():
    #     data_group = nxdata(stream_group, col_name)
    #     is_internal = internal is not None and col_name in internal
    #     # Check for pathologies
    #     if not is_internal and col_name not in node:
    #         warnings.warn(f"'{col_name}' in {node.uri} has a data key but no array.")
    #         continue
    #     # Write the dataset
    #     if is_internal:
    #         # Save internal dataset
    #         try:
    #             nxfield(data_group, "value", internal[col_name].values)
    #         except KeyError:
    #             raise SerializationError(
    #                 f"Could not find internal dataset '{col_name}'"
    #             )
    #         try:
    #             times = internal[f"ts_{col_name}"].values
    #         except KeyError, TypeError:
    #             log.error(
    #                 f"Could not find timestamps for internal dataset '{col_name}'"
    #             )
    #         else:
    #             nxfield(data_group, "EPOCH", times)
    #             data_group["time"] = times - np.min(times)
    #             data_group["time"].attrs["units"] = "s"
    #             data_group.attrs["axes"] = "time"
    #     elif True:
    #         # Load array data from files on disk
    #         if len(sources) < 1:
    #             continue
    #         if len(sources) > 1:
    #             raise ValueError(
    #                 "Exporter cannot yet export multi-source data. "
    #                 "Please submit an issue describing the use case."
    #             )
    #         (source,) = sources
    #         insert_data_source(data_group, source)
    #     else:
    #         # Most likely an external dataset
    #         arr = node[col_name].read()
    #         nxfield(data_group, "value", arr)
    #         try:
    #             times = node[f"ts_{col_name}"].read()
    #         except KeyError, TypeError:
    #             log.debug(
    #                 f"Could not find timestamps for external dataset '{col_name}'"
    #             )
    #         else:
    #             nxfield(data_group, "EPOCH", times)
    #             data_group["time"] = times - np.min(times)
    #             data_group["time"].attrs["units"] = "s"
    #             data_group.attrs["axes"] = "time"
    #     data_group.attrs["signal"] = "value"
    #     if "units" in desc.keys():
    #         data_group["value"].attrs["units"] = desc["units"]

    # Add links to the main NXdata group
    if name == "baseline":
        # We don't want to see baseline fields in the data NXdata group
        stream_hints = {}
    else:
        stream_hints = metadata.get("hints", {})
    root_nxdata = entry["data"]
    for device, hints in stream_hints.items():
        for field in hints.get("fields", []):
            # Make sure the field name is not already used in another stream
            link_name = field if field not in root_nxdata.keys() else f"field_{name}"
            # Write the link
            link_target = "/".join([stream_group.name, field, "value"])
            try:
                nxlink(root_nxdata, link_name, link_target, soft=True)
                # root_nxdata[link_name] = NXlinkfield(stream_group[field]["value"])
            except RuntimeError:
                raise SerializationError(
                    f"Could not link hinted '{name}' field: '{field}'"
                )
    return stream_group


@task()
async def serialize_hdf(
    buff: IO[bytes] | Path,
    run: Container,
    results_runs: Container | None = None,
    *,
    semaphore: asyncio.Semaphore,
    force: bool = False,
):
    """Encode a bluesky run into an HDF5 file with NeXus annotations.

    Follows the NeXuS XAS spectroscopy definition.

    """
    log = get_run_logger()
    if isinstance(buff, BytesIO):
        buff.seek(0)
    h5_mode = "w" if force else "x"
    log.info(f"Opening file '{buff}', in mode '{h5_mode}'")
    with h5py.File(buff, mode=h5_mode) as nxfile:
        uid = run.metadata.get("start", {}).get("uid", "")
        await create_link_artifact(
                key=f"export-{uid or '<Unknown UID>'}-hdf",
                link=f"file://{nxfile.filename}",
                description=f"# Exported HDF5 File\n\nRun UID: '{uid}'.\n",
            )
        # Write data entry to the nexus file
        entry = await write_run(
            nxfile=nxfile, run=run, force=force, semaphore=semaphore
        )
        # Write the results after the initial raw data have been written
        results = results_runs.values() if results_runs is not None else []
        coros = [
            write_results(entry=entry, run=run, force=force, semaphore=semaphore)
            for run in results
        ]
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(coro) for coro in coros]


def build_file_name(metadata: Mapping[str, Any]) -> str:
    """Build the name of the target HDF5 file based on metadata."""
    start_doc = metadata.get("start", {})
    start_time = dt.datetime.fromtimestamp(start_doc.get("time", 0))
    sample_name = start_doc.get("sample_name")
    scan_name = start_doc.get("scan_name")
    plan_name = start_doc.get("plan_name")
    uid_base = start_doc.get("uid", "").split("-")[0]
    bits = [
        start_time.strftime("%Y%m%d%H%M"),
        sample_name,
        scan_name,
        plan_name,
        uid_base,
    ]
    bits = [bit for bit in bits if bit not in ["", None]]
    base_name = "-".join(bits)
    base_name = re.sub(r"[ ]", "_", base_name)
    base_name = re.sub(r"[/]", "", base_name)
    return f"{base_name}.h5"


@flow()
def export_hdf(
    uid: str,
    *,
    raw_profile: str = "",
    target_dir: str = "",
    results_profile: str | None = None,
    force: bool = False,
    semaphore: asyncio.Semaphore | None = None,
):
    """Export a Tiled run with UID *uid* to an HDF5 file in *target_dir*.

    The name of the resulting HDF5 file will be generated from the run
    metadata. It will include the first portion of the UID, so
    presumably it will be unique. If a file of the same name already
    exists in *target_dir*, this operation will fail unless *force* is
    True, in which case the existing HDF file will be overwritten.

    If *target_dir* is not provided, a default destination for the new
    file will be determined from the corresponding DM experiment
    metadata, whose credentials are stored in the block
    "aps-dm-environments".

    Parameters
    ==========
    uid
      The UID of the Bluesky run to read from in the Tiled catalog.
    target_dir
      An existing folder in which to create a new HDF5 file. If
      omitted, a default location will be used.
    raw_profile
      The name of the Tiled profile to use for reading Bluesky
      runs. If an empty string (default), the default Tiled profile
      will be used.
    results_profile
      The name of the Tiled profile to use for reading processed
      results data.
    sempahore
      A locking semaphore to limit concurrent API connections. If
      omitted, a default will be created.

    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(10)
    raw_catalog = from_profile(raw_profile)
    run = raw_catalog[uid]
    if results_profile:
        results_catalog = from_profile(results_profile)
        results_runs = results_catalog.search(Eq("run_uid", uid))
    else:
        results_runs = None
    # DM experiments contain the export path, which is our default
    if not target_dir:
        dmax_client = load_client(run.metadata['start']['dm_station_name'])
        dm_exp = dmax_client.experiment(name=run.metadata['start']['dm_exp'])
        target_dir = dm_exp.data_path
    target_dir_ = Path(target_dir)
    target_file = target_dir_ / build_file_name(run.metadata)
    asyncio.run(
        serialize_hdf(
            buff=target_file,
            run=run,
            results_runs=results_runs,
            force=force,
            semaphore=semaphore,
        )
    )

