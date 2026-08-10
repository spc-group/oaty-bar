import asyncio
import datetime as dt
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from prefect import task
from prefect.artifacts import create_link_artifact
from prefect.concurrency.asyncio import rate_limit
from tiled.client.container import Container
from tiled.utils import SerializationError

from . import xdi

__all__ = ["serialize_tsv"]


log = logging.getLogger(__name__)


def headers(
    metadata: Mapping[str, Mapping],
    data_keys: Mapping[str, Mapping],
    *,
    strict: bool,
):
    """Generate individual header lines for the XDI file."""
    start_doc = metadata.get("start", {})
    # Version information
    if strict:
        versions = ["XDI/1.0"]
        version_md = start_doc.get("versions", {})
        versions += [f"{name}/{ver}" for name, ver in version_md.items()]
        yield f"# {' '.join(versions)}"
    # Column Names
    for num, (key, info) in enumerate(data_keys.items()):
        yield f"# Column.{num+1}: {key} {info.get('units', '')}"
    # X-ray edge information
    if strict and "edge" not in start_doc:
        raise SerializationError(
            "Metadata *edge* is required with strict XDI formatting."
        )
    edge_str = start_doc.get("edge", "") or ""  # Empty string in case it's `None`
    match = re.match(r"([A-Z][a-z]?)[-_]([K-Z]\d*)", edge_str)
    if match:
        elem, edge = match.groups()
        yield f"# Element.symbol: {elem}"
        yield f"# Element.edge: {edge}"
    elif strict:
        raise SerializationError(
            f"Metadata *edge* '{start_doc.get('edge')}' not in expected format."
        )
    # Instrument metadata
    d_spacing = metadata.get("start", {}).get("d_spacing")
    if d_spacing == "None":
        d_spacing = None
    if d_spacing is None and strict:
        raise SerializationError(
            "Argument *d_spacing* cannot be none with strict XDI formatting."
        )
    elif d_spacing is not None:
        yield f"# Mono.d_spacing: {d_spacing}"
    # Facility information
    if "time" in start_doc or strict:
        start_time = dt.datetime.fromtimestamp(start_doc["time"], dt.timezone.utc)
        start_time = start_time.astimezone()
        yield f"# Scan.start_time: {start_time.strftime('%Y-%m-%d %H:%M:%S%z')}"
    md_mappings = [
        # metadata key, XDI key
        ("facility_id", "Facility.name"),
        ("beamline_id", "Beamline.name"),
        ("uid", "uid"),
    ]
    md_mappings = [key for key in md_mappings if key[0] in start_doc]
    for md_key, xdi_key in md_mappings:
        yield f"# {xdi_key}: {start_doc[md_key]}"
    # Header end token
    if strict:
        yield "# -------------"


def data_keys(metadata: Mapping[str, Any]) -> dict[str, dict]:
    """Prepare valid hinted data keys for a stream.

    *metadata* should be the metadata dictionary for a specific stream.

    """
    dkeys = metadata["data_keys"]
    hints_ = metadata["hints"]
    hints = [
        hint for dev_hints in hints_.values() for hint in dev_hints.get("fields", [])
    ]
    dkeys = {key: desc for key, desc in dkeys.items() if key in hints}
    # Remove external datasets that won't be in the internal dataframe
    dkeys = {key: desc for key, desc in dkeys.items() if desc["dtype"] == "number"}
    return dkeys


def build_xdi(
    metadata: dict[str, Any],
    stream_metadata: dict[str, Any],
    data: xr.Dataset,
    old_data: xr.Dataset,
    *,
    strict: bool,
) -> str:
    """Build an XDI string based on data and metadata.

    Parameters
    ==========
    metadata
      Run-level metadata as received from Tiled.
    stream_metadata
      Stream-level metadata as received from Tiled.
    data
      The dataset loaded from Tiled with the data to save.
    old_data
      A dataset with any data loaded from an existing TSV/XDI
      file. These data will be merged with the new data.
    strict
      If true, raise an exception if required metadata keys are not
      found. Otherwise, missing keys are omitted from the header.

    """
    data_keys_ = data_keys(stream_metadata)
    start_doc = metadata.get("start", {})
    headers = {**old_data.attrs.get("header", {})}
    for num, (key, info) in enumerate(data_keys_.items()):
        headers[f"Column.{num+1}"] = f"{key} {info.get('units', '')}"
    # X-ray edge and d-spacing is required for proper XDI files
    if strict and "edge" not in start_doc:
        raise SerializationError(
            "Metadata *edge* is required with strict XDI formatting."
        )
    edge_str = start_doc.get("edge", "") or ""  # Empty string in case it's `None`
    match = re.match(r"([A-Z][a-z]?)[-_]([K-Z]\d*)", edge_str)
    if match:
        elem, edge = match.groups()
        headers["Element.symbol"] = elem
        headers["Element.edge"] = edge
    elif strict:
        raise SerializationError(
            f"Metadata *edge* '{start_doc.get('edge')}' not in expected format."
        )
    d_spacing = start_doc.get("d_spacing")
    d_spacing = None if d_spacing == "None" else d_spacing
    if d_spacing is None and strict:
        raise SerializationError(
            "Argument *d_spacing* cannot be none with strict XDI formatting."
        )
    elif d_spacing is not None:
        headers["Mono.d_spacing"] = d_spacing
    # Other header metadata
    if "plan_args" in start_doc:
        headers["plan_args"] = json.dumps(start_doc["plan_args"])
    if "time" in start_doc or strict:
        start_time = dt.datetime.fromtimestamp(start_doc["time"], dt.timezone.utc)
        start_time = start_time.astimezone()
        headers["Scan.start_time"] = start_time.strftime("%Y-%m-%d %H:%M:%S%z")
    md_mappings = [
        # metadata key, XDI key
        ("facility_id", "Facility.name"),
        ("beamline_id", "Beamline.name"),
        ("uid", "uid"),
    ]
    md_mappings = [key for key in md_mappings if key[0] in start_doc]
    for md_key, xdi_key in md_mappings:
        headers[xdi_key] = start_doc[md_key]
    attrs: xdi.Attrs = {
        "xdi_version": "1.0" if strict else "",
        "header": headers,
        "user_comment": start_doc.get("notes", old_data.attrs.get("user_comment", "")),
        "versions": start_doc.get("versions", {**old_data.attrs.get("versions", {})}),
    }
    # We need to pick a single coordinate to produce well-structured dataframes
    dims_ = start_doc.get("hints", {}).get("dimensions", [])
    dimensions = [dim for dims, stream in dims_ if stream == "primary" for dim in dims]
    possible_coords = [
        "monochromator-energy",
        "mono-energy",
        "energy",
        *dimensions,
        "seq_num",
    ]
    data_vars = {**old_data.data_vars, **old_data.coords, **data.data_vars}
    possible_coords = [name for name in possible_coords if name in data_vars]
    coords = {}
    if len(possible_coords) > 0:
        name, *_ = possible_coords
        coords[name] = (name, data_vars[name].data)
        data_vars = {
            key: (name, val.data) for key, val in data_vars.items() if key != name
        }
    # Build a combined dataset from the old and new data
    # data = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
    # **old_data.data_vars, **old_data.coords, **data.data_vars}
    xarr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    xdi_text = xdi.dump(xarr, strict=strict)
    return xdi_text


async def load_dataset(node: Container) -> xr.Dataset:
    """Load hinted signals for a given the given Tiled node."""
    hinted_keys = data_keys(node.metadata)
    await rate_limit("tiled-api")
    data = await asyncio.to_thread(node.read, list(hinted_keys.keys()))
    # data = await asyncio.to_thread(node.read)
    if isinstance(data, pd.DataFrame):
        data = data.to_xarray()
    return data


@task(
    tags=["export"],
    retries=3,
    retry_delay_seconds=10,
    retry_jitter_factor=3,
)
async def serialize_tsv(
    filepath: str | Path,
    run: Container,
    results_runs: Container | None = None,
    use_xdi: bool = False,
):
    """Write a bluesky run as tab-separated values.

    Assumes that *node* is a BlueskyRun.

    Includes some headers, though nothing is required.

    Matches the XDI specification if *use_xdi* is true, or if the
    start document *plan_name* is `"xafs_scan"`.

    """
    filepath = Path(filepath)
    streams = run
    if "streams" in run.keys():
        # Older versions of Tiled have an additional "streams" node here
        streams = run["streams"]
    stream_node = streams["primary"]
    data = await load_dataset(stream_node)
    # Get extra data (results, etc)
    if results_runs is not None:
        await rate_limit("tiled-api")
        _result_runs = await asyncio.to_thread(list, results_runs.values())
        result_streams = [
            node
            for run in _result_runs
            for node in await asyncio.to_thread(list, run.values())
        ]
    else:
        result_streams = []
    other_datasets = await asyncio.gather(
        *[load_dataset(node) for node in result_streams]
    )
    for other in other_datasets:
        data = data.merge(other)
    # Load existing data from file so we can merge them
    if filepath.exists():
        with open(filepath, mode="r") as fd:
            old_data = xdi.load(fd.read(), strict=use_xdi)
    else:
        old_data = xr.Dataset()
    # Write the new dataset to the XDI file
    xdi_text = build_xdi(
        metadata=run.metadata,
        stream_metadata=stream_node.metadata,
        data=data,
        old_data=old_data,
        strict=use_xdi,
    )
    with open(filepath, mode="w") as fd:
        uid = run.metadata.get("start", {}).get("uid", "")
        await create_link_artifact(
            key=f"export-{uid or '<Unknown UID>'}-tsv",
            link=f"file://{filepath}",
            description=f"# Exported HDF5 File\n\nRun UID: '{uid}'.\n",
        )
        fd.write(xdi_text)
    return xdi_text
