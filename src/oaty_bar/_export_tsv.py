import asyncio
import datetime as dt
import io
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

from pandas import DataFrame
from prefect import task
from tiled.client.container import Container
from tiled.utils import SerializationError

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


def data_keys(metadata: Mapping[str, Mapping | str | float | int]) -> dict[str, dict]:
    """Prepare valid hinted data keys for a stream.

    *metadata* should be the metadata dictionary for a specific stream.

    """
    dkeys = metadata["data_keys"]
    hints = metadata["hints"]
    hints = [
        hint for dev_hints in hints.values() for hint in dev_hints.get("fields", [])
    ]
    dkeys = {key: desc for key, desc in dkeys.items() if key in hints}
    # Remove external datasets that won't be in the internal dataframe
    dkeys = {key: desc for key, desc in dkeys.items() if desc["dtype"] == "number"}
    return dkeys


def build_xdi(
    metadata: dict[str, Any],
    stream_metadata: dict[str, Any],
    data: DataFrame,
    *,
    strict: bool,
) -> IO[bytes]:
    """Build an XDI string based on data and metadata.

    Parameters
    ==========
    strict
      If true, raise an exception if required metadata keys are not
      found. Otherwise, missing keys are omitted from the header.

    """
    data_keys_ = data_keys(stream_metadata)
    # Write headers
    xdi_text = ""
    hdrs = headers(metadata, data_keys=data_keys_, strict=strict)
    xdi_text += "\n".join(hdrs) + "\n"
    # Write data
    cols = "\t".join(data_keys_.keys())
    xdi_text += f"# {cols}\n"
    buffer = io.StringIO()
    # Convert it from an xarray into a pandas data frame for easy serialization
    # data_dict = data.to_dict()
    # data_dict = {**data_dict["data_vars"], **data_dict["coords"]}
    # data_dict = {name: val["data"] for name, val in data_dict.items()}
    # df = DataFrame(data_dict)
    df = data.to_dataframe()
    df.to_csv(buffer, sep="\t", header=False, index=False)
    buffer.seek(0)
    xdi_text += buffer.read()
    return xdi_text


@task()
async def serialize_tsv(
    buff: IO[bytes] | Path, run: Container, use_xdi: bool | None = None
):
    """Write a bluesky run as tab-separated values.

    Assumes that *node* is a BlueskyRun.

    Includes some headers, though nothing is required.

    Matches the XDI specification if *use_xdi* is true, or if the
    start document *plan_name* is `"xafs_scan"`.

    """
    streams = run
    if "streams" in run.keys():
        # Older versions of Tiled have an additional "streams" node here
        streams = run["streams"]
    stream_node = streams["primary"]
    # Get extra data
    hinted_keys = data_keys(stream_node.metadata)
    data = await asyncio.to_thread(stream_node.read, hinted_keys.keys())
    xdi_text = build_xdi(
        metadata=run.metadata,
        stream_metadata=stream_node.metadata,
        data=data,
        strict=use_xdi,
    )
    with open(buff, mode="w") as fd:
        fd.write(xdi_text)
    return xdi_text
