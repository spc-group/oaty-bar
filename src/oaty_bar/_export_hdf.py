import argparse
import datetime as dt
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import h5py
import numpy as np
from tiled.client import from_profile
from tiled.client.container import Container
from tiled.utils import SerializationError

log = logging.getLogger(__name__)


def nxgroup(parent: h5py.Group, name: str, nx_class: str | None = None) -> h5py.Group:
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


def nxnote(parent: h5py.Group, name: str) -> h5py.Group:
    return nxgroup(parent=parent, name=name, nx_class="NXnote")


def nxfield(parent: h5py.Group, name: str, value) -> h5py.Dataset:
    field = parent.create_dataset(name, data=value)
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


def write_run(
    nxfile: h5py.File,
    run: Container,
    streams_prefix: str = "streams",
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
    nxnote(bluesky, "streams")
    # Write stream data
    write_metadata(run.metadata, entry=entry)
    streams = (run[streams_prefix]) if streams_prefix else run
    for stream_name, stream_node in streams.items():
        write_stream(
            name=stream_name,
            node=stream_node,
            entry=entry,
        )
    # Write attributes
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


def write_stream(name: str, node, entry: h5py.Group) -> h5py.Group:
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
    stream_group = nxnote(entry["instrument/bluesky/streams"], name)
    # Make sure we have access to these data
    try:
        internal = node["internal"].read()
    except KeyError:
        # We don't have an internal dataset for some reason
        internal = None
    for col_name, desc in metadata["data_keys"].items():
        data_group = nxdata(stream_group, col_name)
        is_internal = internal is not None and col_name in internal
        if is_internal:
            # Save internal dataset
            try:
                nxfield(data_group, "value", internal[col_name].values)
            except KeyError:
                raise SerializationError(
                    f"Could not find internal dataset '{col_name}'"
                )
            try:
                times = internal[f"ts_{col_name}"].values
            except (KeyError, TypeError):
                log.error(
                    f"Could not find timestamps for internal dataset '{col_name}'"
                )
            else:
                nxfield(data_group, "EPOCH", times)
                data_group["time"] = times - np.min(times)
                data_group["time"].attrs["units"] = "s"
                data_group.attrs["axes"] = "time"
        else:
            # Most likely an external dataset
            arr = node[col_name].read()
            nxfield(data_group, "value", arr)
            try:
                times = node[f"ts_{col_name}"].read()
            except (KeyError, TypeError):
                log.debug(
                    f"Could not find timestamps for external dataset '{col_name}'"
                )
            else:
                nxfield(data_group, "EPOCH", times)
                data_group["time"] = times - np.min(times)
                data_group["time"].attrs["units"] = "s"
                data_group.attrs["axes"] = "time"
        data_group.attrs["signal"] = "value"
        if "units" in desc.keys():
            data_group["value"].attrs["units"] = desc["units"]

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


def serialize_hdf(buff: IO[bytes], run: Container):
    """Encode a bluesky run into an HDF5 file with NeXus annotations.

    Follows the NeXuS XAS spectroscopy definition.

    """
    buff.seek(0)
    with h5py.File(buff, mode="w") as nxfile:
        # Write data entry to the nexus file
        write_run(nxfile=nxfile, run=run)


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


def export_hdf(uid: str, target_dir: Path, *, raw_profile: str, processed_profile: str):
    raw_catalog = from_profile(raw_profile)
    run = raw_catalog[uid]
    target_file = target_dir / build_file_name(run.metadata)
    with open(target_file, mode="ab+") as fd:
        print(fd)
        serialize_hdf(buff=fd, run=run)


def main(args: Sequence[str] | None = None):
    """Main entry-point for exporting an HDF5 file for a given run."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="export-hdf",
        description="Export an HDF5 file for a given run",
    )
    parser.add_argument("uid", help="The UID of the bluesky run to export.")
    parser.add_argument(
        "target_dir", help="The DM directory to receive the exported file."
    )
    parser.add_argument(
        "--raw-profile", help="The name of the Tiled profile used for raw runs."
    )
    parser.add_argument(
        "--processed-profile",
        help="The name of the Tiled profile used for processed run data.",
    )
    parsed = parser.parse_args(args)
    # Do the actual exporting
    export_hdf(
        uid=parsed.uid,
        target_dir=Path(parsed.target_dir),
        raw_profile=parsed.raw_profile,
        processed_profile=parsed.processed_profile,
    )
