import io
from pathlib import Path
from typing import IO
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from nexusformat.nexus import NXFile
from prefect import flow

from oaty_bar._export_hdf import (
    nxexternallink,
    serialize_hdf,
    write_event_stream,
)

specification = """
root:NXroot
  @default = '7d1daf1d-60c7-4aa7-a668-d1cd97e5335f'
  7d1daf1d-60c7-4aa7-a668-d1cd97e5335f:NXentry
    data:NXdata
      It-net_current -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/streams/primary/It-net_current/value
      energy -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/streams/primary/energy/value
      energy-id-energy-readback -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/streams/primary/energy-id-energy-readback/value
      ge_8element -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/streams/primary/ge_8element/value
    duration = 38.35049033164978
    entry_identifier -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/metadata/start.uid
    instrument:NXinstrument
      bluesky:NXnote
        metadata:NXnote
          start.beamline_id = '255-ID-Z'
          start.d_spacing = 3.131562
          start.detectors = '["I0"]'
          start.edge = 'Ni-K'
          start.facility_id = 'Advanced Photon Source'
          start.hints = '{"dimensions": [[["pitch2"], "primary"]]}'
          start.motors = '["pitch2"]'
          start.num_intervals = 19
          start.num_points = 20
          start.plan_args = '{"args": ["EpicsMotor(prefix='25idDCM:AS:m6', name='pitc...'
          start.plan_name = 'xafs_scan'
            @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
          start.plan_pattern = 'inner_product'
          start.plan_pattern_args = '{"args": ["EpicsMotor(prefix='25idDCM:AS:m6', name='pitc...'
          start.plan_pattern_module = 'bluesky.plan_patterns'
          start.plan_type = 'generator'
          start.purpose = 'alignment'
          start.sample_name = 'NMC-811'
            @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
          start.scan_id = 1
          start.scan_name = 'Pristine'
            @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
          start.time = 1665065697.3635247
          start.uid = '7d1daf1d-60c7-4aa7-a668-d1cd97e5335f'
            @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
          start.versions = '{"bluesky": "1.9.0", "ophyd": "1.7.0"}'
          stop.exit_status = 'success'
          stop.num_events = '{"primary": 20}'
          stop.reason = ''
          stop.run_start = '7d1daf1d-60c7-4aa7-a668-d1cd97e5335f'
          stop.time = 1665065735.714015
          stop.uid = 'c1eac86f-d568-41a1-b601-a0e2fd6ed55e'
          summary.datetime = '2022-10-06 09:14:57.363525'
          summary.duration = 38.35049033164978
          summary.plan_name = 'xafs_scan'
          summary.scan_id = 1
          summary.stream_names = '["primary"]'
          summary.timestamp = 1665065697.3635247
          summary.uid = '7d1daf1d-60c7-4aa7-a668-d1cd97e5335f'
        plan_name -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/metadata/start.plan_name
        streams:NXnote
          baseline:NXnote
            aps_current:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = [10 25]
              time = [ 0 15]
                @units = 's'
              value = [130.  204.1]
                @units = 'mA'
            aps_fill_number:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = [10 25]
              time = [ 0 15]
                @units = 's'
              value = [1 2]
            aps_global_feedback:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = [10 25]
              time = [ 0 15]
                @units = 's'
              value = [ True False]
          primary:NXnote
            I0-net_current:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = float64(100)
              time = float64(100)
                @units = 's'
              value = float64(100)
                @units = 'A'
            It-net_current:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = float64(100)
              time = float64(100)
                @units = 's'
              value = float64(100)
                @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
                @units = 'A'
            energy:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = float64(100)
              time = float64(100)
                @units = 's'
              value = float64(100)
                @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
                @units = 'eV'
            energy-id-energy-readback:NXdata
              @axes = 'time'
              @signal = 'value'
              EPOCH = float64(100)
              time = float64(100)
                @units = 's'
              value = float64(100)
                @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
                @units = 'keV'
            ge_8element:NXdata
              @signal = 'value'
              value = uint32(100x8x1024)
                @target = '/7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrume...'
            ge_8element-element0-all_event:NXdata
              @signal = 'value'
              value = float64(100)
        uid -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/metadata/start.uid
    plan_name -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/metadata/start.plan_name
    sample_name -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/metadata/start.sample_name
    scan_name -> /7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/instrument/bluesky/metadata/start.scan_name
    start_time = '2022-10-06T09:14:57.363525-05:00'
    stop_time = '2022-10-06T09:15:35.714015-05:00'
"""


@pytest.fixture()
def results_catalog(results_catalog):
    run = results_catalog.create_container(
        "blahblah", metadata={"run_uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f"}
    )
    stream = run.write_table(
        pd.DataFrame({"Ni": [1, 2, 3, 4]}),
        key="ge_8element_fit",
    )
    return results_catalog


class NexusIO(NXFile):
    def __init__(self, bytesio: IO[bytes], mode: str = "r", **kwargs):
        self.h5 = h5py
        self.name = ""
        self._file = None
        self._filename = "/dev/null"
        self._filedir = "/tmp"
        self._lock = None
        self._lockdir = None
        self._path = "/"
        self._root = None
        self._with_count = 0
        self.recursive = True

        self._file = self.h5.File(bytesio, mode, **kwargs)

        if mode == "r":
            self._mode = "r"
        else:
            self._mode = "rw"

    def acquire_lock(self, timeout=None):
        pass

    def release_lock(self, timeout=None):
        pass

    def open(self, **kw):
        pass


@pytest_asyncio.fixture()
async def nxfile(xafs_run):
    # Generate the headers
    buff = io.BytesIO()

    @flow()
    async def do():
        await serialize_hdf(buff, xafs_run)

    await do()
    with NexusIO(buff, mode="r") as fd:
        # Write data entry to the nexus file
        yield fd


def test_xafs_specification(nxfile):
    tree = nxfile.readfile().tree
    # with open("/home/beams/WOLFMAN/tmp/test_file.tree", mode='w') as fd:
    #     fd.write(tree)
    tree = "\n" + tree + "\n"  # Extra newlines to match the triple quote string
    assert tree == specification


def test_file_structure(nxfile):
    uid = "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f"
    # Check the top-level entry
    assert nxfile.attrs["default"] == uid
    assert uid in nxfile.keys()
    entry = nxfile[uid]
    assert entry.attrs["NX_class"] == "NXentry"


@pytest.mark.asyncio
async def test_missing_hints(xafs_run):
    """Make sure the stream still writes if there are not hints."""
    node = mock.AsyncMock()
    node.metadata = {
        "data_keys": {},
        "hints": {"I0": {}},
    }
    await write_event_stream(
        name="primary",
        node=node,
        entry=mock.MagicMock(),
        overwrite=False,
    )


def test_external_datasets(nxfile):
    """Make sure the data from an external dataset gets properly written
    to the HDF5 file.

    """
    uid = "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f"
    ds = nxfile[f"{uid}/data/ge_8element"]
    assert ds.shape == (100, 8, 1024)
    assert str(ds.dtype) == "uint32"
    assert np.min(ds) == 1


def test_nxexternallink_targets():
    buff = io.BytesIO()
    with h5py.File(buff, mode="w") as fd:
        nxexternallink(
            parent=fd, name="externA", target="/entry/data/", filepath=Path("/dev/null")
        )
        nxexternallink(
            parent=fd,
            name="externB",
            target=["entry", "data"],
            filepath=Path("/dev/null"),
        )
        linkA = fd.get("externA", getlink=True)
        linkB = fd.get("externB", getlink=True)
    assert linkA.path == "/entry/data"
    assert linkB.path == "/entry/data"


@pytest.mark.asyncio
async def test_export_results(xafs_run, results_catalog, mocker):
    buff = io.BytesIO()

    @flow()
    async def do():
        await serialize_hdf(buff, xafs_run, results_runs=results_catalog)

    await do()
    with h5py.File(buff, mode="r") as fd:
        # Check that all datasets are in the 'results' group
        results = fd["7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/results"]
        assert "ge_8element_fit" in results.keys()
        assert "Ni" in results["ge_8element_fit"].keys()
        # Check that a link is created in the 'data' group
        # data = fd["7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/data"]
        # assert "ge_8element-Ni" in data.keys()


@pytest.mark.asyncio
async def test_idempotence(xafs_run, results_catalog, mocker):
    """Can we export the same file twice with consistent results."""
    buff = io.BytesIO()

    @flow()
    async def do():
        await serialize_hdf(buff, xafs_run, results_runs=results_catalog, force=True)

    await do()
    await do()
    with h5py.File(buff, mode="r") as fd:
        # Check that all datasets are in the 'results' group
        results = fd["7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/results"]
        assert "ge_8element_fit" in results.keys()
        assert "Ni" in results["ge_8element_fit"].keys()
        # Check that a link is created in the 'data' group
        # data = fd["7d1daf1d-60c7-4aa7-a668-d1cd97e5335f/data"]
        # assert "ge_8element-Ni" in data.keys()
