import io

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import xarray as xr
from prefect import flow

from oaty_bar import xdi
from oaty_bar._export_tsv import (
    headers,
    serialize_tsv,
)


@pytest_asyncio.fixture()
async def xdi_text(xafs_run, tmp_path):
    # Generate the headers
    tmp_file = tmp_path / "xdi_export.xdi"

    @flow()
    async def do_flow():
        return await serialize_tsv(
            tmp_file,
            run=xafs_run,
            use_xdi=True,
        )

    await do_flow()
    # Read the text from the file to make sure it got written
    with open(tmp_file, mode="r") as fd:
        yield fd.read()


@pytest_asyncio.fixture()
async def tsv_text(xafs_run, tmp_path):
    tmp_file = tmp_path / "xdi_export.tsv"

    @flow()
    async def do_flow():
        return await serialize_tsv(
            tmp_file,
            run=xafs_run,
            use_xdi=False,
        )

    await do_flow()
    # Read the text from the file to make sure it got written
    with open(tmp_file, mode="r") as fd:
        yield fd.read()


@pytest.fixture()
def results_catalog(results_catalog):
    run = results_catalog.create_container(
        "blahblah", metadata={"run_uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f"}
    )
    print(help(run.write_table))
    stream = run.write_table(
        pd.DataFrame({"Ni": np.arange(100)}),
        key="ge_8element_fit",
        metadata={
            "data_keys": {
                "Ni": {
                    "dtype": "number",
                    "dtype_numpy": "<i4",
                    "shape": [],
                },
            },
            "hints": {
                "ge_8element": {"fields": ["Ni"]},
            },
        },
    )
    return results_catalog


def test_required_headers(xdi_text):
    assert "# XDI/1.0 bluesky/1.9.0 ophyd/1.7.0" in xdi_text
    assert "# Column.1: energy eV" in xdi_text
    assert "# Column.3: It-net_current A" in xdi_text
    assert "# Element.symbol: Ni" in xdi_text
    assert "# Element.edge: K" in xdi_text
    assert "# Mono.d_spacing: 3.13" in xdi_text
    assert "# -----" in xdi_text


def test_optional_headers(xdi_text):
    expected_metadata = {
        "Facility.name": "Advanced Photon Source",
        # "Facility.xray_source": "insertion device",
        "Beamline.name": "255-ID-Z",
        "Scan.start_time": "2022-10-06 09:14:57-0500",
        "uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    }
    for key, val in expected_metadata.items():
        assert f"# {key.lower()}: {val.lower()}\n" in xdi_text.lower()


def test_tsv_headers(tsv_text):
    """Do we still get a valid TSV file without any metadata."""
    assert "# energy\tenergy-id-energy-readback\tIt-net_current" in tsv_text
    # assert "# energy energy-id-energy-readback It-net_current" in tsv_text
    # assert "d_spacing" not in tsv_text
    # Check the data
    buff = io.StringIO(tsv_text)
    df = pd.read_csv(buff, comment="#", sep="\t")
    assert len(df.columns) == 3


@pytest.mark.asyncio
async def test_export_results(xafs_run, results_catalog, tmp_path):
    xdi_path = tmp_path / "example.xdi"

    @flow()
    async def do():
        await serialize_tsv(xdi_path, xafs_run, results_runs=results_catalog)

    await do()

    with open(xdi_path, mode="r") as fd:
        xdi_text = fd.read()
        xarr = xdi.load(xdi_text, strict=False)
    assert set(xarr.coords.keys()) == {"energy"}
    assert set(xarr.keys()) == {
        "energy-id-energy-readback",
        "It-net_current",
        "Ni",
    }


def test_missing_edge(tsv_text):
    """Can we export with missing edge information."""
    list(
        headers(
            metadata={
                "start": {
                    "edge": None,
                }
            },
            data_keys={},
            strict=False,
        )
    )


def test_data(xdi_text):
    """Check that the TSV data section is present and correct."""
    # Read as if it were a pandas dataframe
    buff = io.StringIO(xdi_text)
    # Check for the header
    assert "# energy\tenergy-id-energy-readback\tIt-net_current" in xdi_text
    # Check the data
    df = pd.read_csv(buff, comment="#", sep="\t")
    assert len(df.columns) == 3


@pytest.mark.asyncio
async def test_update_existing(xafs_run, results_catalog, mocker, tmp_path):
    """Can we export the same file twice with consistent results."""
    xdi_path = tmp_path / "example.xdi"

    # Write a partial XDI file that we will update later
    n_events = xafs_run["primary/energy"].shape[0]
    initial_arr = xr.Dataset(
        {"monochromator-energy": np.arange(100)}, attrs={"xdi_version": "1.0"}
    )
    with open(xdi_path, mode="w") as fd:
        xdi_text = xdi.dump(initial_arr)
        fd.write(xdi_text)

    @flow()
    async def do():
        await serialize_tsv(xdi_path, xafs_run)

    await do()

    with open(xdi_path, mode="r") as fd:
        xdi_text = fd.read()
        xarr = xdi.load(xdi_text, strict=False)
    assert set(xarr.keys()) == {"energy", "energy-id-energy-readback", "It-net_current"}
    assert set(xarr.coords.keys()) == {"monochromator-energy"}
