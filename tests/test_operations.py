import numpy as np
import pytest
import xarray as xr

from oaty_bar._operations import apply_correction, extract_rois


@pytest.mark.asyncio
async def test_extract_rois():
    event_dataset = xr.Dataset(
        data_vars={
            "vortex_me4": (["seq_num", "dim1", "dim2"], np.ones((3, 4, 256))),
        },
        coords={
            "seq_num": [1, 2, 3],
        },
    )

    async def events():
        yield event_dataset

    roi_slices = {"Ni_Ka": [[0, 2]], "full": [slice(None)]}
    rois = extract_rois(events=events(), run=None, source="vortex_me4", rois=roi_slices)
    rois = [roi async for roi in rois]
    assert len(rois) > 0
    roi_row = rois[0]
    assert roi_row["vortex_me4-full-sum"].data[0] == 4 * 256
    assert roi_row["vortex_me4-Ni_Ka-sum"].data[0] == 2 * 256


@pytest.mark.asyncio
async def test_apply_correction():
    clock_ticks = np.random.randint(79_980_000, 80_020_000, (3, 2))
    counts = np.random.randint(10_000, 1_300_000, (3, 2, 256))
    event_dataset = xr.Dataset(
        data_vars={
            "vortex_me4": (["seq_num", "sensor", "bin"], counts),
            "clock_ticks": (["seq_num", "sensor"], clock_ticks),
        },
        coords={
            "seq_num": [1, 2, 3],
        },
    )

    async def events():
        yield event_dataset

    results = apply_correction(
        events=events(),
        run=None,
        source_signal="vortex_me4",
        correction_signal="clock_ticks",
        operation=np.divide,
        scale_factor=80_000_000,
    )
    results = [res async for res in results]
    result_row = results[0]
    assert np.all(result_row["vortex_me4"].data < 1)
