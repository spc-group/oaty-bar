import pytest
import xarray as xr
from tiled.client.composite import CompositeClient

from oaty_bar._writers import write_results


@pytest.mark.slow
@pytest.mark.asyncio
async def test_create_new_containers(tiled_catalog):
    tiled_catalog.create_container("867-5309", metadata={"run_start": "975-3146"})
    event_dataset = xr.Dataset(
        data_vars={
            "vortex_me4-Ni_Ka_guess": (["index"], [56084, 49553, 12535]),
            "vortex_me4-deadtime": (
                ["index", "sensor"],
                [[10001, 10002], [10003, 10004], [10005, 10006]],
            ),
            "seq_num": (["index"], [1, 2, 3]),
        },
        coords={
            "index": [0, 1, 2],
        },
        attrs={
            "source_client": tiled_catalog["867-5309"],
        },
    )

    async def events():
        yield event_dataset

    writer = write_results(events(), catalog=tiled_catalog)
    (event,) = [ev async for ev in writer]
    # Load the results again to make sure it's the right type of
    # client
    results_client = event.attrs["results_client"]
    results = tiled_catalog[results_client.path_parts[-1]]
    assert isinstance(results, CompositeClient)
    assert "vortex_me4-deadtime" in list(results.keys())
    assert "vortex_me4-Ni_Ka_guess" in list(results.keys())
    assert results["vortex_me4-deadtime"].shape == (3, 2)
    assert results["vortex_me4-Ni_Ka_guess"].shape == (3,)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_add_to_existing_containers(tiled_catalog):
    tiled_catalog.create_container("867-5309", metadata={"run_start": "975-3146"})
    event_dataset = xr.Dataset(
        data_vars={
            "vortex_me4-Ni_Ka_guess": (["index"], [56084, 49553, 12535]),
            "vortex_me4-deadtime": (
                ["index", "sensor"],
                [[10001, 10002], [10003, 10004], [10005, 10006]],
            ),
            "seq_num": (["index"], [1, 2, 3]),
        },
        coords={
            "index": [0, 1, 2],
        },
        attrs={
            "source_client": tiled_catalog["867-5309"],
        },
    )

    async def events():
        # Emit the rows one at a time
        for idx in range(3):
            yield event_dataset.isel(index=[idx])

    writer = write_results(events(), catalog=tiled_catalog)
    events = [ev async for ev in writer]
    *_, event = events
    # Load the results again to make sure it's the right type of
    # client
    results_client = event.attrs["results_client"]
    results = tiled_catalog[results_client.path_parts[-1]]
    assert isinstance(results, CompositeClient)
    assert "vortex_me4-deadtime" in list(results.keys())
    assert "vortex_me4-Ni_Ka_guess" in list(results.keys())
    assert results["vortex_me4-deadtime"].shape == (3, 2)
    assert results["vortex_me4-Ni_Ka_guess"].shape == (3,)
