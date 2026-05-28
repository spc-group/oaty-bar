import pytest

from oaty_bar._data_management import DataManagementStation, load_client


@pytest.mark.asyncio
async def test_dm_client(prefect_server):
    """Do we build a DM client from the prefect variable."""
    station = DataManagementStation(
        username="monty",
        password="secret",
        station_name="255IDZ",
        scheduling_uri="blah",
        data_storage_uri="spam",
        processing_uri="eggs",
    )
    await station.save("255idz")
    client = await load_client(station_name="255IDZ")
    assert client.username == "monty"
    assert client.password == "secret"
    assert client.station_name == "255IDZ"
    assert client._bss_context.base_uri == "blah/dm"
    assert client._ds_context.base_uri == "spam/dm"
    assert client._proc_context.base_uri == "eggs/dm"
