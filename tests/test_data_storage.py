from pathlib import Path

import pytest
import pytest_asyncio
from prefect.variables import Variable

from oaty_bar._data_storage import prepare_data_storage


@pytest_asyncio.fixture()
async def local_storage(tmp_path):
    storage_var = {"255IDZ": str(tmp_path)}
    await Variable.set("local-storage-by-station", storage_var, overwrite=True)
    return storage_var


@pytest.fixture()
def dmax_client(mocker):
    mock_api = mocker.AsyncMock()
    mocker.patch(
        "oaty_bar._data_storage.load_client",
        new=mocker.AsyncMock(return_value=mock_api),
    )
    return mock_api


@pytest.mark.asyncio
async def test_start_data_transfer(prefect_server, local_storage, dmax_client):
    await prepare_data_storage(dm_exp="commission-1999-C0", dm_station_name="255IDZ")
    assert dmax_client.start_data_archive_queue.called
    assert dmax_client.start_data_archive_queue.call_args.kwargs == {
        "source_directory": f"{local_storage['255IDZ']}/commission-1999-C0",
        "experiment_name": "commission-1999-C0",
        "skip": ".pixi/*",
    }


@pytest.mark.asyncio
async def test_prepare_local_folder(prefect_server, local_storage, dmax_client):
    await prepare_data_storage(dm_exp="commission-1999-C0", dm_station_name="255IDZ")
    expected_folder = Path(local_storage["255IDZ"]) / "commission-1999-C0"
    assert expected_folder.exists()


@pytest.mark.asyncio
async def test_populate_skeleton_files(prefect_server, local_storage, dmax_client):
    await prepare_data_storage(dm_exp="commission-1999-C0", dm_station_name="255IDZ")
    expected_folder = Path(local_storage["255IDZ"]) / "commission-1999-C0"
    assert (expected_folder / "pixi.toml").exists()
