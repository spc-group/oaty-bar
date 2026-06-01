from pathlib import Path

from prefect import flow, task
from prefect.variables import Variable

from ._data_management import load_client


@flow()
async def prepare_data_storage(dm_exp: str, dm_station_name: str) -> None:
    """Workflow to prepare storage space for an experiment.

    - Create a local folder
    - Start a data synchronization (DAQ)
    - Populate the local folder with skeleton files

    """
    new_path = create_local_folder(dm_exp=dm_exp, dm_station_name=dm_station_name)
    await start_data_transfer(
        dm_exp=dm_exp, dm_station_name=dm_station_name, source_path=new_path
    )
    populate_skeleton_files(new_path)


@task()
def create_local_folder(dm_exp: str, dm_station_name: str) -> str:
    storage = Variable.get("local-storage-by-station")
    root = Path(storage[dm_station_name])
    new_dir = root / dm_exp
    new_dir.mkdir(exist_ok=True)
    return str(new_dir.resolve())


@task()
async def start_data_transfer(dm_exp: str, dm_station_name: str, source_path: str):
    """Start a data managamenet data transfer (DAQ) that monitors a
    folder for new files and uploads them to shared storage.

    Parameters
    ==========

    """
    dm_client = await load_client(dm_station_name)
    await dm_client.start_data_archive_queue(
        source_directory=source_path, experiment_name=dm_exp
    )


@task()
def populate_skeleton_files(destination_path: str):
    dest = Path(destination_path)
    template_dir = Path(__file__).parent / "experiment_template"
    pixi_toml = template_dir / "pixi.toml"
    pixi_toml.copy_into(dest)
