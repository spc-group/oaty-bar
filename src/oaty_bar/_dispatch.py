import tomllib
import argparse
import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Sequence

import dmax
from websockets.asyncio.client import connect, ClientConnection
from pydantic import BaseModel, Field

log = logging.getLogger("oaty-bar")

class TiledConfig(BaseModel):
    websocket_uri: str


class DataManagementStation(BaseModel):
    """Maps directly onto ``dmax.AsyncClient()``."""
    username: str
    password: str
    station_name: str
    scheduling_uri: str = ""
    data_storage_uri: str = ""
    processing_uri: str = ""


class OatyBarConfig(BaseModel):
    tiled: TiledConfig
    data_management_stations: Mapping[str, DataManagementStation] = {}


def load_dm_apis(config: OatyBarConfig):
    """Load a set of data management API clients from a configuration file."""
    api_models = config.data_management_stations.items()
    apis = {key: dmax.AsyncClient(**model.model_dump()) for key, model in api_models}
    return apis


async def dispatch_new_runs(websocket: ClientConnection, dm_apis: Mapping[str, dmax.AsyncClient]):
    remote_addr = ":".join([str(val) for val in websocket.remote_address])
    log.info(f"Listening for new runs on {remote_addr}.")
    async for msg in websocket:
        log.debug(f"Received websocket message: {msg}")
        msg = json.loads(msg)
        if msg['type'] != "container-child-metadata-updated":
            # Not a message we can process, so skip it for now
            continue
        metadata = msg.get("metadata", {})
        uid = metadata['start'].get('uid', "<unknown UID>")
        # Guards to make sure we have all the right metadata to proceed
        if "stop" not in metadata:
            # Run is not finished yet, so ignore it for now
            log.debug(f"Run `{uid}` is not finished, skipping workflows.")
            continue
        if metadata["stop"].get("exit_status", "") != "success":
            log.info(f"Run `{uid}` did not succeed, skipping workflows.")
            continue
        if "dm_exp" not in metadata["start"]:
            log.warning(f"Run `{uid}` is missing metadata key 'dm_exp'. Skipping workflows.")
            continue
        if "dm_station_name" not in metadata['start']:
            log.warning(f"Run `{uid}` is missing metadata key 'dm_station_name'. Skipping workflows.")
            continue
        station_name = metadata['start']['dm_station_name']
        if station_name not in dm_apis:
            log.warning(f"Unknown data management station name '{station_name}'. Skipping workflows.")
            continue
        # The data management API can tell us where the experiment should export data
        experiment_name = metadata["start"]["dm_exp"]
        dm_api = dm_apis[station_name]
        experiment = await dm_api.experiment(name=experiment_name)
        target_folder = Path(experiment.data_path)
        # The specific workflow to execute depends on the goal of the run
        workflow = "simple"
        await dm_api.submit_processing_job(
            workflow=workflow,
            run_uid=metadata["start"]["uid"],
            target_folder=str(target_folder),
        )


async def run_dispatcher(config_file: Path):
    """Run the dispatcher in a continuous loop as defined in *config_file*.

    The dispatcher with listen to Tiled for runs that have completed,
    and submit workflows to various data managament APIs.

    """
    with open(config_file, mode='rb') as config_fd: 
        cfg_dict = tomllib.load(config_fd)
    config = OatyBarConfig(**cfg_dict)
    # Error out here if we can't access the DM API
    dm_apis = load_dm_apis(config)
    log.info("Checking DM API connections…")
    coros = [coro for api in dm_apis.values() for coro in (api.workflows(), api.experiments())]
    await asyncio.gather(*coros)
    log.info("DM APIs connected successfully.")
    async with connect(config.tiled.websocket_uri) as websocket:
        await dispatch_new_runs(websocket, dm_apis)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="dispatch-workflows",
        description="Listen for new Tiled runs, and dispatch data management workflows in response.",
    )
    parser.add_argument(
        "config_file",
        help="Path to a configuration TOML file describing how the dispatcher should operate.",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run_dispatcher(config_file=Path(args.config_file)))
