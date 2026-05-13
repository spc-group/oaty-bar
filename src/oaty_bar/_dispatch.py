import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Sequence

import dmax
from websockets.asyncio.client import connect

log = logging.getLogger("oaty-bar")


async def dispatch_new_runs(websocket, dm_api):
    remote_addr = ":".join([str(val) for val in websocket.remote_address])
    log.info(f"Listening for new runs on {remote_addr}.")
    async for msg in websocket:
        msg = json.loads(msg)
        metadata = msg.get("metadata", {})
        if "stop" not in metadata:
            # Run is not finished yet, so ignore it for now
            continue
        if metadata["stop"].get("exit_status", "") != "success":
            continue
        if "dm_exp" not in metadata["start"]:
            continue
        # The data management API can tell us where the experiment should export data
        experiment_name = metadata["start"].get("dm_exp")
        if experiment_name is None:
            # No experiment, so nowhere to export data
            return
        experiment = await dm_api.experiment(name=experiment_name)
        target_folder = Path(experiment.data_path)
        # The specific workflow to execute depends on the goal of the run
        workflow = "simple"
        await dm_api.submit_processing_job(
            workflow=workflow,
            run_uid=metadata["start"]["uid"],
            target_folder=str(target_folder),
        )


async def run_dispatcher(
    websocket_uri: str,
    dm_username: str = "",
    dm_password: str = "",
    dm_station_name: str = "",
    dm_scheduling_uri: str = "",
    dm_data_storage_uri: str = "",
    dm_processing_uri: str = "",
):
    dm_api = dmax.AsyncClient(
        username=dm_username,
        password=dm_password,
        station_name=dm_station_name,
        scheduling_uri=dm_scheduling_uri,
        data_storage_uri=dm_data_storage_uri,
        processing_uri=dm_processing_uri,
    )
    # Error out here if we can't access the DM API
    log.info("Checking DM API connections…")
    await asyncio.gather(
        dm_api.workflows(),
        dm_api.experiments(),
    )
    log.info("DM APIs connected successfully.")
    async with connect(websocket_uri) as websocket:
        await dispatch_new_runs(websocket, dm_api)


def main(argv: Sequence[str] | None = None):
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="dispatch-workflows",
        description="Listen for new Tiled runs, and dispatch data management workflows in response.",
    )
    parser.add_argument(
        "websocket_uri", help="URI of a websocket to listen for new runs."
    )
    parser.add_argument(
        "--dm-username",
        default="",
        help="Username for accessing the data management system.",
    )
    parser.add_argument(
        "--dm-password",
        default="",
        help="Password for accessing the data management system.",
    )
    parser.add_argument(
        "--dm-station-name",
        default="",
        help="Name assigned to this station by the data management system.",
    )
    parser.add_argument(
        "--dm-scheduling-uri",
        default="",
        help="URI for accessing the data management scheduling (BSS) API.",
    )
    parser.add_argument(
        "--dm-data-storage-uri",
        default="",
        help="URI for accessing the data management storage (DS) API.",
    )
    parser.add_argument(
        "--dm-processing-uri",
        default="",
        help="URI for accessing the data management processing (PROC) API.",
    )
    args = parser.parse_args(argv)
    asyncio.run(
        run_dispatcher(
            websocket_uri=args.websocket_uri,
            dm_username=args.dm_username,
            dm_password=args.dm_password,
            dm_station_name=args.dm_station_name,
            dm_scheduling_uri=args.dm_scheduling_uri,
            dm_data_storage_uri=args.dm_data_storage_uri,
            dm_processing_uri=args.dm_processing_uri,
        )
    )
