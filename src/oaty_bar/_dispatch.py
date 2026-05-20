import argparse
import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import dmax
import httpx
from websockets.asyncio.client import ClientConnection, connect

from ._config import OatyBarConfig, load_config
from .workflows import load_workflow

log = logging.getLogger("oaty-bar")


def load_dm_apis(config: OatyBarConfig):
    """Load a set of data management API clients from a configuration file."""
    api_models = config.data_management_stations.items()
    apis = {key: dmax.AsyncClient(**model.model_dump()) for key, model in api_models}
    return apis


async def process_msg(msg: str, dm_apis: Mapping[str, dmax.AsyncClient]):
    """Process an individual websocket message."""
    log.debug(f"Received websocket message: {msg}")
    payload = json.loads(msg)
    if payload["type"] != "container-child-metadata-updated":
        # Not a message we can process, so skip it for now
        return
    metadata = payload.get("metadata", {})
    uid = metadata["start"].get("uid", "<unknown UID>")
    # Guards to make sure we have all the right metadata to proceed
    if "stop" not in metadata:
        # Run is not finished yet, so ignore it for now
        log.debug(f"Run `{uid}` is not finished, skipping workflows.")
        return
    if metadata["stop"].get("exit_status", "") != "success":
        log.info(f"Run `{uid}` did not succeed, skipping workflows.")
        return
    if "dm_exp" not in metadata["start"]:
        log.warning(
            f"Run `{uid}` is missing metadata key 'dm_exp'. Skipping workflows."
        )
        return
    if "dm_station_name" not in metadata["start"]:
        log.warning(
            f"Run `{uid}` is missing metadata key 'dm_station_name'. Skipping workflows."
        )
        return
    station_name = metadata["start"]["dm_station_name"]
    if station_name not in dm_apis:
        log.warning(
            f"Unknown data management station name '{station_name}'. Skipping workflows."
        )
        return
    # The data management API can tell us where the experiment should export data
    experiment_name = metadata["start"]["dm_exp"]
    dm_api = dm_apis[station_name]
    try:
        experiment = await dm_api.experiment(name=experiment_name)
    except httpx.HTTPStatusError as exc:
        log.error(exc)
        return
    if getattr(experiment, "folders_are_managed", False):
        target_folder = Path(experiment.data_path)
        log.info(f"Detected managed folder structure: data='{target_folder}'.")
    else:
        # Default to root storage path if specific folders are not set up
        target_folder = Path(experiment.storage_path)
        log.info(f"No managed folder structure detector: storage='{target_folder}'.")
    # Make sure the workflow is up to date in the API
    workflow_name = "simple"
    workflow = load_workflow(workflow_name, username=dm_api.username)
    existing_workflows = await dm_api.workflows()
    existing_workflows = [
        workflow for workflow in existing_workflows if workflow.name == workflow_name
    ]
    log.info(workflow)
    if len(existing_workflows) == 0:
        log.info(f"Adding workflow {workflow_name} to station {station_name}.")
        await dm_api.add_workflow(workflow)
    elif getattr(existing_workflows[0], "version", 0) < getattr(
        workflow, "version", -1
    ):
        log.info(
            f"Updating workflow {workflow_name} due to new version {getattr(workflow, 'version')}."
        )
        await dm_api.set_workflow(name=workflow_name, workflow=workflow)
    # The specific workflow to execute depends on the goal of the run
    await dm_api.submit_processing_job(
        workflow=workflow.name,
        filePath="/dev/null",
        run_uid=metadata["start"]["uid"],
        target_folder=str(target_folder),
    )


async def dispatch_new_runs(
    websocket: ClientConnection, dm_apis: Mapping[str, dmax.AsyncClient]
):
    remote_addr = ":".join([str(val) for val in websocket.remote_address])
    log.info(f"Listening for new runs on {remote_addr}.")
    async for msg in websocket:
        try:
            await process_msg(msg=msg, dm_apis=dm_apis)
        except Exception as exc:
            log.exception(exc)
            continue


async def run_dispatcher(config_file: Path):
    """Run the dispatcher in a continuous loop as defined in *config_file*.

    The dispatcher with listen to Tiled for runs that have completed,
    and submit workflows to various data managament APIs.

    """
    config = load_config(config_file)
    # Error out here if we can't access the DM API
    dm_apis = load_dm_apis(config)
    log.info("Checking DM API connections…")
    coros = [
        coro
        for api in dm_apis.values()
        for coro in (api.workflows(), api.experiments())
    ]
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
        "-d", "--debug", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run_dispatcher(config_file=Path(args.config_file)))
