import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Sequence

log = logging.getLogger("oaty-bar")


async def dispatch_new_runs(websocket, dm_api):
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
        # The data management API can tell us where the experiment should save data
        experiment = await dm_api.experiment(name=metadata["start"]["dm_exp"])
        target_folder = Path(experiment.data_path)
        # The specific workflow to execute depends on the goal of the run
        workflow = "simple"
        await dm_api.submit_processing_job(
            workflow=workflow,
            run_uid=metadata["start"]["uid"],
            target_folder=str(target_folder),
        )


async def run_dispatcher(args):
    websocket = None
    dm_api = None
    await dispatch_new_runs(websocket, dm_api)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="dispatch-workflows",
        description="Listen for new Tiled runs, and dispatch data management workflows in response.",
    )
    args = parser.parse_args(argv)
    asyncio.run(run_dispatcher(args))
