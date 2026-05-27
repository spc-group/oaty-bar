import uuid
import argparse
import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from websockets.asyncio.client import ClientConnection, connect
from prefect.events import emit_event
from tiled.client import from_profile
from tiled.profiles import get_default_profile_name

from .workflows import load_workflow

log = logging.getLogger("oaty-bar")


async def process_msg(msg: str, instance_uuid: str):
    """Process an individual websocket message."""
    payload = json.loads(msg)
    if payload["type"] != "container-child-metadata-updated":
        # Not a message we can process, so skip it for now
        return
    metadata = payload.get("metadata", {})
    uid = metadata.get("start", {}).get("uid", "")
    # Guards to make sure we have all the right metadata to proceed
    if "stop" not in metadata:
        # Run is not finished yet, so ignore it for now
        log.debug(f"Run `{uid or '<unknown UID>'}` is not finished, skipping.")
        return
    # Create a prefect event so it can be routed to the correct workflows
    resource = {
        "prefect.resource.id": f"oaty-bar.dispatcher.{instance_uuid}",
        "run_uid": uid,
    }
    log.info(f"Emitting events for resource: {resource}")
    emit_event("bluesky.run.stopped", resource=resource, payload=payload)
    if exit_status := metadata.get("stop", {}).get("exit_status"):
        emit_event(f"bluesk.run.{exit_status}", resource=resource, payload=payload)


async def dispatch_new_runs(
        websocket: ClientConnection,
):
    instance_uuid=uuid.uuid4()
    async for msg in websocket:
        log.info(f"Received message {msg}")
        try:
            await process_msg(msg=msg, instance_uuid=instance_uuid)
        except Exception as exc:
            log.exception(exc)
            continue


async def run_dispatcher(*, tiled_profile: str = "", config_file: Path | None = None):
    """Run the dispatcher in a continuous loop as defined in *config_file*.

    The dispatcher with listen to Tiled for runs that have completed,
    and submit workflows to various data managament APIs.

    """
    instance_uuid = uuid.uuid4()
    # Extract the websocket URI from the Tiled profile info
    if not tiled_profile:
        tiled_profile = get_default_profile_name()
    tiled_client = from_profile(tiled_profile)
    api_uri, _ = tiled_client.uri.split("/api/v1/")
    path = "/".join(tiled_client.path_parts)
    ws_uri = f"{api_uri}/api/v1/stream/single/{path}"
    scheme, rest = ws_uri.split("://")
    ws_uri = f"{scheme.replace('http', 'ws')}://{rest}"
    # Launch the websocket loop
    async with connect(ws_uri) as websocket:
        log.info(f"Listening for new runs on {ws_uri}")
        await dispatch_new_runs(websocket)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="dispatch-workflows",
        description="Listen for new Tiled runs, and dispatch data management workflows in response.",
    )
    parser.add_argument(
        "-p", "--tiled_profile",
        type=str,
        help="The name of the Tiled profile to monitor for new runs.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run_dispatcher(tiled_profile=args.tiled_profile))
