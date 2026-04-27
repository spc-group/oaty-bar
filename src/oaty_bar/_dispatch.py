import asyncio
from typing import Sequence
import argparse
import logging

log = logging.getLogger("oaty-bar")


async def run_dispatcher():
    while True:
        # Just keep the event loop turning
        await asyncio.sleep(1)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="dispatch-workflows",
        description="Listen for new Tiled runs, and dispatch data management workflows in response.",
    )
    args = parser.parse(argv)
    asyncio.run(run_dispatcher())
