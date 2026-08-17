"""General routines for interacting with the Tiled catalog."""

import asyncio

from tiled.client.container import Container
from tiled.queries import Eq


async def results_container(run_uid: str, catalog: Container) -> Container:
    """Return (and create?) the results container for the given raw run
    UID.

    If a results container exists, it will be returned. If not, a new
    container will be created and returned.

    Parameters
    ==========
    run_uid
      The UID of the raw run that this scan references.
    catalog
      The Tiled catalog in which the results container should exist.

    """
    existing_runs = catalog.search(Eq("run_uid", run_uid))
    if len(existing_runs) == 0:
        run = await asyncio.to_thread(
            catalog.create_container, metadata={"run_uid": run_uid}
        )
        return run
    else:
        return await asyncio.to_thread(existing_runs.values().first)
