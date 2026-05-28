"""A Prefect workflow for processing a raw Tiled run."""

import asyncio

from prefect import flow
from tiled.client import from_profile

from ._fit_fluorescence import fit_run_fluorescence

@flow()
async def process_run(run_uid: str, raw_profile: str = "oaty-bar", results_profile: str = "oaty-bar-results"):
    """Execute several processing steps in parallel for a Tiled Bluesky run.

    Parameters
    ==========
    run_uid
      The UUID for the run as found in the raw catalog.
    raw_profile
      The name of the Tiled profile to use for finding the raw Bluesky
      run data.
    results_profile
      The name of the Tiled profile to use for storing calculated
      results.

    """
    # Load the necessary Tiled catalogs
    raw_catalog = from_profile(raw_profile)
    run = raw_catalog[run_uid]
    results_catalog = from_profile(results_profile)
    results = await asyncio.gather(
        fit_run_fluorescence(
            run=run, results_catalog=results_catalog
        ),
        return_exceptions=True,
    )
    exceptions = [exc for exc in results if isinstance(exc, BaseException)]
    if any(exceptions):
        raise ExceptionGroup("Run processing failed", exceptions)
