"""A Prefect workflow for processing a raw Tiled run."""

import asyncio

from prefect import flow, task

from ._fit_fluorescence import fit_fluorescence

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
    results = await asyncio.gather(
        fit_fluorescence(
            run_uid=run_uid,
            raw_profile=raw_profile,
            results_profile=results_profile
        ),
        return_exceptions=True,
    )
    exceptions = [exc for exc in results if isinstance(exc, BaseException)]
    if any(exceptions):
        raise ExceptionGroup("Run processing failed", exceptions)
