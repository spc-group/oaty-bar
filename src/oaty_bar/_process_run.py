"""A Prefect workflow for processing a raw Tiled run."""

import argparse
import asyncio
from collections.abc import Sequence

from prefect import flow
from prefect.task_runners import ProcessPoolTaskRunner
from tiled.client import from_profile

from ._fit_fluorescence import fit_run_fluorescence


@flow(task_runner=ProcessPoolTaskRunner())
async def process_run(
    run_uid: str,
    raw_profile: str = "oaty-bar",
    results_profile: str = "oaty-bar-results",
):
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
        fit_run_fluorescence(run=run, results_catalog=results_catalog),
        return_exceptions=True,
    )
    exceptions = [exc for exc in results if isinstance(exc, Exception)]
    if any(exceptions):
        raise ExceptionGroup("Run processing failed", exceptions)


def main(argv: Sequence[str] | None = None):
    """Main entry-point for processing a successful Bluesky run."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="process-run",
        description="Apply corrections and fit fluorescence spectra to produce elemental contributions",
    )
    parser.add_argument(
        "run_uid", help="The UID of the bluesky run to process.", type=str
    )
    parser.add_argument(
        "--raw-profile", help="The name of the Tiled profile used for raw runs."
    )
    parser.add_argument(
        "--results-profile",
        help="The name of the Tiled profile used for processed run data.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="How many worker threads will be processing spectra.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Start worker listening for new work instead of running immediately.",
    )
    args = parser.parse_args(argv)
    # Do the actual exporting
    if args.deploy:
        process_run.serve()
    else:
        asyncio.run(
            process_run(
                run_uid=args.run_uid,
                raw_profile=args.raw_profile,
                results_profile=args.results_profile,
            )
        )
