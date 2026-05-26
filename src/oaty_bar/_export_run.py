import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from prefect import flow
from tiled.client import from_profile
from tiled.profiles import get_default_profile_name
from tiled.queries import Eq

from ._export_hdf import build_file_name, serialize_hdf


@flow()
async def export_run(
    uid: str,
    target_dir: str,
    raw_profile: str = get_default_profile_name(),
    results_profile: str = get_default_profile_name(),
    force: bool = False,
    semaphore: asyncio.Semaphore | None = None,
):
    """Export a Tiled run with UID *uid* to files in *target_dir*.

    The names of the resulting files will be generated from the run
    metadata. They will include the first portion of the UID, so
    presumably it will be unique. If a file of the same name already
    exists in *target_dir*, this operation will fail unless *force* is
    True, in which case the existing files will be overwritten.

    Parameters
    ==========
    uid
      The UID of the Bluesky run to read from in the Tiled catalog.
    target_dir
      An existing folder in which to create a new HDF5 file.
    raw_profile
      The name of the Tiled profile to use for reading Bluesky runs.
    results_profile
      The name of the Tiled profile to use for reading processed
      results data.
    sempahore
      A locking semaphore to limit concurrent API connections. If
      omitted, a default will be created.

    """
    target_dir_ = Path(target_dir)
    if semaphore is None:
        semaphore = asyncio.Semaphore(10)
    raw_catalog = from_profile(raw_profile)
    run = raw_catalog[uid]
    if results_profile:
        results_catalog = from_profile(results_profile)
        results_runs = results_catalog.search(Eq("run_uid", uid))
    else:
        results_runs = None
    target_file = target_dir_ / build_file_name(run.metadata)
    await serialize_hdf(
        buff=target_file,
        run=run,
        results_runs=results_runs,
        force=force,
        semaphore=semaphore,
    )


def main(argv: Sequence[str] | None = None):
    """Main entry-point for exporting data files for a given run."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="export-run",
        description="A prefect flow that exports files for a given Bluesky run",
    )
    parser.add_argument("--uid", help="The UID of the bluesky run to export.",         )
    parser.add_argument(
        "--target_dir", help="The DM directory to receive the exported file.",         
    )
    parser.add_argument(
        "--raw-profile", help="The name of the Tiled profile used for raw runs."
    )
    parser.add_argument(
        "--results-profile",
        help="The name of the Tiled profile used for processed run reuslts data.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Provide verbose information.",
    )
    parser.add_argument(
        "--max-workers",
        default=10,
        type=int,
        help="Number of concurrent network connections to allow. Higher number can improve performance but also overload the server.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Start worker listening for new work instead of running immediately.",
    )
    args = parser.parse_args(argv)
    # Do the actual exporting
    if args.deploy:
        export_run.serve()
    else:
        semaphore = asyncio.Semaphore(args.max_workers)
        asyncio.run(
            export_run(
                uid=args.uid,
                target_dir=args.target_dir,
                force=args.force,
                raw_profile=args.raw_profile or "",
                results_profile=args.results_profile or "",
                semaphore=semaphore,
            )
        )
