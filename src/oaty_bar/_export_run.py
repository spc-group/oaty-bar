"""A Prefect flow for created exported files from a Tiled Bluesky run."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from prefect import flow
from prefect.logging import get_run_logger
from tiled.client import from_profile
from tiled.queries import Eq

from ._export_hdf import build_file_name, serialize_hdf
from ._export_tsv import serialize_tsv


@flow()
async def export_run(
    run_uid: str,
    *,
    target_dir: str = "",
    raw_profile: str = "oaty-bar",
    results_profile: str = "oaty-bar-results",
    force: bool = True,
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
    run_uid
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
    log = get_run_logger()
    if semaphore is None:
        semaphore = asyncio.Semaphore(10)
    raw_catalog = from_profile(raw_profile)
    run = raw_catalog[run_uid]
    if results_profile:
        results_catalog = from_profile(results_profile)
        results_runs = results_catalog.search(Eq("run_uid", run_uid))
    else:
        log.warning("No results profile specified, only raw data will be exported.")
        results_runs = None
    # DM experiments contain the export path, which is our default
    if not target_dir:
        beamline_id = run.metadata["start"]["beamline_id"]
        exp_name = run.metadata["start"]["dm_exp"]
        target_dir_ = Path("/net/s25data/export") / beamline_id / exp_name
        target_dir_.mkdir(exist_ok=True, parents=False)
        # dmax_client = await load_client(run.metadata['start']['dm_station_name'], asyncio=True)
        # dm_exp = await dmax_client.experiment(name=run.metadata['start']['dm_exp'])
        # target_dir = dm_exp.data_path
    else:
        target_dir_ = Path(target_dir)
    hdf_file = target_dir_ / build_file_name(run.metadata, extension=".hdf")
    coros = [
        serialize_hdf(
            buff=hdf_file,
            run=run,
            results_runs=results_runs,
            force=force,
            semaphore=semaphore,
        ),
    ]
    # Not all scans are compatile with TSV exporting (e.g. fly scans)
    can_make_tsv = "primary" in run.keys()
    if can_make_tsv:
        # Exporting to an XDI file requires certain metadata, otherwise we
        # just export to TSV
        plan_name = run.metadata.get("start", {}).get("plan_name")
        has_edge = "edge" in run.metadata.get("start", {}).keys()
        use_xdi = plan_name == "xafs_scan" and has_edge
        extension = ".xdi" if use_xdi else ".tsv"
        tsv_file = target_dir_ / build_file_name(run.metadata, extension=extension)
        coros.append(
            serialize_tsv(
                filepath=tsv_file,
                run=run,
                use_xdi=use_xdi,
            ),
        )
    else:
        log.warning("Cannot make TSV or XDI file. Skipping.")
    # Now that we've built the tasks, we can execute them in parallel
    results = await asyncio.gather(*coros, return_exceptions=True)
    exceptions = [exc for exc in results if isinstance(exc, Exception)]
    if any(exceptions):
        raise ExceptionGroup("Export runs failed", exceptions)


def main(argv: Sequence[str] | None = None):
    """Main entry-point for exporting data files for a given run."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="export-run",
        description="A prefect flow that exports files for a given Bluesky run",
    )
    parser.add_argument(
        "--uid",
        help="The UID of the bluesky run to export.",
    )
    parser.add_argument(
        "--target_dir",
        help="The DM directory to receive the exported file.",
    )
    parser.add_argument(
        "--raw-profile", help="The name of the Tiled profile used for raw runs.", default="oaty-bar",
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
                run_uid=args.uid,
                target_dir=args.target_dir,
                force=args.force,
                raw_profile=args.raw_profile or "",
                results_profile=args.results_profile or "",
                semaphore=semaphore,
            )
        )
