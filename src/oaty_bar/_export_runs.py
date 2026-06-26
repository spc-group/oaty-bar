"""A Prefect flow for created exported files from a Tiled Bluesky run."""

import argparse
import asyncio
import datetime as dt
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from textwrap import dedent
from typing import Any

from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRun
from prefect import flow
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput
from prefect.logging import get_run_logger
from tiled import queries
from tiled.client import from_profile
from tiled.client.container import Container

from ._export_hdf import build_file_name, serialize_hdf
from ._export_tsv import serialize_tsv
from .exceptions import NoRuns, TooManyRuns


def valid_datetime(value: str) -> float:
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {value!r}")


@dataclass(frozen=True)
class QuerySet:
    before: int | float | None = None
    after: int | float | None = None
    exit_status: str | None = None
    plan_name: str | None = None
    sample_name: str | None = None
    sample_formula: str | None = None
    scan_name: str | None = None
    edge: str | None = None
    proposal: str | None = None
    esaf: str | None = None
    beamline: str | None = None
    uid: str | None = None

    def queries(self) -> list[tuple[Any, None, str]]:
        """Build the individual queries that should be applied for this
        queryset.

        """
        # Process the queries parameters into actual queries
        qs = []
        query_params = [
            # filter_name: (query type, metadata key)
            (self.exit_status, queries.Eq, "stop.exit_status"),
            (self.plan_name, queries.Eq, "start.plan_name"),
            (self.sample_name, queries.Contains, "start.sample_name"),
            (self.sample_formula, queries.Contains, "start.sample_formula"),
            (self.scan_name, queries.Contains, "start.scan_name"),
            (self.edge, queries.Contains, "start.edge"),
            (self.proposal, queries.Eq, "start.proposal_id"),
            (self.beamline, queries.Contains, "start.beamline_id"),
            (self.esaf, queries.Eq, "start.esaf_id"),
            (self.before, partial(queries.Comparison, "le"), "stop.time"),
            (self.after, partial(queries.Comparison, "ge"), "start.time"),
            (self.uid, queries.Eq, "start.uid"),
        ]
        for arg, query, key in query_params:
            if arg is not None:
                qs.append(query(key, arg))
        return qs

    def apply(self, container: Container):
        runs = container
        for query in self.queries():
            runs = runs.search(query)
        return runs


async def _export_run(
    run: BlueskyRun,
    target_dir: str | Path = "",
    raw_profile: str = "oaty-bar",
    results_profile: str = "oaty-bar-results",
    force: bool = True,
):
    """Export a Tiled run with UID *uid* to files in *target_dir*.

    The names of the resulting files will be generated from the run
    metadata. They will include the first portion of the UID, so
    presumably it will be unique. If a file of the same name already
    exists in *target_dir*, this operation will fail unless *force* is
    True, in which case the existing files will be overwritten.

    If *target_dir* is not provided, a default destination for the new
    file will be determined from the corresponding DM experiment
    metadata.

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

    """
    log = get_run_logger()
    run_uid = run.metadata["start"]["uid"]
    if results_profile:
        results_catalog = from_profile(results_profile)
        results_runs = results_catalog.search(queries.Eq("run_uid", run_uid))
    else:
        log.warning("No results profile specified, only raw data will be exported.")
        results_runs = None
    # DM experiments contain the export path, which is our default
    start_doc = run.metadata.get("start", {})
    if not target_dir:
        if "beamline_id" not in start_doc:
            log.warning(
                f"No 'beamline_id' in metadata for run '{run_uid}', specify a *target_dir*."
            )
            return
        if "dm_exp" not in start_doc:
            log.warning(
                f"No 'dm_exp' in metadata for run '{run_uid}', specify a *target_dir*."
            )
            return
        beamline_id = run.metadata["start"]["beamline_id"]
        exp_name = run.metadata["start"]["dm_exp"]
        target_dir_ = Path("/net/s25data/export") / beamline_id / exp_name
        target_dir_.mkdir(exist_ok=True, parents=False)
    else:
        target_dir_ = Path(target_dir)
    hdf_file = target_dir_ / build_file_name(run.metadata, extension=".hdf")
    coros = [
        serialize_hdf(
            buff=hdf_file,
            run=run,
            results_runs=results_runs,
            force=force,
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


class ContinueDecision(RunInput):
    approve: bool
    total_exports: int


@flow()
async def export_runs(
    target_dir: Path | None = None,
    *,
    run_uid: str | None = None,
    exit_status: str | None = "success",
    plan_name: str | None = None,
    sample_name: str | None = None,
    sample_formula: str | None = None,
    scan_name: str | None = None,
    xray_edge: str | None = None,
    dm_exp: str | None = None,
    beamline: str | None = None,
    before: dt.datetime | None = None,
    after: dt.datetime | None = None,
    raw_profile: str = "oaty-bar",
    results_profile: str = "oaty-bar-results",
    force: bool = False,
    max_runs: int = 1,
):
    """Export Tiled runs as HDF and XDI/TSV files.

    The names of the resulting files will be generated from the run
    metadata. They will include the first portion of the UID, so
    presumably it will be unique. If a file of the same name already
    exists in *target_dir*, this operation will fail unless *force* is
    True, in which case the existing files will be overwritten.

    If *target_dir* is not provided, a default destination for the new
    file will be determined from the corresponding DM experiment
    metadata.

    To avoid accidentally exporting the entire catalog, this operation
    fails if the number of runs exceeds *max_runs*. Parameters should
    be specified to specify a specific set of runs to export.

    Parameters
    ==========

    target_dir
      An existing folder in which to create a new HDF5 file.
    run_uid
      The UID of the Bluesky run to read from in the Tiled catalog.
    exit_status
      Only includes runs with this exit status (None matches all
      scans).
    plan_name
      Only include runs containing this plan name.
    sample_name
      Only include runs containing this sample name.
    sample_formula
      Only include runs containing this chemical formula.
    scan_name
      Only include runs with this scan name.
    xray_edge
      Only include runs with that specific this x-ray absorption edge
      (e.g. "Ni-K")
    dm_exp
      Only include runs with this data management experiment name.
    beamline
      Only include runs from this beamline.
    before
      Only include runs stopped before this date-time.
    after
      Only include runs started after this date-time.
    force
      If true, overwrite existing files.
    max_runs
      Refuse to export if the number of runs is greater than this
      value.
    raw_profile
      The name of the Tiled profile to use for reading Bluesky runs.
    results_profile
      The name of the Tiled profile to use for reading processed
      results data.

    """
    log = get_run_logger()
    # Get only the runs requested by the user
    queries = QuerySet(
        before=before.timestamp() if before is not None else None,
        after=after.timestamp() if after is not None else None,
        exit_status=exit_status,
        plan_name=plan_name,
        sample_name=sample_name,
        sample_formula=sample_formula,
        scan_name=scan_name,
        edge=xray_edge,
        beamline=beamline,
        uid=run_uid,
    )
    catalog = from_profile(raw_profile)
    runs = queries.apply(catalog)
    if len(runs) == 0:
        raise NoRuns("No runs found matching query parameters.")
    if len(runs) > max_runs:
        # We need to make sure we're not exporting too many runs
        description_md = dedent(f"""
            # Approval Needed
            
            This flow will export **{len(runs)} runs**, higher than the expected maximum of {max_runs}. Please confirm that this is correct.
            
            Optionally, use *total_exports* to only export the first N runs.
            
            """)
        response = await pause_flow_run(
            wait_for_input=ContinueDecision.with_initial_data(
                description=description_md, total_exports=len(runs)
            )
        )
        if response.approve:
            max_runs = response.total_exports
            log.info(f"Approved to export {max_runs} run(s).")
        else:
            log.error(f"Exporting {len(runs)} run(s) disapproved.")
            raise TooManyRuns(
                f"Queries produced {len(runs)} runs but we're limited to {max_runs}. Either provide additional queries or increase *max_runs*."
            )
    # Start the exporters in parallel
    do_export = partial(
        _export_run,
        target_dir=str(target_dir) if target_dir is not None else "",
        raw_profile=raw_profile,
        results_profile=results_profile,
        force=force,
    )
    coros = (do_export(run) for run in islice(runs.values(), max_runs))
    results = await asyncio.gather(*coros, return_exceptions=True)
    exceptions = [exc for exc in results if isinstance(exc, Exception)]
    if any(exceptions):
        raise ExceptionGroup("Export runs failed", exceptions)


def parse_metadata(md):
    """Load the metadata for *runs* and produce a structure dataframe."""
    # columns = ["uid", "esaf_id", "start_time", "exit_status", "beamline", "sample_name", "scan_name", "plan_name", "experiment_name", "filename"]
    uid = md["start"]["uid"]
    esaf = md["start"].get("esaf_id")
    start_time = md["start"].get("time", 0)
    sample_name = md["start"].get("sample_name")
    scan_name = md["start"].get("scan_name")
    plan_name = md["start"].get("plan_name")
    pi_name = None  # TODO: Extract the PI name
    start_dt = dt.datetime.fromtimestamp(start_time)
    experiment = (
        f"{pi_name if pi_name else 'noPI'}_"
        f"{start_dt.strftime('%Y-%m')}_"
        f"{esaf if esaf else 'noesaf'}"
    )
    # Decide on how to structure the file storage
    uid_base = uid.split("-")[0]
    bits = [
        start_dt.strftime("%Y%m%d%H%M"),
        sample_name,
        scan_name,
        plan_name,
        uid_base,
    ]
    bits = [bit for bit in bits if bit not in ["", None]]
    base_name = "-".join(bits)
    base_name = re.sub(r"[ ]", "_", base_name)
    base_name = re.sub(r"[/]", "", base_name)
    return {
        "uid": uid,
        "esaf_id": esaf,
        "start_time": start_time,
        "exit_status": md.get("stop", {}).get("exit_status"),
        "beamline": md["start"].get("beamline_id"),
        "sample_name": sample_name,
        "scan_name": scan_name,
        "plan_name": plan_name,
        "experiment_name": experiment,
        "filename": base_name,
    }


def main(argv: Sequence[str] | None = None):
    """Main entry-point for exporting data files for a collection of runs."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="export-runs",
        description="""Export runs from the database as files on disk.

        Connects to a Prefect server and executes as a flow.
        """,
    )
    parser.add_argument(
        "--target-dir",
        help="The directory for storing files.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--raw-profile",
        help="The name of the Tiled profile used for raw runs.",
        default="oaty-bar",
    )
    parser.add_argument(
        "--results-profile",
        help="The name of the Tiled profile used for processed run reuslts data.",
    )
    # Arguments for filtering runs
    # parser.add_argument(
    #     "--all",
    #     help="Also include scans that did not complete successfully.",
    #     action="store_true",
    # )
    parser.add_argument(
        "--exit-status",
        help="Export runs with specific exit status.",
        default="success",
        choices=["success", "fail", "abort", "all"],
    )
    parser.add_argument("--plan", help="Export runs with plan name.", type=str)
    parser.add_argument("--sample", help="Export runs with this sample name.", type=str)
    parser.add_argument(
        "--formula",
        help="Export runs with samples matching this chemical formula.",
        type=str,
    )
    parser.add_argument("--scan", help="Export runs with this scan name.", type=str)
    parser.add_argument(
        "--edge", help="Export runs that contain the given X-ray edge. E.g. 'Ni-K'"
    )
    parser.add_argument(
        "--dm-exp",
        help="Export runs associated with this data management (DM) experiment.",
        type=str,
    )
    parser.add_argument(
        "--beamline",
        help="Export runs only on this beamline. Incomplete matches are allowed, so '25-ID' will match both '25-ID-C' and '25-ID-D'.",
    )
    parser.add_argument(
        "--before",
        "-B",
        help="Only include runs before this timestamp. E.g. 2025-04-22T8:00:00.",
        type=valid_datetime,
    )
    parser.add_argument(
        "--after",
        "-A",
        help="Only include runs after this ISO datetime. E.g. 2025-04-22T8:00:00.",
        type=valid_datetime,
    )
    parser.add_argument("--uid", help="Export runs with this UID.", type=str)

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="The maximum number of runs expected as the results of the given queries. This is to prevent accidental dumps of an excessive number of runs.",
    )

    args = parser.parse_args(argv)
    asyncio.run(
        export_runs(
            target_dir=args.target_dir,
            run_uid=args.uid,
            exit_status=None if args.exit_status == "all" else args.exit_status,
            plan_name=args.plan,
            sample_name=args.sample,
            sample_formula=args.formula,
            scan_name=args.scan,
            xray_edge=args.edge,
            dm_exp=args.dm_exp,
            beamline=args.beamline,
            before=args.before,
            after=args.after,
            raw_profile=args.raw_profile or "",
            results_profile=args.results_profile or "",
            force=args.force,
            max_runs=args.max_runs,
        )
    )
