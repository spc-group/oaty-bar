import argparse
import asyncio
import logging
import time
from asyncio import TaskGroup
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor as ThreadExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from chemformula import ChemFormula
from larch import Group
from larch.xrf.xrf_model import XRF_Model as XRFModel
from pint import Quantity, UnitRegistry
from prefect import flow
from prefect.assets import materialize
from pybaselines import Baseline
from tiled.client import from_profile
from tiled.client.array import ArrayClient
from tiled.client.container import Container
from tiled.queries import Eq

log = logging.getLogger("oaty-bar")
ureg = UnitRegistry()


@dataclass(frozen=True, eq=True)
class FitResult:
    goodness: float
    weights: Mapping[str, float]
    predicted: np.ndarray
    model: XRFModel


def parse_chemical_formula(formula: str) -> Mapping[str, int]:
    """Extract the elements and their abundances from a chemical formula."""
    return ChemFormula(formula).element


def xrf_model(
    xray_energy: int | float,
    detector_material: str | None,
    detector_thickness: Quantity,
    energy_min=1.5,
    energy_max=None,
    use_bgr=True,
    elements: Mapping[str, int | float] = {},
) -> XRFModel:
    """create an XRF model

    Returns:
    ---------
     an XRF_Model instance
    """
    model = XRFModel(
        xray_energy=xray_energy,
        energy_min=energy_min,
        energy_max=energy_max,
    )
    if detector_material is not None:
        thickness_cm = detector_thickness.to("cm").magnitude
        model.set_detector(material=detector_material, thickness=thickness_cm)
    model.add_escape()
    model.add_pileup()
    # Add sample elements plus Ar since it's in a lot of detectors
    for element, amplitude in elements.items():
        model.add_element(element, amplitude=amplitude)
    model.add_element("Ar")
    # Add an elastic peak
    model.add_scatter_peak()
    return model


def xrf_datasets(run):
    """Iterate over the datasets in a run that contain fluorescence
    data.

    Yields
    ======
    target
      Details of how the fit should be performed

    """
    for stream_name, stream in run.items():
        config = stream.metadata.get("configuration", {})
        for name, this_config in config.items():
            # Check if this is actually a fluorescence spectrum
            config_data = this_config.get("data", {})
            ev_per_bin = config_data.get(f"{name}-ev_per_bin", None)
            if ev_per_bin is not None:
                yield stream[name]


async def _fit_spectrum(
    spectrum,
    ev_per_bin: int | float,
    acquisition_time: float,
    deadtime_factor: float,
    model: XRFModel,
):
    """Read and fit an individual spectrum from the array of spectra."""
    t0 = time.perf_counter()
    # Larch expects data packaged into `Group` objects
    mca = Group()
    counts = spectrum * deadtime_factor / acquisition_time
    mca.counts = counts
    num_bins = counts.shape[0]
    energy_start = ev_per_bin * 0.5
    energy_stop = ev_per_bin * (num_bins + 0.5)
    mca.energy = np.linspace(energy_start, energy_stop, num=num_bins)
    # Execute the fit, use peakutils until we can get larch background working
    # bg = xrf_background(energy=mca.energy/1000, counts=mca.counts)
    # bg = peakutils.baseline(mca.counts, deg=4)
    baseline = Baseline(x_data=mca.energy)
    bg, bg_params = baseline.imodpoly(counts)
    model.add_background(bg)
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, model.fit_spectrum, mca)
    decomp = output.decompose_spectrum(counts)
    result = FitResult(
        goodness=output.redchi,
        weights=decomp.weights,
        predicted=decomp.total,
        model=model,
    )
    log.debug(f"Fit spectrum in {time.perf_counter()-t0:.2f} seconds")
    return result


async def read_frame(parent: Container, name: str, slice):
    """Read a given frame (slice) from an array."""
    loop = asyncio.get_running_loop()
    node = parent[name]
    results = await loop.run_in_executor(None, node.read, slice)
    return results


async def fit_frame(
    node: ArrayClient,
    frame_num: int,
    energy: int | float,
    ev_per_bin: float | int,
    elements: Mapping[str, int | float],
    detector_material: str,
    detector_thickness: Quantity,
):
    loop = asyncio.get_running_loop()
    array_name = node.path_parts[-1]
    elem_indices = list(range(node.shape[1]))
    # Read livetime correction data along with the frame itself
    async with TaskGroup() as tg:
        clock_freq = 80e6
        coros = [
            read_frame(
                node.parent, f"{array_name}-element{elem}-deadtime_factor", frame_num
            )
            for elem in elem_indices
        ]
        dt_tasks = [tg.create_task(coro) for coro in coros]
        coros = [
            read_frame(
                node.parent, f"{array_name}-element{elem}-clock_ticks", frame_num
            )
            for elem in elem_indices
        ]
        clock_tasks = [tg.create_task(coro) for coro in coros]
        frame_task = tg.create_task(
            read_frame(node.parent, name=array_name, slice=frame_num)
        )
    frame = frame_task.result()
    dt_factors = np.asarray([task.result() for task in dt_tasks])
    ticks = np.asarray([task.result() for task in clock_tasks])
    times = ticks / clock_freq
    # Do the actual fitting here
    async with TaskGroup() as tg:
        models = [
            xrf_model(
                xray_energy=energy / 1000,
                energy_max=40,
                elements=elements,
                detector_material=detector_material,
                detector_thickness=detector_thickness,
            )
            for i in range(frame.shape[0])
        ]
        coros = [
            _fit_spectrum(
                spectrum,
                ev_per_bin,
                deadtime_factor=dt,
                acquisition_time=time,
                model=model,
            )
            for spectrum, dt, time, model in zip(frame, dt_factors, times, models)
        ]
        tasks = [tg.create_task(coro) for coro in coros]
    results = [task.result() for task in tasks]
    return results


async def fit_array(
    node: ArrayClient,
    energies,
    results_node: Container,
    elements: Mapping[str, int | float],
    ev_per_bin: int | float,
    detector_material: str,
    detector_thickness: Quantity,
):
    """Fit an array of spectra in an array."""
    ev_per_bin = 10
    detector_name = node.path_parts[-1]
    async with TaskGroup() as tg:
        tasks = [
            tg.create_task(
                fit_frame(
                    node,
                    event,
                    energy,
                    ev_per_bin,
                    elements=elements,
                    detector_material=detector_material,
                    detector_thickness=detector_thickness,
                )
            )
            for event, energy in zip(range(node.shape[0]), energies)
        ]
    # Results come out nested, so flatten them
    results = [task.result() for task in tasks]
    results = [result for result_set in results for result in result_set]
    # from pprint import pprint
    # print(len(results))
    # pprint(results[0])
    # # pprint([p for p in results[0].params])
    # import matplotlib.pyplot as plt

    # for result, slc in zip(results, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):
    #     plt.plot(result.predicted)
    #     plt.plot(node.read()[slc[0], slc[1]])
    #     # plt.xlim(400, 800)
    #     plt.show()

    # Extract the elemental abundance from the spectrum results
    def merge_values(arr, op):
        arr = np.asarray(arr)
        arr = arr.reshape(node.shape[:2])
        arr = op(arr, axis=1)
        return arr

    peak_names = list(results[0].weights.keys())
    new_signals = {}
    for peak in peak_names:
        weights = [result.weights[peak] for result in results]
        new_signals[peak] = merge_values(weights, op=np.sum)
    new_signals["χ²"] = merge_values(
        [result.goodness for result in results], op=np.mean
    )
    # Write these are a table
    df = pd.DataFrame(new_signals)
    table_name = f"{detector_name}-fit"
    if table_name in await asyncio.to_thread(list, results_node.keys()):
        await asyncio.to_thread(results_node[table_name].write, df)
    else:
        await asyncio.to_thread(results_node.write_table, df, key=table_name)
    return results


def _results_container(run, catalog):
    run_uid = run.metadata["start"]["uid"]
    existing_runs = catalog.search(Eq("run_uid", run_uid))
    if len(existing_runs) == 0:
        run = catalog.create_container(metadata={"run_uid": run_uid})
        return run
    else:
        return existing_runs.values().first()


def detector_metadata(config, name):
    material = config["data"][f"{name}-sensor_material"]
    thickness = config["data"][f"{name}-sensor_thickness"]
    units = config["data_keys"][f"{name}-sensor_thickness"]["units"]
    thickness = ureg(f"{thickness} {units}")
    return material, thickness


@flow()
async def fit_fluorescence(
    run_uid: str,
    raw_profile: str,
    results_profile: str,
    max_workers: int | None = None,
):
    # Load the necessary Tiled catalogs
    raw_catalog = from_profile(raw_profile)
    run = raw_catalog[run_uid]
    results_catalog = from_profile(results_profile)
    return await fit_run_fluorescence(
        run=run, results_catalog=results_catalog, max_workers=max_workers
    )


async def fit_run_fluorescence(
    run: Container, results_catalog: Container, max_workers=None
):
    tasks: list[asyncio.Task] = []
    # We need a baseline energy to use for non-energy-scanning streams
    energy_signal = run.metadata.get("start", {}).get(
        "energy_signal", "monochromator-energy"
    )
    if "baseline" in run.keys() and energy_signal in run["baseline"].keys():
        baseline_energy = np.mean(run["baseline"][energy_signal].read())
    else:
        log.info(
            f"Could not read baseline energy '{energy_signal}' for run '{run.uri}'."
        )
        baseline_energy = None
    # Fit the whole array concurrently
    results_run = _results_container(run, results_catalog)
    elements = parse_chemical_formula(run.metadata["start"].get("sample_formula", ""))
    if len(elements) == 0:
        log.warning(f"Fitting 0 chemical elements for run '{run.uri}'")
    with ThreadExecutor(max_workers=max_workers) as executor:
        async with TaskGroup() as tg:
            loop = asyncio.get_running_loop()
            loop.set_default_executor(executor)
            nodes = xrf_datasets(run)
            tasks = []
            for source_node in nodes:
                stream_name, array_name = source_node.path_parts[-2:]
                stream = source_node.parent
                config = stream.metadata["configuration"][array_name]
                ev_per_bin = config["data"][f"{array_name}-ev_per_bin"]
                try:
                    material, thickness = detector_metadata(config, array_name)
                except KeyError as exc:
                    log.error(
                        f"Missing configuration '{exc.args[0]}' for '{array_name}'"
                    )
                    continue
                baseline_energies = np.full(
                    shape=source_node.shape[:1], fill_value=baseline_energy
                )
                energies = (
                    stream[energy_signal].read()
                    if energy_signal in stream.keys()
                    else baseline_energies
                )
                # Let prefect know what assets we expect to produce
                expected_table_uri = (
                    f"{results_run.uri}/{source_node.path_parts[-1]}-fit"
                )
                node_name = source_node.path_parts[-1]
                coro = materialize(
                    expected_table_uri,
                    asset_deps=[source_node.uri],
                    task_run_name=f"fit-xrf-{node_name}",
                )(fit_array)(
                    source_node,
                    energies,
                    results_node=results_run,
                    elements=elements,
                    ev_per_bin=ev_per_bin,
                    detector_material=material,
                    detector_thickness=thickness,
                )
                tasks.append(tg.create_task(coro))
    results = [task.result() for task in tasks]
    if len(results) == 0:
        log.warning("Fitting produced no results.")
    else:
        log.info("Fit {len(results)} detectors.")
    return results


def main(argv: Sequence[str] | None = None):
    """Main entry-point for exporting an HDF5 file for a given run."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="fit-fluorescence",
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
        fit_fluorescence.serve()
    else:
        asyncio.run(
            fit_fluorescence(
                run_uid=args.run_uid,
                raw_profile=args.raw_profile,
                results_profile=args.results_profile,
                max_workers=args.max_workers,
            )
        )
