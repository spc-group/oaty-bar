import argparse
import asyncio
import logging
import time
from asyncio import TaskGroup
from collections.abc import Sequence

import numpy as np
from larch import Group
from larch.xrf.xrf_model import XRF_Model as XRFModel
from tiled.client import from_profile
from tiled.client.array import ArrayClient
from tiled.client.container import Container
from tiled.queries import Eq
from xraydb import material_mu

log = logging.getLogger("oaty-bar")


def xrf_model(
    xray_energy=None, energy_min=1.5, energy_max=None, use_bgr=False, **kws
) -> XRFModel:
    """create an XRF model

    Returns:
    ---------
     an XRF_Model instance
    """
    model = XRFModel(
        xray_energy=xray_energy,
        use_bgr=use_bgr,
        energy_min=energy_min,
        energy_max=energy_max,
        **kws,
    )
    model.set_detector(material="Ge", thickness=1.0)
    model.add_escape()
    model.add_pileup()
    model.add_element("Cr")
    model.add_element("Ar")
    model.add_scatter_peak()
    return model


def xrf_datasets(run, results_node):
    """Iterate over the datasets in a run that contain fluorescence
    data.

    Also creates result stream nodes for each run.

    Yields
    ======
    node
      The node with the source data.
    result_node
      A node in the results node for the given datasets stream.

    """
    for stream_name, stream in run.items():
        config = stream.metadata.get("configuration", {})
        results_stream = results_node.get(stream_name, None)
        for name, this_config in config.items():
            # Check if this is actually a fluorescence spectrum
            ev_per_bin = this_config.get(f"{name}-ev_per_bin", None)
            if ev_per_bin is not None:
                if results_stream is None:
                    # We don't have a node for stream results yet, so make one
                    results_stream = results_node.create_container(stream_name)
                yield (stream[name], results_stream)


async def _fit_spectrum(spectrum, ev_per_bin: int | float, model: XRFModel):
    """Read and fit an individual spectrum from the array of spectra."""
    t0 = time.perf_counter()
    # Larch expects data packaged into `Group` objects
    mca = Group()
    mca.counts = spectrum
    num_bins = spectrum.shape[0]
    energy_start = ev_per_bin * 0.5
    energy_stop = ev_per_bin * (num_bins + 0.5)
    mca.energy = np.linspace(energy_start, energy_stop, num=num_bins)
    # Execute the fit
    result = model.fit_spectrum(mca)
    decomp = result.decompose_spectrum(spectrum)
    log.debug(f"Fit spectrum in {time.perf_counter()-t0:.2f} seconds")
    return decomp


async def fit_frame(
    node: ArrayClient, frame_num: int, energy: int | float, ev_per_bin: float | int
):
    loop = asyncio.get_running_loop()
    frame = await loop.run_in_executor(None, node.read, frame_num)
    async with TaskGroup() as tg:
        models = [
            xrf_model(xray_energy=energy / 1000, energy_max=40)
            for i in range(frame.shape[0])
        ]
        coros = [
            _fit_spectrum(spectrum, ev_per_bin, model)
            for spectrum, model in zip(frame, models)
        ]
        tasks = [tg.create_task(coro) for coro in coros]
    results = [task.result() for task in tasks]
    return results


async def fit_array(node: ArrayClient, energies, results_stream: Container):
    """Fit an array of spectra in an array."""
    ev_per_bin = 10
    detector_name = node.path_parts[-1]
    async with TaskGroup() as tg:
        tasks = [
            tg.create_task(fit_frame(node, event, energy, ev_per_bin))
            for event, energy in zip(range(node.shape[0]), energies)
        ]
    # Results come out nested, so flatten them
    results = [task.result() for task in tasks]
    results = [result for result_set in results for result in result_set]
    # Extract the elemental abundance from the spectrum results
    # from pprint import pprint
    # print(len(results))
    # pprint(results[0])
    # pprint([p for p in results[0].params])
    import matplotlib.pyplot as plt

    for result, slc in zip(results, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):
        plt.plot(result.total)
        plt.plot(node.read()[slc[0], slc[1]])
        plt.show()
    peak_names = list(results[0].weights.keys())
    new_signals = {}
    for peak in peak_names:
        weights = np.asarray([result.weights[peak] for result in results])
        weights = weights.reshape(node.shape[:2])
        weights = np.sum(weights, axis=1)
        new_signals[peak] = weights
    # Write these are separate arrays, maybe this could be a table in the future?
    for element_name, arr in new_signals.items():
        results_stream.write_array(arr, key=f"{detector_name}-{element_name}")


def _results_container(run, catalog):
    run_uid = run.metadata["start"]["uid"]
    existing_runs = catalog.search(Eq("run_uid", run_uid))
    if len(existing_runs) == 0:
        run = catalog.create_container(metadata={"run_uid": run_uid})
        return run
    else:
        return existing_runs.values().first()


async def fit_fluorescence(run: Container, results_catalog: Container):
    tasks = []
    # xraydb.MATERIALS is not thread-safe, so pre-load the db
    material_mu("Ge", 1000)
    # Fit the whole array concurrently
    energies = run["primary/monochromator-energy"].read()
    results_run = _results_container(run, results_catalog)

    async with TaskGroup() as tg:
        for source_node, results_stream in xrf_datasets(run, results_run):
            tasks.append(
                tg.create_task(
                    fit_array(
                        source_node, energies=energies, results_stream=results_stream
                    )
                )
            )


def main(args: Sequence[str] | None = None):
    """Main entry-point for exporting an HDF5 file for a given run."""
    # Argument handling
    parser = argparse.ArgumentParser(
        prog="fit-fluorescence",
        description="Apply corrections and fit fluorescence spectra to produce elemental contributions",
    )
    parser.add_argument("uid", help="The UID of the bluesky run to process.")
    parser.add_argument(
        "--raw-profile", help="The name of the Tiled profile used for raw runs."
    )
    parser.add_argument(
        "--results-profile",
        help="The name of the Tiled profile used for processed run data.",
    )
    parsed = parser.parse_args(args)
    # Load the necessary Tiled catalogs
    raw_catalog = from_profile(parsed.raw_profile)
    run = raw_catalog[parsed.uid]
    results_catalog = from_profile(parsed.results_profile)
    # Do the actual exporting
    asyncio.run(fit_fluorescence(run=run, results_catalog=results_catalog))
