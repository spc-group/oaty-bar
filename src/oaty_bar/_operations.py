from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import xarray as xr
import xraydb
from chemformula import ChemFormula
from tiled.client.container import Container

from ._data_sources import datapipe


@datapipe
async def extract_rois(
    events: AsyncIterator[xr.Dataset],
    source: str,
    rois: Mapping[str, Sequence[list[int] | slice]],
) -> AsyncGenerator[xr.Dataset, Any]:
    """Calculate stats for regions-of-interest from a source array.

    The keys of *rois* will be used to construct names for the new ROI
    datasets. Values of *rois* should be indexes to apply to the
    source array.

    For example…

    ```python
    results = extract_rois(…, rois={
      "Ni_Ka": [[0, 2, 3], slice(5, 20)],
      "total": [slice(None), slice(None)],
    }, source="vortex_me4")
    ```

    …will produce results such that…

    ```python
    result['vortex_me4-Ni_Ka-sum'][0] == np.sum(event['vortex'][0, [0, 2, 3], 5:20])
    result['vortex_me4-Ni_Ka-sum'][0] == np.sum(event['vortex'][0])
    ```

    Parameters
    ==========
    events:
      Async-iterable of datasets containing data from events (e.g. the
      scanning axis in a Bluesky scan).
    source:
      The name of the data variable in events from which to calculate
      the ROIs.
    rois:
      Definitions of the ROI's to compute.

    """
    result = None
    async for event in events:
        arr = event[source]
        new_ds = event.copy()
        for roi_name, roi in rois.items():
            sub_arr = arr[:, *roi]
            axis = tuple(range(1, arr.ndim))
            result = np.sum(sub_arr, axis=axis)
            roi_key = f"{source}-{roi_name}-sum"
            new_ds[roi_key] = ("seq_num", result.data)
        yield new_ds


@datapipe
async def extract_fluorescence_rois(
    events: AsyncIterator[xr.Dataset],
    minimum_intensity: float = 0.1,
) -> AsyncGenerator[xr.Dataset, Any]:
    """Calculate stats for regions-of-interest from fluorescence arrays.

    The operation requires that each array in *events* has a "source"
    attribute that corresponds to a
    `bluesky_tiled_plugins.BlueskyEventStream` tiled client.

    Fluorescence arrays are determined based on metadata in the
    *source* client. For each *signal* listed in
    `source.metadata['configuration']`, if the configuration has
    "<signal_name>-ev_per_bin" set, then the signal is interpreted as
    a fluorescence array.

    ROIs are derived from the "chemical_formula" metadat key in the
    parent start document:
    `source.parent.metadata['start']['chemical_formula']`. If not
    found, then only basic stats for the whole array will be
    calculated.

    Parameters
    ==========
    events:
      Async-iterable of datasets containing data from events (e.g. the
      scanning axis in a Bluesky scan).
    minimum_intensity
      X-ray emission lines below this relative intensity will not be
      measured.

    """
    result = None
    async for ds in events:
        stream = ds.attrs["source"]
        new_ds = ds.copy()
        # Decide which ROIs to use
        formula = stream.parent.metadata["start"]["chemical_formula"]
        elements = ChemFormula(formula).element.keys()
        lines = {elem: xraydb.xray_lines(elem) for elem in elements}
        min_intensity = 0.1
        lines = {
            f"{elem}_{line.initial_level}{line.final_level}": line.energy
            for elem, lines in lines.items()
            for line in lines.values()
            if line.intensity > minimum_intensity
        }
        # Decide which signals are fluorescence data
        xrf_configs = {
            sig: cfg["data"]
            for sig, cfg in stream.metadata["configuration"].items()
            if f"{sig}-ev_per_bin" in cfg["data"].keys()
        }
        # Apply the ROIs
        for (sig, cfg), (line, line_energy) in product(
            xrf_configs.items(), lines.items()
        ):
            ev_per_bin = cfg[f"{sig}-ev_per_bin"]
            nbins = ds[sig].shape[-1]
            energies = np.linspace(
                ev_per_bin / 2, ev_per_bin * (nbins - 0.5), num=nbins
            )
            min_energy = ev_per_bin * nbins * 0.05
            if line_energy < min_energy:
                # Ignore peaks in the bottom 5% of the spectrum
                continue
            roi = np.logical_and(
                line_energy - 100 < energies, energies < line_energy + 100
            )
            roi_sum = np.sum(ds[sig][:, :, :, roi], axis=(1, 2, 3))
            new_ds[f"{sig}-{line}_guess-sum"] = roi_sum
        # Apply overall stats
        for sig, cfg in xrf_configs.items():
            full_sum = np.sum(ds[sig], axis=(1, 2, 3))
            new_ds[f"{sig}-full-sum"] = full_sum
        yield new_ds


@datapipe
async def apply_correction(
    events: AsyncIterator[xr.Dataset],
    run: Container,
    source_signal: str,
    correction_signal: str,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.divide,
    scale_factor: float | int | None = None,
) -> AsyncGenerator[xr.Dataset, Any]:
    """Apply some operation to a target signal.

    *source_signal*, *correction_signal*, *operation*, and
     *scale_factor* will be used to apply the correction…

    ```python
    operation(source, correction * scale_factor)
    ```

    *scale_factor* can be used to convert e.g. a clock signal into
    seconds.

    Parameters
    ==========
    events:
      Async-iterable of datasets containing data from events (e.g. the
      scanning axis in a Bluesky scan).
    run:
      The Tiled run container for this scan.
    source_signal:
      The name of the data variable in events on which to apply the
      correction.
    correction_signal:
      The name of the data variable with the correction signal to be
      applied.

    """
    result = None
    async for event in events:
        new_ds = event.copy()
        source = event[source_signal]
        correction = event[correction_signal]
        if scale_factor is not None:
            correction = np.multiply(correction, scale_factor)
        new_arr = operation(
            source,
            correction,
        )
        new_ds[source_signal] = new_arr
        yield new_ds
