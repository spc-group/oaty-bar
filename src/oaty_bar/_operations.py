from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from tiled.client.container import Container

from ._data_sources import datapipe


@datapipe
async def extract_rois(
    events: AsyncIterator[xr.Dataset],
    run: Container,
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
    run:
      The Tiled run container for this scan.
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
