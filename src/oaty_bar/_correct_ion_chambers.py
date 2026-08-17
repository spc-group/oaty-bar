"""Tools for correcting ion chamber data for

- Dark current
- Pre-amp gain
- Voltage-to-frequency conversion

"""

import asyncio

import numpy as np
import pandas as pd
import xarray as xr
from pint import UnitRegistry
from prefect import flow
from prefect.logging import get_run_logger
from tiled.client import from_profile
from tiled.client.container import Container

from ._catalog import results_container

ureg = UnitRegistry()


def apply_corrections(
    counts: np.ndarray,
    clock_ticks: np.ndarray,
    dark_count_rate: int,
    preamp_gain: float,
    name: str,
    clock_frequency: float | int,
    hertz_per_volt: float | int,
) -> tuple[pd.DataFrame, dict]:
    """Apply data corrections to a single ion chamber count signal."""
    time = clock_ticks / clock_frequency
    dark_count = dark_count_rate * time
    net_count = counts - dark_count
    count_rate = net_count / time
    voltage = count_rate / hertz_per_volt
    current = voltage / preamp_gain
    df = pd.DataFrame(
        {
            f"{name}-net_count": net_count,
            f"{name}-count_rate": count_rate,
            f"{name}-voltage": voltage,
            f"{name}-current": current,
        }
    )
    data_keys = {
        f"{name}-net_count": {
            "dtype": "array",
            "dtype_numpy": "<i4",
            "units": "",
            "shape": counts.shape,
        },
        f"{name}-count_rate": {
            "dtype": "array",
            "dtype_numpy": "<f8",
            "units": "/s",
            "shape": counts.shape,
        },
        f"{name}-voltage": {
            "dtype": "array",
            "dtype_numpy": "<f8",
            "units": "V",
            "shape": counts.shape,
        },
        f"{name}-current": {
            "dtype": "array",
            "dtype_numpy": "<f8",
            "units": "A",
            "shape": counts.shape,
        },
    }
    return df, data_keys


async def correct_stream(
    stream: Container,
    dark_data: xr.Dataset,
    baseline: xr.Dataset,
    run_uid: str,
    results_catalog: Container,
) -> Container | None:
    """Apply data corrections to all ion chambers in a stream and save
    results.

    This will create a new container in *results_catalog* if one does
    not exist. The results container will receive a new table with the
    calculated results.

    Returns the new results table if one was created.

    """
    config = stream.metadata.get("configuration", {})
    ic_signals = [
        (name, cfg["data"].get(f"{name}-dark_current_signals", []))
        for name, cfg in config.items()
    ]
    ion_chamber_signals = [(dev, sig) for dev, sigs in ic_signals for sig in sigs]
    signals = [f"{name}-count" for _, name in ion_chamber_signals]
    counters = set([dev for dev, name in ion_chamber_signals])
    clock_signals = [f"{dev_name}-clock-count" for dev_name in counters]
    raw_data = await asyncio.to_thread(
        stream.read, signals + clock_signals + ["seq_num"]
    )
    # Apply the data corrections now that we have data
    dataframes = []
    hints: dict[str, dict] = {}
    data_keys = {}
    for device_name, basename in ion_chamber_signals:
        dark_counts = dark_data[f"{basename}-count"].values[0]
        dark_time = dark_data[f"{device_name}-clock-count"].values[0]
        preamp_config = baseline.metadata["configuration"][f"{basename}-preamp"]
        device_config = stream.metadata["configuration"][device_name]
        df, _data_keys = apply_corrections(
            counts=raw_data[f"{basename}-count"].values,
            dark_count_rate=dark_counts / dark_time,
            clock_ticks=raw_data[f"{device_name}-clock-count"].values,
            preamp_gain=preamp_config["data"][f"{basename}-preamp-gain"],
            clock_frequency=device_config["data"][
                f"{device_name}-scaler-clock_frequency"
            ],
            hertz_per_volt=device_config["data"][f"{basename}-hertz_per_volt"],
            name=basename,
        )
        dataframes.append(df)
        data_keys.update(_data_keys)
        hint_fields = hints.setdefault(device_name, {"fields": []})["fields"]
        hint_fields.extend([f"{basename}-net_count", f"{basename}-current"])
    if len(dataframes) == 0:
        return None
    # Save the results in one Tiled table
    full_df = pd.DataFrame().join(dataframes, how="outer")
    results_node = await results_container(run_uid=run_uid, catalog=results_catalog)
    table_name = f"{stream.path_parts[-1]}-ion_chambers"
    md = {
        #     "data_keys": {
        #         key: {"dtype": "number", "dtype_numpy": "<f8", "shape": []}
        #         for key in df.columns
        #     },
        "data_keys": data_keys,
        "hints": hints,
    }
    if table_name in await asyncio.to_thread(list, results_node.keys()):
        table = results_node[table_name]
        await asyncio.gather(
            asyncio.to_thread(table.write, full_df),
            asyncio.to_thread(table.update_metadata, md),
        )
    else:
        table = await asyncio.to_thread(
            results_node.write_table, full_df, key=table_name, metadata=md
        )
    return table


async def correct_run(
    run: Container, dark_current_run: Container, results_catalog: Container
):
    dark_data, streams, baseline = await asyncio.gather(
        asyncio.to_thread(dark_current_run["primary"].read),
        asyncio.to_thread(list, run.values()),
        asyncio.to_thread(run.get, "baseline"),
    )
    run_uid = run.metadata["start"]["uid"]
    await asyncio.gather(
        *[
            correct_stream(
                stream=stream,
                dark_data=dark_data,
                baseline=baseline,
                run_uid=run_uid,
                results_catalog=results_catalog,
            )
            for stream in streams
            if stream != "baseline"
        ]
    )


@flow()
async def correct_ion_chambers(
    run_uid: str,
    raw_profile: str,
    results_profile: str,
):
    log = get_run_logger()
    # Load the necessary Tiled catalogs
    raw_catalog, results_catalog = await asyncio.gather(
        asyncio.to_thread(from_profile, results_profile),
        asyncio.to_thread(from_profile, raw_profile),
    )
    run = raw_catalog[run_uid]
    dark_current_uid = run.metadata.get("start", {}).get("dark_current_uid")
    if dark_current_uid is None:
        log.warning(
            f"Could not find dark_current_uid in metadata for {'/'.join(run.path_parts)}"
        )
        return
    dark_current_run = raw_catalog[dark_current_uid]
    return await correct_run(
        run=run, results_catalog=results_catalog, dark_current_run=dark_current_run
    )
