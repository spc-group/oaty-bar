"""Uses scan "1af26a59-6d65-43f1-8ac8-97462f124b3e" as an example of
recorded dark current.

"""

import numpy as np
import pytest
import xarray as xr
import yaml
from prefect import flow
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.xarray import DatasetAdapter, _DatasetMap
from tiled.client import Context, from_context
from tiled.server.app import build_app
from tiled.structures.core import Spec

from oaty_bar._correct_ion_chambers import apply_corrections, correct_run

from .tiled_trees import (
    data_dir,
    xafs_tree,
)

dark_current_tree = MapAdapter(
    {
        "primary": DatasetAdapter(
            _DatasetMap(
                xr.Dataset(
                    {
                        "I0-count": (["dim0"], np.asarray([2387])),
                        "It-count": (["dim0"], np.asarray([9843])),
                        "counter-clock-count": (["dim0"], np.asarray([5e6])),
                    }
                )
            ),
            # metadata={
            #     "data_keys": data_keys,
            #     "configuration": xafs_config,
            # },
            specs=[Spec("BlueskyEventStream", version="3.0"), Spec("xarray_dataset")],
            # Spec('BlueskyEventStream', version='3.0')
        ),
        # "baseline": DatasetAdapter(
        #     _DatasetMap(xafs_baseline),
        #     metadata={
        #         "attrs": xafs_primary.attrs,
        #         "data_keys": baseline_data_keys,
        #     },
        #     specs=[Spec("BlueskyEventStream", version="3.0"), Spec("xarray_dataset")],
        # ),
    },
    metadata=yaml.safe_load(open(data_dir / "dark_current_metadata.yaml")),
    specs=[Spec("BlueskyRun", version="3.0")],
)


@pytest.fixture(scope="module")
def xrf_catalog():
    tree = MapAdapter(
        {
            "xafs_run": xafs_tree,
            "dark_current_run": dark_current_tree,
        }
    )
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
        yield client


@pytest.mark.asyncio
async def test_writes_result_container(xrf_catalog, results_catalog):
    run = xrf_catalog["xafs_run"]
    dark_current_run = xrf_catalog["dark_current_run"]

    @flow()
    async def do_fit():
        await correct_run(
            run=run, dark_current_run=dark_current_run, results_catalog=results_catalog
        )

    await do_fit()
    assert len(results_catalog.keys()) == 1
    result = results_catalog.values().first()
    assert result.metadata["run_uid"] == "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f"
    result_table = result["primary-ion_chambers"].read()
    assert "It-net_count" in result_table
    assert "It-count_rate" in result_table
    assert "It-voltage" in result_table
    assert "It-current" in result_table
    assert "I0-net_count" in result_table
    assert "I0-count_rate" in result_table
    assert "I0-voltage" in result_table
    assert "I0-current" in result_table
    table_metadata = result["primary-ion_chambers"].metadata
    assert table_metadata["hints"] == {
        "counter": {
            "fields": ["I0-net_count", "I0-current", "It-net_count", "It-current"]
        },
    }
    base_data_key = {
        "dtype": "array",
        "dtype_numpy": "<f8",
        "shape": [100],
        "units": "",
    }
    assert table_metadata["data_keys"] == {
        "I0-net_count": {**base_data_key, "dtype_numpy": "<i4"},
        "I0-count_rate": {**base_data_key, "units": "/s"},
        "I0-voltage": {**base_data_key, "units": "V"},
        "I0-current": {**base_data_key, "units": "A"},
        "It-net_count": {**base_data_key, "dtype_numpy": "<i4"},
        "It-count_rate": {**base_data_key, "units": "/s"},
        "It-voltage": {**base_data_key, "units": "V"},
        "It-current": {**base_data_key, "units": "A"},
    }


def test_correction_math():
    """Check the math used for applying corrections.

    With a 10 MHz clock, then a clock tick of 20_000_000 is 2
    seconds.

    Let's say we have a V2F set to 50 MHz/10V or 5 MHz / V.  Then 2.5V
    output would be 2s * 12.5 MHz, and a dark "current" of 0.1 V would be 500 kHz.

    """
    df, _ = apply_corrections(
        counts=np.asarray([25_000_000]),
        clock_ticks=np.asarray([20e6]),
        dark_count_rate=500_000,
        preamp_gain=12,
        name="It",
        clock_frequency=10e6,
        hertz_per_volt=5_000_000,
    )
    assert df["It-net_count"][0] == 24_000_000
    assert df["It-count_rate"][0] == 12_000_000
    assert df["It-voltage"][0] == 2.4
    assert df["It-current"][0] == pytest.approx(0.2)


def test_correction_data_keys():
    _, data_keys = apply_corrections(
        counts=np.asarray([25_000_000]),
        clock_ticks=np.asarray([20e6]),
        dark_count_rate=500_000,
        preamp_gain=12,
        name="It",
        clock_frequency=10e6,
        hertz_per_volt=5_000_000,
    )
    assert data_keys["It-net_count"] == {
        "dtype": "array",
        "dtype_numpy": "<i4",
        "units": "",
        "shape": (1,),
    }
    assert data_keys["It-count_rate"] == {
        "dtype": "array",
        "dtype_numpy": "<f8",
        "units": "/s",
        "shape": (1,),
    }
    assert data_keys["It-voltage"] == {
        "dtype": "array",
        "dtype_numpy": "<f8",
        "units": "V",
        "shape": (1,),
    }
    assert data_keys["It-current"] == {
        "dtype": "array",
        "dtype_numpy": "<f8",
        "units": "A",
        "shape": (1,),
    }
