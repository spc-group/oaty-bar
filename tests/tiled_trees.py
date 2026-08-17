from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.xarray import DatasetAdapter, _DatasetMap
from tiled.client import Context, from_context
from tiled.server.app import build_app
from tiled.structures.core import Spec

data_dir = Path(__file__).parent / "data"

# Tiled data to use for testing
# Some mocked test data

xafs_primary = xr.Dataset(
    {
        "energy": (["dim0"], np.linspace(8300, 8400, num=100)),
        "energy-id-energy-readback": (["dim0"], np.linspace(8.32, 8.42, num=100)),
        "ts_energy": (["dim0"], np.linspace(0, 15, num=100)),
        "ts_energy-id-energy-readback": (["dim0"], np.linspace(0, 15, num=100)),
        "It-count": (["dim0"], np.linspace(4e6, 5e6, num=100)),
        "counter-clock-count": (["dim0"], np.linspace(10e6, 10e6, num=100)),
        "It-net_current": (
            ["dim0"],
            np.abs(np.sin(np.linspace(0, 4 * np.pi, num=100))),
        ),
        "ts_It-net_current": (["dim0"], np.linspace(0, 15, num=100)),
        "I0-net_current": (["dim0"], np.linspace(1, 2, num=100)),
        "I0-count": (["dim0"], np.linspace(1e6, 2e6, num=100)),
        "ts_I0-net_current": (["dim0"], np.linspace(0, 15, num=100)),
        "ge_8element": (["dim0", "dim1", "dim2"], np.ones(shape=(100, 8, 1024))),
        "ge_8element-element0-all_event": (["dim0"], np.ones(shape=(100,))),
    }
)

xafs_baseline = xr.Dataset(
    {
        "aps_current": np.asarray([130.0, 204.1]),
        "aps_fill_number": np.asarray([1, 2]),
        "aps_global_feedback": np.asarray([True, False]),
        "ts_aps_current": np.asarray([10, 25]),
        "ts_aps_fill_number": np.asarray([10, 25]),
        "ts_aps_global_feedback": np.asarray([10, 25]),
    }
)


grid_scan = pd.DataFrame(
    {
        "CdnIPreKb": np.linspace(0, 104, num=105),
        "It_net_counts": np.linspace(0, 104, num=105),
        "aerotech_horiz": np.linspace(0, 104, num=105),
        "aerotech_vert": np.linspace(0, 104, num=105),
    }
)


xafs_tree = MapAdapter(
    {
        "primary": DatasetAdapter(
            _DatasetMap(xafs_primary),
            metadata=yaml.safe_load(open(data_dir / "xafs_scan_primary_metadata.yaml")),
            specs=[Spec("BlueskyEventStream", version="3.0"), Spec("xarray_dataset")],
            # Spec('BlueskyEventStream', version='3.0')
        ),
        "baseline": DatasetAdapter(
            _DatasetMap(xafs_baseline),
            metadata=yaml.safe_load(
                open(data_dir / "xafs_scan_baseline_metadata.yaml")
            ),
            specs=[Spec("BlueskyEventStream", version="3.0"), Spec("xarray_dataset")],
        ),
    },
    metadata=yaml.safe_load(open(data_dir / "xafs_scan_metadata.yaml")),
    specs=[Spec("BlueskyRun", version="3.0")],
)


# Reference 4219b3e8-97ba-434c-b564-95b9d0fc921b
xrf_xafs_tree = MapAdapter(
    {
        "primary": MapAdapter(
            {
                # Add energies for points 1-3: [6911. , 6915.8, 6921.]
                "ge_2element": ArrayAdapter.from_array(
                    np.load(data_dir / "xrf_spectra.npy")
                ),
                "ge_2element-element0-clock_ticks": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=160e6,
                    )
                ),
                "ge_2element-element1-clock_ticks": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=160e6,
                    )
                ),
                "ge_2element-element0-deadtime_factor": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=1.5,
                    )
                ),
                "ge_2element-element1-deadtime_factor": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=1.5,
                    )
                ),
                "monochromator-energy": ArrayAdapter.from_array(
                    [6911.0, 6915.8, 6921.0]
                ),
            },
            metadata={
                "configuration": {
                    "ge_2element": {
                        "data": {
                            "ge_2element-ev_per_bin": 10,
                            "ge_2element-sensor_material": "Ge",
                            "ge_2element-sensor_thickness": 6.0,
                        },
                        "data_keys": {
                            "ge_2element-sensor_thickness": {
                                "units": "mm",
                            }
                        },
                    },
                },
            },
        ),
    },
    metadata={"start": {"uid": "12345", "sample_formula": "Cr3O4"}},
)


xrf_xafs_tree_no_metadata = MapAdapter(
    # Equivalent to xrf_xafs_tree but with minimal metadata to count as an xrf scan
    {
        "primary": MapAdapter(
            {
                # Add energies for points 1-3: [6911. , 6915.8, 6921.]
                "ge_2element": ArrayAdapter.from_array(
                    np.load(data_dir / "xrf_spectra.npy")
                ),
                "ge_2element-element0-clock_ticks": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "ge_2element-element1-clock_ticks": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "ge_2element-element0-deadtime_factor": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "ge_2element-element1-deadtime_factor": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "monochromator-energy": ArrayAdapter.from_array(
                    [6911.0, 6915.8, 6921.0]
                ),
            },
            metadata={
                "configuration": {
                    "ge_2element": {
                        "data": {
                            "ge_2element-ev_per_bin": 10,
                        },
                        "data_keys": {},
                    },
                },
            },
        ),
    },
    metadata={"start": {"uid": "12345"}},
)


xrf_line_tree = MapAdapter(
    {
        "primary": MapAdapter(
            {
                # Add energies for points 1-3: [6911. , 6915.8, 6921.]
                "ge_2element": ArrayAdapter.from_array(
                    np.load(data_dir / "xrf_spectra.npy")
                ),
                "ge_2element-element0-clock_ticks": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "ge_2element-element1-clock_ticks": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "ge_2element-element0-deadtime_factor": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "ge_2element-element1-deadtime_factor": ArrayAdapter.from_array(
                    np.full(
                        shape=[
                            3,
                        ],
                        fill_value=0.5,
                    )
                ),
                "aerotech_horiz": ArrayAdapter.from_array([-100, 0, 100]),
            },
            metadata={
                "configuration": {
                    "ge_2element": {
                        "data": {
                            "ge_2element-ev_per_bin": 10,
                            "ge_2element-sensor_material": "Ge",
                            "ge_2element-sensor_thickness": 6.0,
                        },
                        "data_keys": {
                            "ge_2element-sensor_thickness": {
                                "units": "mm",
                            }
                        },
                    },
                },
            },
        ),
        "baseline": MapAdapter(
            {
                # Add energies for points 1-3: [6911. , 6915.8, 6921.]
                "secondary-mono-energy": ArrayAdapter.from_array([9000, 9002]),
            },
            metadata={},
        ),
    },
    metadata={
        "start": {
            "uid": "12345",
            "sample_formula": "Cr3O4",
            "energy_signal": "secondary-mono-energy",
        }
    },
)


@contextmanager
def build_tree():
    with Context.from_app(build_app(xafs_tree)) as context:
        client = from_context(context)
        yield client
