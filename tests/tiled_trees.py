from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.xarray import DatasetAdapter, _DatasetMap
from tiled.client import Context, from_context
from tiled.server.app import build_app
from tiled.structures.core import Spec

data_dir = Path(__file__).parent / "data"

# Tiled data to use for testing
# Some mocked test data

xafs_run_metadata = {
    "start": {
        "detectors": ["I0"],
        "hints": {"dimensions": [[["pitch2"], "primary"]]},
        "motors": ["pitch2"],
        "facility_id": "Advanced Photon Source",
        "beamline_id": "255-ID-Z",
        "edge": "Ni-K",
        "d_spacing": 3.131562,
        "num_intervals": 19,
        "num_points": 20,
        "plan_args": {
            "args": [
                "EpicsMotor(prefix='25idDCM:AS:m6', "
                "name='pitch2', settle_time=0.0, "
                "timeout=None, read_attrs=['user_readback', "
                "'user_setpoint'], "
                "configuration_attrs=['user_offset', "
                "'user_offset_dir', 'velocity', "
                "'acceleration', 'motor_egu'])",
                -100,
                100,
            ],
            "detectors": [
                "IonChamber(prefix='25idcVME:3820:scaler1', "
                "name='I0', read_attrs=['raw_counts'], "
                "configuration_attrs=[])"
            ],
            "num": 20,
            "per_step": "None",
        },
        "plan_name": "xafs_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_args": {
            "args": [
                "EpicsMotor(prefix='25idDCM:AS:m6', "
                "name='pitch2', settle_time=0.0, "
                "timeout=None, "
                "read_attrs=['user_readback', "
                "'user_setpoint'], "
                "configuration_attrs=['user_offset', "
                "'user_offset_dir', 'velocity', "
                "'acceleration', 'motor_egu'])",
                -100,
                100,
            ],
            "num": 20,
        },
        "plan_pattern_module": "bluesky.plan_patterns",
        "plan_type": "generator",
        "purpose": "alignment",
        "sample_name": "NMC-811",
        "scan_name": "Pristine",
        "scan_id": 1,
        "time": 1665065697.3635247,
        "uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
        "versions": {"bluesky": "1.9.0", "ophyd": "1.7.0"},
    },
    "stop": {
        "exit_status": "success",
        "num_events": {"primary": 20},
        "reason": "",
        "run_start": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
        "time": 1665065735.714015,
        "uid": "c1eac86f-d568-41a1-b601-a0e2fd6ed55e",
    },
    "summary": {
        "datetime": "2022-10-06 09:14:57.363525",
        "duration": 38.35049033164978,
        "plan_name": "xafs_scan",
        "scan_id": 1,
        "stream_names": ["primary"],
        "timestamp": 1665065697.3635247,
        "uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    },
}


xafs_primary = xr.Dataset(
    {
        "energy": (["dim0"], np.linspace(8300, 8400, num=100)),
        "energy-id-energy-readback": (["dim0"], np.linspace(8.32, 8.42, num=100)),
        "ts_energy": (["dim0"], np.linspace(0, 15, num=100)),
        "ts_energy-id-energy-readback": (["dim0"], np.linspace(0, 15, num=100)),
        "It-net_current": (
            ["dim0"],
            np.abs(np.sin(np.linspace(0, 4 * np.pi, num=100))),
        ),
        "ts_It-net_current": (["dim0"], np.linspace(0, 15, num=100)),
        "I0-net_current": (["dim0"], np.linspace(1, 2, num=100)),
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


xafs_config = {
    "monochromator": {
        "data": {
            "monochromator-d_spacing": 3.13,
        },
        "data_keys": {
            "monochromator-d_spacing": {
                "dtype": "number",
                "dtype_numpy": "<f8",
                "shape": [],
                "source": "ca://255idbNP:d_spacing",
            },
        },
    }
}


grid_scan = pd.DataFrame(
    {
        "CdnIPreKb": np.linspace(0, 104, num=105),
        "It_net_counts": np.linspace(0, 104, num=105),
        "aerotech_horiz": np.linspace(0, 104, num=105),
        "aerotech_vert": np.linspace(0, 104, num=105),
    }
)

data_keys = {
    "energy": {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "limits": {
            "control": {"high": 0.0, "low": 0.0},
            "display": {"high": 0.0, "low": 0.0},
        },
        "object_name": "energy",
        "precision": 3,
        "shape": [],
        "source": "ca://25idcVME:3820:scaler1.T",
        "units": "eV",
    },
    "energy-id-energy-readback": {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "limits": {
            "control": {"high": 0.0, "low": 0.0},
            "display": {"high": 0.0, "low": 0.0},
        },
        "object_name": "energy",
        "precision": 3,
        "shape": [],
        "source": "ca://...",
        "units": "keV",
    },
    # "I0-mcs-scaler-channels-0-net_count": {
    #     "dtype": "number",
    #     "dtype_numpy": "<f8",
    #     "limits": {
    #         "control": {"high": 0.0, "low": 0.0},
    #         "display": {"high": 0.0, "low": 0.0},
    #     },
    #     "object_name": "I0",
    #     "precision": 0,
    #     "shape": [],
    #     "source": "ca://25idcVME:3820:scaler1_netA.A",
    #     "units": "",
    # },
    # "I0-mcs-scaler-channels-3-net_count": {
    #     "dtype": "number",
    #     "dtype_numpy": "<f8",
    #     "limits": {
    #         "control": {"high": 0.0, "low": 0.0},
    #         "display": {"high": 0.0, "low": 0.0},
    #     },
    #     "object_name": "I0",
    #     "precision": 0,
    #     "shape": [],
    #     "source": "ca://25idcVME:3820:scaler1_netA.D",
    #     "units": "",
    # },
    # "I0-mcs-scaler-elapsed_time": {
    #     "dtype": "number",
    #     "dtype_numpy": "<f8",
    #     "limits": {
    #         "control": {"high": 0.0, "low": 0.0},
    #         "display": {"high": 0.0, "low": 0.0},
    #     },
    #     "object_name": "I0",
    #     "precision": 3,
    #     "shape": [],
    #     "source": "ca://25idcVME:3820:scaler1.T",
    #     "units": "",
    # },
    "I0-net_current": {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "object_name": "I0",
        "shape": [],
        "source": (
            "soft://I0-net_current(gain,count,clock_count,clock_frequency,counts_per_volt_second)"
        ),
        "units": "A",
    },
    "It-net_current": {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "object_name": "It",
        "shape": [],
        "source": (
            "soft://It-net_current(gain,count,clock_count,clock_frequency,counts_per_volt_second)"
        ),
        "units": "A",
    },
    "ge_8element": {
        "dtype": "array",
        "dtype_numpy": "<u4",
        "external": "STREAM:",
        "object_name": "ge_8element",
        "shape": [1, 8, 1024],
        "source": "ca://XSP_Ge_8elem:HDF1:FullFileName_RBV",
    },
    "ge_8element-element0-all_event": {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "external": "STREAM:",
        "object_name": "ge_8element",
        "shape": [],
        "source": "ca://XSP_Ge_8elem:HDF1:FullFileName_RBV",
    },
}


baseline_data_keys = {
    "aps_current": {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "limits": {
            "control": {"high": 0.0, "low": 0.0},
            "display": {"high": 0.0, "low": 0.0},
        },
        "object_name": "aps",
        "precision": 3,
        "shape": [],
        "source": "ca://...",
        "units": "mA",
    },
    "aps_fill_number": {
        "dtype": "number",
        "dtype_numpy": "<u4",
        "limits": {
            "control": {"high": 0.0, "low": 0.0},
            "display": {"high": 0.0, "low": 0.0},
        },
        "object_name": "aps",
        "shape": [],
        "source": "ca://...",
    },
    "aps_global_feedback": {
        "dtype": "bool",
        "dtype_numpy": "|u1",
        "limits": {
            "control": {"high": 0.0, "low": 0.0},
            "display": {"high": 0.0, "low": 0.0},
        },
        "object_name": "aps",
        "shape": [],
        "source": "ca://...",
    },
}


hints = {
    "energy": {"fields": ["energy", "energy-id-energy-readback"]},
    "It": {"fields": ["It-net_current"]},
    "ge_8element": {"fields": ["ge_8element"]},
    "no_device": {},  # Make sure we test a device with no hints
}


xafs_tree = MapAdapter(
    {
        "primary": DatasetAdapter(
            _DatasetMap(xafs_primary),
            metadata={
                "attrs": xafs_primary.attrs,
                "hints": hints,
                "data_keys": data_keys,
                "configuration": xafs_config,
            },
            specs=[Spec("BlueskyEventStream", version="3.0"), Spec("xarray_dataset")],
            # Spec('BlueskyEventStream', version='3.0')
        ),
        "baseline": DatasetAdapter(
            _DatasetMap(xafs_baseline),
            metadata={
                "attrs": xafs_primary.attrs,
                "data_keys": baseline_data_keys,
            },
            specs=[Spec("BlueskyEventStream", version="3.0"), Spec("xarray_dataset")],
        ),
    },
    metadata=xafs_run_metadata,
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
