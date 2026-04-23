from contextlib import contextmanager

import numpy as np
import pandas as pd
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.table import TableAdapter
from tiled.client import Context, from_context
from tiled.server.app import build_app

# Tiled data to use for testing
# Some mocked test data

xafs_run_metadata = {
    "start": {
        "detectors": ["I0"],
        "hints": {"dimensions": [[["pitch2"], "primary"]]},
        "motors": ["pitch2"],
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
        "plan_name": "rel_scan",
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
        "plan_name": "rel_scan",
        "scan_id": 1,
        "stream_names": ["primary"],
        "timestamp": 1665065697.3635247,
        "uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    },
}


xafs_events = pd.DataFrame(
    {
        "energy": np.linspace(8300, 8400, num=100),
        "energy-id-energy-readback": np.linspace(8.32, 8.42, num=100),
        "ts_energy": np.linspace(0, 15, num=100),
        "ts_energy-id-energy-readback": np.linspace(0, 15, num=100),
        "It-net_current": np.abs(np.sin(np.linspace(0, 4 * np.pi, num=100))),
        "ts_It-net_current": np.linspace(0, 15, num=100),
        "I0-net_current": np.linspace(1, 2, num=100),
        "ts_I0-net_current": np.linspace(0, 15, num=100),
    }
)

xafs_baseline = {
    "aps_current": ArrayAdapter.from_array(np.asarray([130.0, 204.1])),
    "aps_fill_number": ArrayAdapter.from_array(np.asarray([1, 2])),
    "aps_global_feedback": ArrayAdapter.from_array(np.asarray([True, False])),
    "ts_aps_current": ArrayAdapter.from_array(np.asarray([10, 25])),
    "ts_aps_fill_number": ArrayAdapter.from_array(np.asarray([10, 25])),
    "ts_aps_global_feedback": ArrayAdapter.from_array(np.asarray([10, 25])),
}


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
        "shape": [8, 4096],
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
        "streams": MapAdapter(
            {
                "primary": MapAdapter(
                    {
                        "internal": TableAdapter.from_pandas(xafs_events),
                        "ge_8element": ArrayAdapter.from_array(
                            np.ones(shape=(100, 8, 1024))
                        ),
                        "ge_8element-element0-all_event": ArrayAdapter.from_array(
                            np.ones(shape=(100,))
                        ),
                    },
                    metadata={
                        "hints": hints,
                        "data_keys": data_keys,
                        "configuration": xafs_config,
                    },
                ),
                "baseline": MapAdapter(
                    xafs_baseline,
                    metadata={
                        "data_keys": baseline_data_keys,
                    },
                ),
            }
        ),
    },
    metadata=xafs_run_metadata,
)


@contextmanager
def build_tree(writable_storage):
    # tree = in_memory(writable_storage=writable_storage)
    with Context.from_app(build_app(xafs_tree)) as context:
        client = from_context(context)
        yield client
        # Write sample data
        # primary_metadata = {
        #     "hints": hints,
        #     "data_keys": data_keys,
        #     "configuration": xafs_config,
        # }
        # primary = await client.create_container("primary", metadata=primary_metadata)
        # internal = primary.write_table(xafs_events, key="internal")
        # # Fluorescence data
        # primary.write_array(
        #     np.full(shape=(100, 8, 4096), fill_value=2), key="ge_8element"
        # )
        # primary.write_array(np.ones(shape=(100,)), key="ge_8element-element0-all_event")
        # baseline = client.create_container(
        #     "baseline",
        #     metadata={
        #         "hints": {"aps_current": {"fields": ["aps_current"]}},
        #         "data_keys": baseline_data_keys,
        #     },
        # )
        # internal = baseline.write_table(xafs_baseline, key="internal")
        # yield tree
