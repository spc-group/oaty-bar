import chemformula
import numpy as np
import pytest
from tiled.adapters.mapping import MapAdapter
from tiled.client import Context, from_context
from tiled.server.app import build_app, build_app_from_config

from oaty_bar._fit_fluorescence import (
    _fit_spectrum,
    fit_fluorescence,
    parse_chemical_formula,
    xrf_model,
)

from .tiled_trees import data_dir, xrf_tree


@pytest.fixture(scope="module")
def xrf_catalog():
    tree = MapAdapter({"xrf_run": xrf_tree})
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
        yield client


@pytest.fixture()
def results_catalog(tmp_path):
    "Test 'mounting' sub-trees of a catalog."
    uri = f"sqlite:///{tmp_path}/tiled.sqlite"
    one_tree_config = {
        "trees": [
            {
                "path": "/",
                "tree": "catalog",
                "args": {
                    "uri": uri,
                    "init_if_not_exists": True,
                    "writable_storage": [tmp_path / "data"],
                },
            },
        ]
    }
    with Context.from_app(build_app_from_config(one_tree_config)) as context:
        client = from_context(context)
        yield client


@pytest.mark.asyncio
async def test_writes_result_container(xrf_catalog, results_catalog):
    run = xrf_catalog["xrf_run"]
    await fit_fluorescence(run=run, results_catalog=results_catalog)
    assert len(results_catalog.keys()) == 1
    result = results_catalog.values().first()
    assert result.metadata["run_uid"] == "12345"
    result_keys = list(result["primary"].keys())
    assert "ge_2element-Cr" in result_keys
    assert "ge_2element-O" in result_keys
    assert "ge_2element-Ar" in result_keys
    assert "ge_2element-elastic" in result_keys
    assert "ge_2element-background" in result_keys
    assert "ge_2element-pileup" in result_keys
    assert "ge_2element-χ²" in result_keys


test_datasets = [
    # (data file, UID, xray_energy, χ²)
    ("4219b3e8-97ba-434c-b564-95b9d0fc921b_0-0.npy", "Cr", 6911.0, 0.52),
]


@pytest.mark.parametrize("data_file,formula,xray_energy,target", test_datasets)
@pytest.mark.asyncio
async def test_goodness_of_fits(data_file, formula, xray_energy, target):
    """Check that we can reliable fit a variety of spectra."""
    spectrum = np.load(data_dir / data_file)
    elements = chemformula.ChemFormula(formula).element
    model = xrf_model(xray_energy=xray_energy, elements=elements)
    # Do the fitting
    result = await _fit_spectrum(spectrum, ev_per_bin=10, model=model)
    # Check the fitting accuracy
    assert result.goodness == pytest.approx(target, abs=0.01)


formulae = [
    ("NaCl", {"Na": 1, "Cl": 1}),
    ("Ni(OH)2", {"Ni": 1, "O": 2, "H": 2}),
    # This one should actually be 0.33, but chemformula gets this wrong
    ("Ni0.33Mn0.33Co0.33O2", {"Ni": 33, "Mn": 33, "Co": 33, "O": 2}),
]


@pytest.mark.parametrize("formula,parsed", formulae)
def test_parse_chemical_formula(formula, parsed):
    assert parse_chemical_formula(formula) == parsed
