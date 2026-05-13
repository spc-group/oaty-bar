import time
import warnings

import chemformula
import numpy as np
import pytest
from tiled.adapters.mapping import MapAdapter
from tiled.client import Context, from_context
from tiled.server.app import build_app

from oaty_bar._fit_fluorescence import (
    _fit_spectrum,
    fit_fluorescence,
    parse_chemical_formula,
    ureg,
    xrf_model,
)

from .tiled_trees import (
    data_dir,
    xrf_line_tree,
    xrf_xafs_tree,
    xrf_xafs_tree_no_metadata,
)


@pytest.fixture
def ignore_larch_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Elam tables are unreliable for energies [<>] [18]00 k?eV"
        )
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            message="the matrix subclass is not the recommended way to represent matrices",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"ignoring `maxfev` argument to `Minimizer\(\)`. Use `max_nfev` instead.",
        )
        yield


@pytest.fixture(scope="module")
def xrf_catalog():
    tree = MapAdapter(
        {
            "xafs_run": xrf_xafs_tree,
            "xafs_run_no_metadata": xrf_xafs_tree_no_metadata,
            "line_run": xrf_line_tree,
        }
    )
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
        yield client


@pytest.mark.asyncio
async def test_writes_result_container(
    xrf_catalog, results_catalog, ignore_larch_warnings
):
    run = xrf_catalog["xafs_run"]
    t0 = time.perf_counter()
    await fit_fluorescence(run=run, results_catalog=results_catalog, max_workers=1)
    t1 = time.perf_counter()
    assert len(results_catalog.keys()) == 1
    result = results_catalog.values().first()
    assert result.metadata["run_uid"] == "12345"
    result_table = result["ge_2element-fit"].read()
    assert "Cr" in result_table
    assert "O" in result_table
    assert "Ar" in result_table
    assert "elastic" in result_table
    assert "background" in result_table
    assert "pileup" in result_table
    assert "χ²" in result_table


@pytest.mark.asyncio
async def test_setup_xrf_model(xrf_catalog, results_catalog, ignore_larch_warnings):
    run = xrf_catalog["xafs_run"]
    results = await fit_fluorescence(run=run, results_catalog=results_catalog)
    model = results[0][0].model
    assert model.detector.material == "Ge"
    assert model.detector.thickness == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_missing_metadata(xrf_catalog, results_catalog, ignore_larch_warnings):
    """Make sure the procedure is robust against incomplete metadata."""
    run = xrf_catalog["xafs_run_no_metadata"]
    # Just needs to run, no results expected
    results = await fit_fluorescence(run=run, results_catalog=results_catalog)


@pytest.mark.asyncio
async def test_baseline_energy(xrf_catalog, results_catalog, ignore_larch_warnings):
    """Can the XRF model get the x-ray energy from baseline metadata."""
    run = xrf_catalog["line_run"]
    results = await fit_fluorescence(run=run, results_catalog=results_catalog)
    model = results[0][0].model
    assert model.xray_energy == 9.001


# @pytest.mark.asyncio
# async def test_reuse_results_node(xrf_catalog, results_catalog, ignore_larch_warnings):
#     """Make sure we don't create duplicate nodes in the results catalog."""
#     run = xrf_catalog["xafs_run"]
#     assert len(results_catalog) == 0
#     results = await fit_fluorescence(run=run, results_catalog=results_catalog)
#     assert len(results_catalog) == 1
#     results = await fit_fluorescence(run=run, results_catalog=results_catalog)
#     assert len(results_catalog) == 1


test_datasets = [
    # (data file, UID, xray_energy, χ²)
    ("4219b3e8-97ba-434c-b564-95b9d0fc921b_0-0.npy", "Cr", 6911.0, 0.52),
]


@pytest.mark.parametrize("data_file,formula,xray_energy,target", test_datasets)
@pytest.mark.asyncio
async def test_goodness_of_fits(
    data_file, formula, xray_energy, target, ignore_larch_warnings
):
    """Check that we can reliable fit a variety of spectra."""
    spectrum = np.load(data_dir / data_file)
    elements = chemformula.ChemFormula(formula).element
    model = xrf_model(
        xray_energy=xray_energy,
        elements=elements,
        detector_material="Ge",
        detector_thickness=ureg("6 mm"),
    )
    # Do the fitting
    result = await _fit_spectrum(
        spectrum,
        ev_per_bin=10,
        model=model,
    )
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
