import pytest
from tiled.adapters.mapping import MapAdapter
from tiled.client import Context, from_context
from tiled.server.app import build_app, build_app_from_config

from oaty_bar._fit_fluorescence import fit_fluorescence

from .tiled_trees import xrf_tree


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
    # plt.plot(np.linspace(5, 40965, num=4096), run['primary']['ge_2element'].read()[0,0])
    # plt.show()
    assert len(results_catalog.keys()) == 1
    result = results_catalog.values().first()
    assert result.metadata["run_uid"] == "12345"
    assert "ge_2element-Cr" in result["primary"].keys()
