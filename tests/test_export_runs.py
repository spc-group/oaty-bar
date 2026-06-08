from contextlib import contextmanager

from tiled.adapters.mapping import MapAdapter
from tiled.client import Context, from_context
from tiled.server.app import build_app

from oaty_bar._export_runs import main

from .tiled_trees import xafs_tree


@contextmanager
def build_tree():
    tree = MapAdapter({"7d1daf1d-60c7-4aa7-a668-d1cd97e5335f": xafs_tree})
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
        yield client


def test_export_run_by_uid(tmp_path, xafs_run, mocker, prefect_server):
    with build_tree() as client:
        from_profile = mocker.MagicMock(return_value=client)
        mocker.patch("oaty_bar._export_runs.from_profile", new=from_profile)

        main(
            [
                "--uid",
                "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
                "--target-dir",
                str(tmp_path),
                "--raw-profile",
                "raw_catalog",
                # "--results-profile",
                # "proc_catalog",
            ]
        )
    # Check that the file was created
    target_file = tmp_path / "202210060914-NMC-811-Pristine-xafs_scan-7d1daf1d.hdf"
    [print(x) for x in tmp_path.iterdir()]
    assert target_file.exists()
