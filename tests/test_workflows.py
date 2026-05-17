from pathlib import Path

from oaty_bar.workflows import load_workflow

workflow_dir = Path(__file__).parent.parent / "workflows"


def test_simple_workflow():
    "Check that the given workflow is equivalent to the JSON."
    workflow = load_workflow("simple", username="s255idzuser")
    assert workflow.name == "simple"
    # assert workflow.owner == "s255idzuser"
    # assert workflow.userAccount == "s255idzuser"
    assert (
        workflow.description
        == "Processing for a Bluesky scan that has no significant data processing needs"
    )
    assert workflow.version == 1
    assert workflow.model_dump(by_alias=True)["stages"] == {
        "010-START": {
            "command": "/bin/date +%Y%m%d%H%M%S",
            "outputVariableRegexList": ["(?P<timeStamp>.*)"],
        },
        "020-UPDATE": {
            "command": "/usr/bin/git -C ~s25idcuser/src/oaty-bar pull",
        },
        "030-FIT_FLUORESCENCE": {
            "command": "/APSshare/bin/pixi run --manifest-path ~s25idcuser/src/oaty-bar fit-fluorescence $run_uid --raw-profile oaty-bar --results-profile oaty-bar-results"
        },
        "040-EXPORT": {
            "command": "/APSshare/bin/pixi run --manifest-path ~s25idcuser/src/oaty-bar export-hdf $run_uid $target_folder --raw-profile oaty-bar --results-profile oaty-bar-results"
        },
    }


def test_add_username():
    "Check that loading a workflow file can replace the owner and user account."
    workflow = load_workflow("simple", username="s255idzuser")
    assert workflow.owner == "s255idzuser"
    assert workflow.user_account == "s255idzuser"
