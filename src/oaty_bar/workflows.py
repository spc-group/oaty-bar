import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic.experimental.missing_sentinel import MISSING

workflow_dir = Path(__file__).parent / "workflows"


class Stage(BaseModel):
    command: str
    output_variables: Sequence[str] = Field(
        alias="outputVariableRegexList", default=MISSING
    )


class Workflow(BaseModel):
    """A data management workflow definition."""

    name: str
    description: str
    version: int
    stages: Mapping[str, Stage]
    owner: str
    user_account: str = Field(alias="userAccount")


def load_workflow(name: str, username: str) -> Workflow:
    """Load a workflow from the TOML definition.

    Parameters
    ==========
    name
      Name of the workflow. Will look for a file named
      "{username}.toml".
    username
      The username associated with this workflow. Will be used to set
      *owner* and *userAccount*.

    """
    toml_file = workflow_dir / f"{name}.toml"
    with open(toml_file, mode="rb") as toml_fd:
        wf_dict = {
            **tomllib.load(toml_fd),
            **{"userAccount": username, "owner": username},
        }
        workflow = Workflow(**wf_dict)
    return workflow
