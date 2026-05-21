import tomllib
from collections.abc import Mapping
from pathlib import Path

from platformdirs import user_config_path
from pydantic import BaseModel


class TiledConfig(BaseModel):
    websocket_uri: str


class DataManagementStation(BaseModel):
    """Maps directly onto ``dmax.AsyncClient()``."""

    username: str
    password: str
    station_name: str
    scheduling_uri: str = ""
    data_storage_uri: str = ""
    processing_uri: str = ""


class OatyBarConfig(BaseModel):
    tiled: TiledConfig
    data_management_stations: Mapping[str, DataManagementStation] = {}


def load_config(config_file: Path | None = None) -> OatyBarConfig:
    # Pick a sensible default place to store config files
    if config_file is None:
        config_file = user_config_path("oaty-bar") / "config.toml"
    # Load the configuration
    with open(config_file, mode="rb") as config_fd:
        cfg_dict = tomllib.load(config_fd)
    return OatyBarConfig(**cfg_dict)
