"""Prefect integration with APS data management systen.

The ``load_client()`` function will generate a dmax client by looking
up a DM station block.

The `DataManagementStation` block holds the needed credentials. For new installations, register the block using

```sh
prefect block register -m oaty_bar._data_management
```

"""

import dmax
from prefect.blocks.core import Block
from pydantic import SecretStr


class DataManagementStation(Block):
    """Information needed for accessing the APS data management APIs.

    By convention, the *name* of this block should be the **lowercase
    version of the station name**.

    Parameters
    ==========
    username
      API client username. Assigned by the SDM group.
    password
      API client password. Assigned by the SDM group.
    station_name
      The name of the DM station assigned by the SDM
      group. E.g. "25IDC".
    scheduling_uri
      The URI of the scheduling (BSS) API.
    data_storage_uri
      The URI of the data storage (DS) API.
    processing_uri
      The URI of the workflow processing (PROC) API.
    data_archive_uri
      The URI of the workflow processing (DAQ) API.

    """
    username: str
    password: SecretStr
    station_name: str
    scheduling_uri: str = ""
    data_storage_uri: str = ""
    processing_uri: str = ""
    data_archive_uri: str = ""


async def load_client(station_name: str, asyncio=True) -> dmax.Client:
    creds = await DataManagementStation.load(station_name.lower())
    Client = dmax.AsyncClient if asyncio else dmax.Client
    client = Client(
        username=creds.username,
        password=creds.password.get_secret_value(),
        station_name=creds.station_name,
        scheduling_uri=creds.scheduling_uri,
        data_storage_uri=creds.data_storage_uri,
        processing_uri=creds.processing_uri,
        data_archive_uri=creds.data_archive_uri,
    )
    return client
