"""A prefect flow for backing up databases e.g. the Bluesky catalog of runs."""

import asyncio
import datetime as dt
import os
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.variables import Variable
from prefect_sqlalchemy import AsyncSqlAlchemyConnector


@task()
async def dump_postgres(
    username: str,
    password: str,
    host: str,
    port: str,
    database: str,
) -> Path:
    # Decide where to store the backup
    log = get_run_logger()
    root_dir = await Variable.get("database-backup-path")
    if root_dir is None:
        raise ValueError("Variable 'database-backup-path' not set.")
    root_dir = Path(root_dir)
    now = dt.datetime.now()
    target_dir = root_dir / f"{database}-{now.strftime('%Y-%m-%d-%H-%M')}"
    log.info(f"Backing up postgres server at '{host}:{port}/{database}'.")
    log.info(f"Saving to folder: '{target_dir}'.")
    # Perform the backup
    args = [
        "pg_dump",
        "--dbname",
        database,
        "--host",
        host,
        "--port",
        str(port),
        "--username",
        username,
        "--format",
        "d",
        "--file",
        str(target_dir),
        "--jobs",
        "8",
    ]
    proc = await asyncio.create_subprocess_shell(
        " ".join(args), env={**os.environ, "PGPASSWORD": password}
    )
    await proc.communicate()
    return target_dir


@task()
async def restore_backup(backup_dir: Path):
    log = get_run_logger()
    log.critical(f"Not restoring backup from {backup_dir}")


@task()
async def verify_backup(backup_dir: Path):
    log = get_run_logger()
    log.critical(f"Not verifying backup from {backup_dir}")


@flow()
async def backup_database(name: str):
    """Backup and verify a single database.

    Uses sqlalchemy block, so *name* is the name of the block to
    backup.

    """
    database_block = await AsyncSqlAlchemyConnector.load(name)
    connection = database_block.connection_info
    backup_dir = await dump_postgres(
        username=connection.username,
        password=connection.password.get_secret_value(),
        host=connection.host,
        port=connection.port,
        database=connection.database,
    )
    await restore_backup(backup_dir)
    await verify_backup(backup_dir)


@flow()
async def backup_databases():
    """Backup several databases to local storage and confirm their validity.

    Parameters
    ==========
    uri
      The URI for the postgres
      database. E.g. "postgres://<user>:<password>@host:port".

    """
    results = await asyncio.gather(
        backup_database("bluesky-catalog"),
        backup_database("bluesky-storage"),
        backup_database("bluesky-results"),
        return_exceptions=True,
    )
    exceptions = [exc for exc in results if isinstance(exc, BaseException)]
    if any(exceptions):
        raise ExceptionGroup("Database backup failed", exceptions)


def main():
    asyncio.run(backup_databases())
    # backup_bluesky_catalog.serve(
    #     interval=60
    # )
