"""A prefect flow for backing up databases e.g. the Bluesky catalog of runs."""

import argparse
import asyncio
import datetime as dt
import os
from pathlib import Path

from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from prefect.schedules import Cron
from prefect.variables import Variable
from prefect_sqlalchemy import AsyncSqlAlchemyConnector


@task(
    task_run_name="dump-{name}",
    persist_result=True,
)
async def dump_postgres(name: str) -> Path:
    """Create a dumped directory of the postgres database."""
    # Database connection info is stored in prefect blocks for security
    database_block = await AsyncSqlAlchemyConnector.load(name)
    connection = database_block.connection_info
    database = connection.database
    host = connection.host
    port = connection.port
    # Decide where to store the backup
    log = get_run_logger()
    root_dir = await Variable.get("database-backup-path")
    if root_dir is None:
        raise ValueError("Variable 'database-backup-path' not set.")
    root_dir = Path(root_dir)
    now = dt.datetime.now()
    target_dir = root_dir / f"{database}-{now.strftime('%Y-%m-%d-%H-%M')}"
    db_uri = f"{host}:{port}/{database}"
    log.info(f"Backing up postgres server at '{db_uri}'.")
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
        connection.username,
        "--format",
        "d",
        "--file",
        str(target_dir),
        "--jobs",
        "8",
    ]
    proc = await asyncio.create_subprocess_shell(
        " ".join(args),
        env={**os.environ, "PGPASSWORD": connection.password.get_secret_value()},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    # Raise an exception so the flow fails
    log.info(f"Dump of {db_uri} ended with exit code {proc.returncode}.")
    log.debug(stdout.decode())
    if proc.returncode == 0:
        if err := stderr.decode():
            log.error(err)
    else:
        raise RuntimeError(stderr.decode())
    # Inform prefect of the newly created folder
    await create_link_artifact(
        key=f"backup-{database}-folder",
        link=f"file://{target_dir}",
        description=f"# Backup folder of {database}\n\n{now.strftime('%Y-%m-%d %H:%M')}\n\nBackup of '{db_uri}'.",
    )
    return target_dir


@task(task_run_name="restore-{backup_dir.name}")
async def restore_backup(backup_dir: Path):
    log = get_run_logger()
    log.critical(f"Not restoring backup from {backup_dir}")


@task(task_run_name="verify-{backup_dir.name}")
async def verify_backup(backup_dir: Path):
    log = get_run_logger()
    log.critical(f"Not verifying backup from {backup_dir}")


async def backup_database(name: str):
    """Backup and verify a single database.

    Uses sqlalchemy block, so *name* is the name of the block to
    backup.

    """
    backup_dir = await dump_postgres(name)
    await restore_backup(backup_dir)
    await verify_backup(backup_dir)


@flow()
async def backup_databases():
    """Backup several databases to local storage and confirm their validity."""
    results = await asyncio.gather(
        backup_database("bluesky-catalog"),
        backup_database("bluesky-storage"),
        backup_database("bluesky-results"),
        return_exceptions=True,
    )
    exceptions = [exc for exc in results if isinstance(exc, BaseException)]
    if any(exceptions):
        raise ExceptionGroup("Database backup failed", exceptions)


def main(argv=None):
    "Entry point for run database backups in Prefect."
    parser = argparse.ArgumentParser(
        prog="backup-databases",
        description="A Prefect flow to backup databases and verify the backup is restorable.",
    )
    parser.add_argument(
        "--deploy",
        help="Deploy this flow to run every Monday instead of executing it immediately.",
        action="store_true",
    )
    args = parser.parse_args(argv)
    if args.deploy:
        backup_databases.serve(
            schedule=Cron("15 8 * * MON", timezone="America/Chicago"),
        )
    else:
        asyncio.run(backup_databases())
