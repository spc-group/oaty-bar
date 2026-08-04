"""A prefect flow for backing up databases e.g. the Bluesky catalog of runs."""

import argparse
import asyncio
import dataclasses
import datetime as dt
import os
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from prefect.schedules import Cron
from prefect.variables import Variable
from prefect_sqlalchemy import AsyncSqlAlchemyConnector


@dataclasses.dataclass(frozen=True, eq=True)
class DBInfo:
    port: int
    host: str
    db: str = ""
    username: str = ""
    password: str = ""


async def prefect_db_info(name) -> DBInfo:
    async with await AsyncSqlAlchemyConnector.load(name) as database_block:
        connection = database_block.connection_info
        dbinfo = DBInfo(
            db=connection.database,
            host=connection.host,
            port=int(connection.port),
            username=connection.username,
            password=connection.password.get_secret_value(),
        )
        return dbinfo


@task(
    task_run_name="dump-{name}",
    persist_result=True,
)
async def dump_postgres(name: str) -> Path:
    """Create a dumped directory of the postgres database."""
    # Database connection info is stored in prefect blocks for security
    dbinfo = await prefect_db_info(name)
    # Decide where to store the backup
    log = get_run_logger()
    root_dir = await Variable.get("database-backup-path")
    if root_dir is None:
        raise ValueError("Variable 'database-backup-path' not set.")
    root_dir = Path(root_dir)
    now = dt.datetime.now()
    target_dir = root_dir / f"{name}-{now.strftime('%Y-%m-%d-%H-%M')}"
    db_uri = f"{dbinfo.host}:{dbinfo.port}/{dbinfo.db}"
    log.info(f"Backing up postgres server at '{db_uri}'.")
    log.info(f"Saving to folder: '{target_dir}'.")
    # Perform the backup
    args = [
        "pg_dump",
        "--dbname",
        dbinfo.db,
        "--host",
        dbinfo.host,
        "--port",
        str(dbinfo.port),
        "--username",
        dbinfo.username,
        "--format",
        "d",
        "--file",
        str(target_dir),
        "--jobs",
        "8",
    ]
    proc = await asyncio.create_subprocess_shell(
        " ".join(args),
        env={**os.environ, "PGPASSWORD": dbinfo.password},
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
        key=f"backup-{dbinfo.db}-folder",
        link=f"file://{target_dir}",
        description=f"# Backup folder of {dbinfo.db}\n\n{now.strftime('%Y-%m-%d %H:%M')}\n\nBackup of '{db_uri}'.",
    )
    return target_dir


@task(task_run_name="restore-{backup_dir.name}")
async def restore_backup(backup_dir: Path, dbinfo: DBInfo):
    log = get_run_logger()
    backup_file = backup_dir
    log.info(f"Restoring backup from {backup_file}")
    binary = shutil.which("pg_restore")
    if binary is None:
        raise RuntimeError("Could not determine binary path for 'pg_restore'")
    args = [
        "-h",
        dbinfo.host,
        "-p",
        str(dbinfo.port),
        "-d",
        dbinfo.db,
        "-F",
        "d",
        str(backup_file),
    ]
    log.debug(f"Restoring with command: {binary} {' '.join(args)}")
    proc = await asyncio.create_subprocess_exec(binary, *args)
    out, err = await proc.communicate()
    if err:
        log.error(err)


@asynccontextmanager
async def db_cursor(dbinfo: DBInfo, *, autocommit: bool = False):
    """Wrapper around psycopg cursor."""
    conninfo = f"host={dbinfo.host} port={dbinfo.port} dbname={dbinfo.db or 'postgres'}"
    async with await psycopg.AsyncConnection.connect(
        conninfo, autocommit=autocommit
    ) as aconn:
        async with aconn.cursor() as acur:
            yield acur


@task(task_run_name="verify-{backup_dir.name}")
async def verify_backup(backup_dir: Path, dbinfo: DBInfo):
    log = get_run_logger()
    log.info(f"Verifying backup from {backup_dir}")
    # Get the source database information
    src_info = await prefect_db_info(dbinfo.db)
    async with db_cursor(dbinfo) as cursor:
        await cursor.execute(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';"
        )
        tables = await cursor.fetchall()
        tables = [tablename[0] for tablename in tables]
        table_counts = {}
        for tablename in tables:
            await cursor.execute(f"SELECT COUNT(*) FROM {tablename};")
            table_counts[tablename] = (await cursor.fetchone())[0]
    async with db_cursor(dbinfo) as cursor:
        # Checks that the database was restored properly
        exceptions = []
        for table, count in table_counts.items():
            await cursor.execute(f"SELECT COUNT(*) FROM {table};")
            new_count = (await cursor.fetchone())[0]
            if new_count != count:
                exceptions.append(
                    AssertionError(
                        f"Table {dbinfo.db}.{table} not fully restorable. Expected {count} rows, got {new_count} rows."
                    )
                )
        if len(exceptions) > 2:
            return ExceptionGroup(f"Could not verify {dbinfo.db}.", exceptions)
        elif len(exceptions) == 1:
            return exceptions[0]


@asynccontextmanager
async def temporary_db(timeout=10):
    log = get_run_logger()
    # Get a random port to make sure we don't conflict with any open posgres servers
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    dbinfo = DBInfo(port=port, host="127.0.0.1", db=None)
    sock.close()
    initdb_binary = shutil.which("initdb")
    postgres_binary = shutil.which("postgres")
    with tempfile.TemporaryDirectory() as db_folder:
        subprocess.run([initdb_binary, "-D", db_folder])
        log.debug(f"Created inital database configuration in {db_folder}")
        dbproc = subprocess.Popen(
            [
                "postgres",
                "-D",
                db_folder,
                "-p",
                str(port),
                "-h",
                "127.0.0.1",
                # Reduce the WAL since we don't care about recovery
                "--wal-level=minimal",
                "--max_wal_senders=0",
            ]
        )
        # Make sure the database is ready to accept connections
        t0 = time.monotonic()
        while True:
            try:
                async with db_cursor(dbinfo):
                    break
            except psycopg.OperationalError:
                if (time.monotonic() - t0) > timeout:
                    raise
        log.info(f"Started ephemoral postgres server on port {port}")
        # Create roles and other infrastructure needed to use database
        async with db_cursor(dbinfo) as cursor:
            await cursor.execute(f'CREATE ROLE "bs-spcgroup";')
        # Let the calling code do its thing
        yield dbinfo
        try:
            dbproc.terminate()
            outs, errs = dbproc.communicate(timeout=60)
        except TimeoutError:
            dbproc.kill()
            outs, errs = dbproc.communicate()


async def create_database(dbinfo: DBInfo, name: str):
    """Create a database in the server and return a new dbinfo."""
    async with db_cursor(dbinfo, autocommit=True) as cursor:
        await cursor.execute(f'CREATE DATABASE "{name}";')
    log = get_run_logger()
    log.info(f"Created temporary database {name}")
    return dataclasses.replace(dbinfo, db=name)


async def backup_database(name: str, dbinfo: DBInfo):
    """Backup and verify a single database.

    Uses sqlalchemy block, so *name* is the name of the block to
    backup.

    """
    # Backup the existing database
    backup_dir = await dump_postgres(name)
    # Exercise the backup to make sure it can be restored
    dbinfo = await create_database(name=name, dbinfo=dbinfo)
    await restore_backup(backup_dir, dbinfo=dbinfo)
    return await verify_backup(backup_dir, dbinfo=dbinfo)


@flow()
async def backup_databases():
    """Backup several databases to local storage and confirm their validity."""
    async with temporary_db() as dbinfo:
        results = await asyncio.gather(
            backup_database("bluesky-catalog", dbinfo),
            backup_database("bluesky-storage", dbinfo),
            backup_database("bluesky-results", dbinfo),
            return_exceptions=True,
        )
    exceptions = [exc for exc in results if isinstance(exc, BaseException)]
    if len(exceptions) > 2:
        raise ExceptionGroup("Database backup failed", exceptions)
    elif len(exceptions) == 1:
        raise exceptions[0]


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
