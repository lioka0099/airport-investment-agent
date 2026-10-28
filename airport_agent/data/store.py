"""Read-only query layer over the parquet snapshot.

One in-memory DuckDB connection with a view per parquet file. Each query runs on its own
cursor, so Streamlit's threads can share one Store safely.
"""

import functools
import json
from pathlib import Path

import duckdb
import pandas as pd

from airport_agent import config

TABLES = ("airports", "t100_monthly", "ontime_airport_month", "ontime_routes")


class Store:
    def __init__(self, snapshot_dir: Path = config.SNAPSHOT_DIR) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        missing = [t for t in TABLES if not (self.snapshot_dir / f"{t}.parquet").exists()]
        if missing or not (self.snapshot_dir / "manifest.json").exists():
            raise FileNotFoundError(
                f"Snapshot incomplete in {self.snapshot_dir} "
                f"(missing: {missing or ['manifest.json']}). "
                "Build it with: uv run python -m airport_agent.data.ingest"
            )
        self._con = duckdb.connect()
        for table in TABLES:
            path = (self.snapshot_dir / f"{table}.parquet").as_posix().replace("'", "''")
            # Path is our own constant, not user input; DDL cannot take bound parameters.
            self._con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
        self._manifest = json.loads((self.snapshot_dir / "manifest.json").read_text())

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self._con.cursor().execute(sql, params or []).df()

    def airport_exists(self, code: str) -> bool:
        df = self.query("SELECT 1 FROM airports WHERE code = ? LIMIT 1", [code])
        return not df.empty

    def manifest(self) -> dict:
        return self._manifest


@functools.cache
def get_store() -> Store:
    return Store()
