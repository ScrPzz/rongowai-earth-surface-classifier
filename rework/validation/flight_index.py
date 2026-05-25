"""Build a flight-level index from the Rongowai NetCDF filename convention.

Each NetCDF file in the Rongowai L1 SDR dataset represents a single flight
named ``YYYYMMDD-HHMMSS_ORIGIN-DEST_L1.nc``. This module parses those
filenames to extract a stable ``flight_id``, the flight date, and the
origin/destination airport codes. No NetCDF I/O happens here — only filename
parsing and a single ``stat()`` call per file — so the indexer scales to the
full 7k+-file dataset in seconds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Strict regex for the Rongowai naming convention.
# Matches: 20221026-100450_NZRO-NZAA_L1.nc
_FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<time>\d{6})_"
    r"(?P<origin>[A-Z]{4})-(?P<dest>[A-Z]{4})_L1\.nc$"
)


@dataclass(frozen=True)
class FlightMetadata:
    """Per-flight metadata derived purely from the NetCDF filename + filesystem."""

    flight_id: str
    date: datetime
    origin: str
    dest: str
    file_path: Path
    file_size: int

    @classmethod
    def from_path(cls, path: Path) -> Optional["FlightMetadata"]:
        m = _FILENAME_RE.match(path.name)
        if m is None:
            return None
        dt = datetime.strptime(f"{m['date']}{m['time']}", "%Y%m%d%H%M%S")
        flight_id = f"{m['date']}-{m['time']}_{m['origin']}-{m['dest']}"
        return cls(
            flight_id=flight_id,
            date=dt,
            origin=m["origin"],
            dest=m["dest"],
            file_path=path,
            file_size=path.stat().st_size,
        )


def build_flight_index(data_dir: Path | str) -> pd.DataFrame:
    """Scan ``data_dir`` for Rongowai NetCDF files and return a flight-level DataFrame.

    Columns:
        flight_id, date, origin, dest, file_path, file_size, year, month
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    rows = []
    skipped: list[str] = []
    for nc_path in sorted(data_dir.glob("*.nc")):
        meta = FlightMetadata.from_path(nc_path)
        if meta is None:
            skipped.append(nc_path.name)
            continue
        rows.append(
            {
                "flight_id": meta.flight_id,
                "date": meta.date,
                "origin": meta.origin,
                "dest": meta.dest,
                "file_path": str(meta.file_path),
                "file_size": meta.file_size,
            }
        )

    if skipped:
        preview = ", ".join(skipped[:5])
        print(
            f"[flight_index] WARNING: {len(skipped)} file(s) did not match the "
            f"Rongowai naming convention and were skipped. First 5: {preview}"
        )

    if not rows:
        raise RuntimeError(f"No valid Rongowai NetCDF files found in {data_dir}")

    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.sort_values("date").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    """Return a small summary dict suitable for printing or JSON serialization."""
    return {
        "n_flights": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_unique_origins": int(df["origin"].nunique()),
        "n_unique_dests": int(df["dest"].nunique()),
        "total_bytes": int(df["file_size"].sum()),
        "top_routes": {
            f"{origin}-{dest}": int(count)
            for (origin, dest), count in (
                df.groupby(["origin", "dest"])
                .size()
                .sort_values(ascending=False)
                .head(10)
                .items()
            )
        },
    }
