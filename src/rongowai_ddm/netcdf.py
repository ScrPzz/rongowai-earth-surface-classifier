"""Read one Rongowai L1 NetCDF flight into plain NumPy arrays.

Only the variables used downstream are read. Masked values of float variables
become NaN; masked values of integer variables become 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np

N_CHANNELS = 20
N_LHCP = 10  # channels 0..9 are LHCP, 10..19 the RHCP twins of the same reflections
ANT_LHCP = 2
ANT_RHCP = 3

REQUIRED_VARIABLES = (
    "raw_counts",
    "ddm_ant",
    "sp_surface_type",
    "ddm_snr",
    "sp_rx_gain_copol",
    "sp_rx_gain_xpol",
    "sp_alt",
    "sp_inc_angle",
    "ac_alt",
    "sp_lat",
    "sp_lon",
    "ddm_noise_floor",
)

OPTIONAL_VARIABLES = {
    "prn_code": ("prn", np.uint8),
    "coherence_state": ("coherence_state", np.uint8),
    "coherence_metric": ("coherence_metric", np.float32),
    "sp_dist_to_coast_km": ("dist_to_coast_km", np.float32),
    "quality_flags1": ("quality_flags", np.uint32),
    "ddm_snr_flag": ("snr_flag", np.uint8),
    "ddm_timestamp_utc": ("timestamp_utc", np.float64),
}


@dataclass
class FlightArrays:
    """Arrays of one flight. Shapes: ``(S, 20, 40, 5)`` for DDMs, ``(S, 20)`` per DDM, ``(S,)`` per epoch."""

    flight_id: str
    raw_counts: np.ndarray
    ddm_ant: np.ndarray
    surface_type: np.ndarray
    snr: np.ndarray
    gain_copol: np.ndarray
    gain_xpol: np.ndarray
    sp_alt: np.ndarray
    sp_inc_angle: np.ndarray
    ac_alt: np.ndarray
    sp_lat: np.ndarray
    sp_lon: np.ndarray
    noise_floor: np.ndarray
    prn: np.ndarray
    coherence_state: np.ndarray
    coherence_metric: np.ndarray
    dist_to_coast_km: np.ndarray
    quality_flags: np.ndarray
    snr_flag: np.ndarray
    timestamp_utc: np.ndarray

    @property
    def n_epochs(self) -> int:
        return int(self.raw_counts.shape[0])


def _as_float(var, dtype=np.float32) -> np.ndarray:
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.astype(dtype).filled(np.nan)
    return np.asarray(a, dtype=dtype)


def _as_int(var, dtype) -> np.ndarray:
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.filled(0)
    return np.asarray(a, dtype=dtype)


def read_flight(path: str | Path, flight_id: str | None = None) -> FlightArrays:
    """Read the variables needed by the pipeline; raises ``KeyError`` on a malformed file."""
    path = Path(path)
    with netCDF4.Dataset(str(path), "r") as ds:
        missing = [v for v in REQUIRED_VARIABLES if v not in ds.variables]
        if missing:
            raise KeyError(f"{path.name}: missing variables {missing}")
        rc = ds.variables["raw_counts"]
        if rc.ndim != 4 or rc.shape[1] != N_CHANNELS:
            raise ValueError(f"{path.name}: unexpected raw_counts shape {rc.shape}")
        raw = _as_int(rc, np.uint32)
        n = raw.shape[0]
        optional = {}
        for var, (name, dtype) in OPTIONAL_VARIABLES.items():
            if var in ds.variables:
                v = ds.variables[var]
                optional[name] = (
                    _as_float(v, dtype) if np.issubdtype(dtype, np.floating) else _as_int(v, dtype)
                )
            else:
                shape = (n,) if name == "timestamp_utc" else (n, N_CHANNELS)
                fill = np.nan if np.issubdtype(dtype, np.floating) else 0
                optional[name] = np.full(shape, fill, dtype=dtype)
        return FlightArrays(
            flight_id=flight_id or path.stem,
            raw_counts=raw,
            ddm_ant=_as_int(ds.variables["ddm_ant"], np.uint8),
            surface_type=_as_float(ds.variables["sp_surface_type"]),
            snr=_as_float(ds.variables["ddm_snr"]),
            gain_copol=_as_float(ds.variables["sp_rx_gain_copol"]),
            gain_xpol=_as_float(ds.variables["sp_rx_gain_xpol"]),
            sp_alt=_as_float(ds.variables["sp_alt"]),
            sp_inc_angle=_as_float(ds.variables["sp_inc_angle"]),
            ac_alt=_as_float(ds.variables["ac_alt"]),
            sp_lat=_as_float(ds.variables["sp_lat"], np.float64),
            sp_lon=_as_float(ds.variables["sp_lon"], np.float64),
            noise_floor=_as_float(ds.variables["ddm_noise_floor"]),
            **optional,
        )
