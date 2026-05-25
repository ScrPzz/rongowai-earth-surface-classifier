"""NetCDF quality-filtering and DDM extraction per flight.

Vendored from the production class ``NetCDFPreprocessor`` defined in
``deliver/[ETL]raw_counts_dataset_generator.ipynb``, with the following
deliberate modifications:

* Operates **per flight** (one NetCDF file at a time), not as a corpus-level
  orchestrator. Corpus-level orchestration is handled by
  ``rework/experiments/p0_3_etl.py`` so it can apply the flight-level fold
  assignment from P0.1.
* Always returns the flight_id-aligned ``sp_centers`` and ``sample_idx``
  alongside the data, so downstream joins to the fold table are trivial.
* No filesystem side-effects; the caller decides where to persist the output.

Quality filters and the binary label rule are byte-identical to production:
- SNR > 0 dB (only in the 'filtered' path)
- Copolarized gain ≥ 5 dB
- Cross-polarized gain ≥ 5 dB
- Specular-point distance ∈ [2000, 10000] m
- No NaN values in the filter variables
- No zero-sum DDMs
- Label: ``sp_surface_type`` ∈ {1, …, 7} → 1 (land), else 0 (water).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import netCDF4
import numpy as np

QUALITY_THRESHOLDS = {
    "snr_min_db": 0.0,
    "copol_min_db": 5.0,
    "xpol_min_db": 5.0,
    "sp_distance_min_m": 2000.0,
    "sp_distance_max_m": 10000.0,
}


@dataclass
class FlightSamples:
    """Per-flight output of the preprocessor."""

    flight_id: str
    raw_ddm: np.ndarray            # shape (N, 200), float32
    labels: np.ndarray             # shape (N,), int32 (0 = water, 1 = land)
    sp_centers: np.ndarray         # shape (N, 2), float32 (delay_row, dopp_col)

    def __len__(self) -> int:
        return int(self.raw_ddm.shape[0])


class NetCDFFlightPreprocessor:
    """Per-flight NetCDF reader + quality filter + DDM flattener."""

    REQUIRED_VARIABLES = (
        "raw_counts",
        "sp_alt",
        "sp_inc_angle",
        "sp_rx_gain_copol",
        "sp_rx_gain_xpol",
        "ddm_snr",
        "sp_lat",
        "sp_lon",
        "sp_surface_type",
        "ac_alt",
        "brcs_ddm_peak_bin_delay_row",
        "brcs_ddm_peak_bin_dopp_col",
    )

    def __init__(
        self,
        preprocessing_method: Literal["filtered", "unfiltered"] = "filtered",
    ) -> None:
        if preprocessing_method not in ("filtered", "unfiltered"):
            raise ValueError(
                f"preprocessing_method must be 'filtered' or 'unfiltered'; "
                f"got {preprocessing_method!r}"
            )
        self.preprocessing_method = preprocessing_method

    @classmethod
    def _check_integrity(cls, f: netCDF4.Dataset) -> None:
        missing = [v for v in cls.REQUIRED_VARIABLES if v not in f.variables]
        if missing:
            raise KeyError(
                f"NetCDF file missing required variable(s): {missing}"
            )
        if f.variables["raw_counts"].ndim != 4:
            raise ValueError(
                f"'raw_counts' must have 4 dimensions; got "
                f"{f.variables['raw_counts'].ndim}"
            )

    @staticmethod
    def _label_from_surface_type(surface_types: np.ndarray) -> np.ndarray:
        """Binary land/water label: surface_type ∈ {1..7} → 1, else 0."""
        st = np.nan_to_num(surface_types, nan=0).ravel()
        return np.isin(st, np.arange(1, 8)).astype(np.int32)

    def _quality_mask(self, f: netCDF4.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return (keep_mask, distance_2d) computed from per-sample metadata.

        keep_mask is shape (n_time, n_samples) and selects which DDMs pass
        all quality thresholds; distance_2d is returned for caller inspection.
        """
        ac_alt = f.variables["ac_alt"][:]
        sp_alt = f.variables["sp_alt"][:]
        copol = f.variables["sp_rx_gain_copol"][:]
        xpol = f.variables["sp_rx_gain_xpol"][:]
        sp_inc_angle = f.variables["sp_inc_angle"][:]

        distance_2d = (
            ac_alt[:, np.newaxis] - sp_alt
        ) / np.cos(np.deg2rad(sp_inc_angle))

        keep = (
            (copol >= QUALITY_THRESHOLDS["copol_min_db"])
            & (xpol >= QUALITY_THRESHOLDS["xpol_min_db"])
            & (distance_2d >= QUALITY_THRESHOLDS["sp_distance_min_m"])
            & (distance_2d <= QUALITY_THRESHOLDS["sp_distance_max_m"])
            & ~np.isnan(copol)
            & ~np.isnan(xpol)
            & ~np.isnan(distance_2d)
        )

        if self.preprocessing_method == "filtered":
            snr = f.variables["ddm_snr"][:]
            keep = keep & (snr > QUALITY_THRESHOLDS["snr_min_db"]) & ~np.isnan(snr)

        return keep, distance_2d

    def process_flight(
        self,
        file_path: Path | str,
        flight_id: Optional[str] = None,
    ) -> FlightSamples:
        """Read one NetCDF flight and return the surviving samples."""
        file_path = Path(file_path)
        if flight_id is None:
            flight_id = file_path.stem

        with netCDF4.Dataset(str(file_path), "r") as f:
            self._check_integrity(f)
            keep_mask, _ = self._quality_mask(f)

            raw_counts = f.variables["raw_counts"][:]
            sp_row = f.variables["brcs_ddm_peak_bin_delay_row"][:]
            sp_col = f.variables["brcs_ddm_peak_bin_dopp_col"][:]
            surface_types = f.variables["sp_surface_type"][:]

        # Apply mask: keep only DDMs that pass quality; the rest become NaN
        n_time, n_samples = raw_counts.shape[:2]
        aux_counts = np.full(raw_counts.shape, np.nan, dtype=np.float32)
        aux_row = np.full(sp_row.shape, np.nan, dtype=np.float32)
        aux_col = np.full(sp_col.shape, np.nan, dtype=np.float32)

        i_idx, j_idx = np.where(keep_mask)
        aux_counts[i_idx, j_idx] = raw_counts[i_idx, j_idx]
        aux_row[i_idx, j_idx] = sp_row[i_idx, j_idx]
        aux_col[i_idx, j_idx] = sp_col[i_idx, j_idx]

        # Flatten to (n_time*n_samples, ...) then drop NaN and zero-sum rows
        counts_flat = aux_counts.reshape(n_time * n_samples, *raw_counts.shape[2:])
        row_flat = aux_row.reshape(n_time * n_samples, *aux_row.shape[2:])
        col_flat = aux_col.reshape(n_time * n_samples, *aux_col.shape[2:])

        valid_mask = ~np.any(np.isnan(counts_flat), axis=(1, 2)) & (
            np.sum(counts_flat, axis=(1, 2)) > 0
        )

        if not valid_mask.any():
            return FlightSamples(
                flight_id=flight_id,
                raw_ddm=np.empty((0, 200), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int32),
                sp_centers=np.empty((0, 2), dtype=np.float32),
            )

        ddm_kept = counts_flat[valid_mask].reshape(valid_mask.sum(), -1)
        row_kept = row_flat[valid_mask].astype(np.float32).ravel()
        col_kept = col_flat[valid_mask].astype(np.float32).ravel()
        sp_centers = np.stack([row_kept, col_kept], axis=1)

        # Labels: the surface_type array is shaped like raw_counts[:, :, 0, 0]
        # (i.e. per-epoch). We mirror the original by flattening it to (n_time*n_samples,)
        # via .ravel() and applying the same valid_mask.
        labels = self._label_from_surface_type(surface_types)[valid_mask]

        assert ddm_kept.shape[0] == labels.shape[0] == sp_centers.shape[0], (
            f"Shape mismatch in flight {flight_id}: "
            f"ddm={ddm_kept.shape}, labels={labels.shape}, sp={sp_centers.shape}"
        )

        return FlightSamples(
            flight_id=flight_id,
            raw_ddm=ddm_kept.astype(np.float32),
            labels=labels,
            sp_centers=sp_centers.astype(np.float32),
        )
