"""P0.1 — Build the flight-level validation harness for the Rongowai pipeline.

Usage:
    python -m rework.experiments.p0_1_build_harness
    # or, from the project root:
    python rework/experiments/p0_1_build_harness.py

Outputs (under ``rework/results/``):
    p0_1_flight_index.parquet      # one row per flight with metadata + island/holdout flags
    p0_1_fold_assignments.parquet  # flight_id × fold_id × split_type
    p0_1_report.json               # split counts, class balance per fold, unknown airports
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rework.validation import (  # noqa: E402
    annotate_island,
    assign_holdouts,
    assign_train_cv_folds,
    build_flight_index,
    drop_invalid_icao,
    get_unknown_airports,
    run_all_checks,
)
from rework.validation.flight_index import summarize  # noqa: E402

DATA_DIR = Path("/media/atogni/dati/rongowai/data")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"

TEMPORAL_HOLDOUT_MONTHS = 6
N_SPLITS = 10
RANDOM_STATE = 42


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[P0.1] Scanning {DATA_DIR} for NetCDF files…")
    flights_raw = build_flight_index(DATA_DIR)
    n_raw = len(flights_raw)

    flights, dropped = drop_invalid_icao(flights_raw)
    n_dropped = len(dropped)
    if n_dropped:
        invalid_codes = sorted(
            set(dropped["origin"]).union(set(dropped["dest"]))
            - set(flights["origin"]).union(set(flights["dest"]))
        )
        print(
            f"[P0.1] Dropped {n_dropped} flight(s) with invalid ICAO codes "
            f"({invalid_codes}); kept {len(flights)} / {n_raw}."
        )

    summary = summarize(flights)
    print(
        f"[P0.1] Indexed {summary['n_flights']} flights "
        f"({summary['total_bytes'] / 1024**3:.1f} GiB total), "
        f"date range {summary['date_min']} → {summary['date_max']}"
    )
    print(
        f"[P0.1] {summary['n_unique_origins']} distinct origins, "
        f"{summary['n_unique_dests']} distinct destinations."
    )

    flights = annotate_island(flights)
    unknown = get_unknown_airports(flights)
    unknown_flight_counts = {}
    if unknown:
        for code in sorted(unknown):
            n_origin = int((flights["origin"] == code).sum())
            n_dest = int((flights["dest"] == code).sum())
            n_either = int(((flights["origin"] == code) | (flights["dest"] == code)).sum())
            unknown_flight_counts[code] = {
                "as_origin": n_origin,
                "as_dest": n_dest,
                "either": n_either,
            }
        print(
            f"[P0.1] WARNING: {len(unknown)} airport code(s) unmapped to an island: "
            f"{sorted(unknown)}"
        )
        for code, counts in unknown_flight_counts.items():
            print(
                f"    {code}: {counts['either']} flights affected "
                f"(origin: {counts['as_origin']}, dest: {counts['as_dest']})"
            )
        print(
            "[P0.1]   These flights are treated as 'NOT touching South Island' "
            "(conservative) — review and reclassify before P0.3 if needed."
        )

    flights = assign_holdouts(
        flights, temporal_holdout_months=TEMPORAL_HOLDOUT_MONTHS
    )
    flights = assign_train_cv_folds(
        flights, n_splits=N_SPLITS, random_state=RANDOM_STATE
    )

    report = run_all_checks(flights)
    report["n_flights_raw"] = int(n_raw)
    report["n_flights_dropped_invalid_icao"] = int(n_dropped)
    report["unknown_airports"] = sorted(unknown)
    report["unknown_airport_flight_counts"] = unknown_flight_counts
    report["flight_index_summary"] = summary
    report["temporal_holdout_months"] = TEMPORAL_HOLDOUT_MONTHS
    report["n_splits"] = N_SPLITS
    report["random_state"] = RANDOM_STATE
    report["temporal_cutoff"] = str(flights["temporal_cutoff"].iloc[0])

    print("[P0.1] Leak checks PASSED. Split sizes:")
    for split, n in sorted(report["split_counts"].items()):
        pct = report["split_proportions"][split] * 100.0
        print(f"    {split:42s}: n={n:>5d}  ({pct:5.1f}%)")

    flight_index_path = OUTPUT_DIR / "p0_1_flight_index.parquet"
    fold_assignments_path = OUTPUT_DIR / "p0_1_fold_assignments.parquet"
    report_path = OUTPUT_DIR / "p0_1_report.json"

    flights.to_parquet(flight_index_path, index=False)
    flights[["flight_id", "fold_id", "split_type"]].to_parquet(
        fold_assignments_path, index=False
    )
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[P0.1] Wrote:\n  {flight_index_path}\n  {fold_assignments_path}\n  {report_path}")


if __name__ == "__main__":
    main()
