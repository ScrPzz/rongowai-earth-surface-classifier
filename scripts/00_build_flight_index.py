"""Step 0: index the flight files and assign the flight-level holdouts.

Writes ``results/flights.parquet`` (one row per flight with ``split``) and
``results/flight_index_report.json``. Fold ids are assigned after the ETL, when
the per-flight class mix is known.
"""

from __future__ import annotations

from _common import base_parser, load_config, save_json

from rongowai_ddm.data.flight_index import build_flight_index
from rongowai_ddm.data.splits import assign_holdouts, drop_invalid_airports, unknown_airports


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    flights = build_flight_index(cfg.data_dir)
    n_raw = len(flights)
    flights, dropped = drop_invalid_airports(flights)
    unknown = sorted(unknown_airports(flights))
    flights = assign_holdouts(flights, cfg.split.temporal_holdout_months)
    flights["fold_id"] = -1
    flights["flight_idx"] = range(len(flights))
    flights.to_parquet(cfg.flight_table_path, index=False)

    routes = flights.groupby(["origin", "dest"]).size().sort_values(ascending=False)
    report = {
        "n_files": n_raw,
        "n_dropped_invalid_airport": int(len(dropped)),
        "n_flights": int(len(flights)),
        "skipped_filenames": flights.attrs.get("skipped", []),
        "unknown_airports": unknown,
        "date_min": str(flights["date"].min()),
        "date_max": str(flights["date"].max()),
        "temporal_cutoff": str(flights["temporal_cutoff"].iloc[0]),
        "total_bytes": int(flights["file_size"].sum()),
        "split_counts": flights["split"].value_counts().to_dict(),
        "island_counts": {
            f"{o}-{d}": int(n)
            for (o, d), n in flights.groupby(["origin_island", "dest_island"]).size().items()
        },
        "top_routes": {f"{o}-{d}": int(n) for (o, d), n in routes.head(15).items()},
    }
    for k, v in report["split_counts"].items():
        print(f"  {k:18s} {v:5d} flights")
    if unknown:
        print(f"  WARNING airports not mapped to an island (treated as North Island): {unknown}")
    print(f"wrote {cfg.flight_table_path}")
    save_json(report, cfg.results_dir / "flight_index_report.json")


if __name__ == "__main__":
    main()
