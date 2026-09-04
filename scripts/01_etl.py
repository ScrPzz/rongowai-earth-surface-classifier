"""Step 1: extract the sample table from every flight and assign the CV folds.

Writes ``data/samples/part-*.parquet`` (one row per DDM), updates
``results/flights.parquet`` with per-flight counts and ``fold_id``, and writes
``results/etl_report.json``. Resumable: complete chunks are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads
from _common import Timer, base_parser, load_config, save_json

from rongowai_ddm.data.etl import run_etl
from rongowai_ddm.data.splits import assign_folds, check_flight_splits


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=64, help="flights per output file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test on the first N flights (writes to data/samples_smoke)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default: data/samples)"
    )
    args = parser.parse_args()
    cfg = load_config(args)

    flights = pd.read_parquet(cfg.flight_table_path)
    smoke = args.limit is not None
    out_dir = args.out or (cfg.work_dir / "data" / "samples_smoke" if smoke else cfg.samples_dir)
    if smoke:
        flights = flights.head(args.limit)
    print(f"{len(flights)} flights -> {out_dir}  ({args.workers} workers)")

    with Timer("etl") as t:
        summaries = run_etl(
            flights, cfg, out_dir, n_workers=args.workers, chunk_size=args.chunk_size
        )

    ds = pads.dataset(str(out_dir), format="parquet")
    rows_per_split = ds.to_table(columns=["split"]).to_pandas()["split"].value_counts().to_dict()
    failures = summaries[summaries["error"].notna()]
    report = {
        "n_flights": int(len(summaries)),
        "n_failed": int(len(failures)),
        "n_empty": int((summaries["n_selected"] == 0).sum()),
        "n_rows": int(sum(rows_per_split.values())),
        "rows_per_split": rows_per_split,
        "reflections_valid_total": int(summaries["n_valid"].sum()),
        "reflections_total": int(summaries["n_reflections"].sum()),
        "water_valid_total": int(summaries["n_water_valid"].sum()),
        "land_valid_total": int(summaries["n_land_valid"].sum()),
        "seconds": t.seconds,
        "failures": failures[["flight_id", "error"]].to_dict("records"),
    }
    print(
        f"rows: {report['n_rows']:,}  per split: {rows_per_split}  failed flights: {report['n_failed']}"
    )

    if smoke:
        save_json(report, out_dir / "_summaries" / "etl_report.json")
        return

    counts = summaries[
        [
            "flight_id",
            "n_epochs",
            "n_valid",
            "n_water_valid",
            "n_land_valid",
            "n_selected",
            "land_fraction",
            "error",
        ]
    ]
    flights = flights.drop(
        columns=[c for c in counts.columns if c != "flight_id" and c in flights.columns]
    )
    flights = flights.merge(counts, on="flight_id", how="left")
    flights = assign_folds(
        flights, n_folds=cfg.split.n_folds, seed=cfg.split.seed, stratify_col="land_fraction"
    )
    report["split_check"] = check_flight_splits(flights, cfg.split.n_folds)
    flights.to_parquet(cfg.flight_table_path, index=False)
    print(f"updated {cfg.flight_table_path} with counts and fold ids")
    save_json(report, cfg.results_dir / "etl_report.json")


if __name__ == "__main__":
    main()
