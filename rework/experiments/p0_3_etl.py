"""P0.3 ETL: extract 78-D features per flight, respecting the P0.1 fold harness.

For each flight in the P0.1 ``flight_index.parquet``:

1. Read the NetCDF, apply quality filters.
2. Normalize the (N, 200) raw counts per file (MinMaxScaler.fit_transform, matching production).
3. Encode through the pre-trained autoencoder → (N, 20) latent.
4. Extract 58 statistical features per sample → (N, 58).
5. Concatenate → (N, 78) feature matrix.
6. Attach metadata (flight_id, sample_idx, fold_id, split_type, label).
7. Append to an Arrow Parquet dataset partitioned by split_type.

Usage::

    # smoke test on a tiny subset
    python rework/experiments/p0_3_etl.py --max-flights 10

    # full run (slow!)
    python rework/experiments/p0_3_etl.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from rework.data import (  # noqa: E402
    DDMFeatureExtractor,
    NetCDFFlightPreprocessor,
    STAT_FEATURE_NAMES,
    compress_array,
    load_encoder,
)

# Default encoder weights = the production encoder shipped under deliver/.
DEFAULT_ENCODER_PATH = _PROJECT_ROOT / "geo_k" / "raw_counts" / "encoder_enh.pth"
DEFAULT_FOLD_INDEX = _PROJECT_ROOT / "rework" / "results" / "p0_1_flight_index.parquet"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "rework" / "results" / "p0_3_features"

# 20 encoder dims + 58 stats = 78 features (production composition)
ENC_FEATURE_NAMES = tuple(f"enc_{i:02d}" for i in range(20))
FEATURE_NAMES = ENC_FEATURE_NAMES + STAT_FEATURE_NAMES


def _build_arrow_schema() -> pa.Schema:
    fields = [
        pa.field("flight_id", pa.string()),
        pa.field("sample_idx", pa.int32()),
        pa.field("fold_id", pa.int8()),
        pa.field("split_type", pa.string()),
        pa.field("label", pa.int8()),
        pa.field("sp_delay_row", pa.float32()),
        pa.field("sp_dopp_col", pa.float32()),
    ]
    for name in FEATURE_NAMES:
        fields.append(pa.field(name, pa.float32()))
    return pa.schema(fields)


def _row_table(
    flight_id: str,
    fold_id: int,
    split_type: str,
    encoded: np.ndarray,
    stats: np.ndarray,
    labels: np.ndarray,
    sp_centers: np.ndarray,
    schema: pa.Schema,
) -> pa.Table:
    n = encoded.shape[0]
    cols: dict[str, np.ndarray | list] = {
        "flight_id": [flight_id] * n,
        "sample_idx": np.arange(n, dtype=np.int32),
        "fold_id": np.full(n, int(fold_id), dtype=np.int8),
        "split_type": [split_type] * n,
        "label": labels.astype(np.int8),
        "sp_delay_row": sp_centers[:, 0].astype(np.float32),
        "sp_dopp_col": sp_centers[:, 1].astype(np.float32),
    }
    combined = np.concatenate([encoded, stats], axis=1)
    for i, name in enumerate(FEATURE_NAMES):
        cols[name] = combined[:, i].astype(np.float32)
    return pa.table(cols, schema=schema)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-index",
        type=Path,
        default=DEFAULT_FOLD_INDEX,
        help="P0.1 flight_index.parquet path.",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=DEFAULT_ENCODER_PATH,
        help="Path to the trained encoder .pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory under which per-split Parquet datasets are written.",
    )
    parser.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="If set, process only the first N flights (smoke test).",
    )
    parser.add_argument(
        "--split-types",
        nargs="+",
        default=None,
        help=(
            "Restrict processing to these split_type values "
            "(e.g. 'train_cv val_geographic_holdout'). Default: all."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the encoder forward pass.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoder DataLoader batch size (production default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.fold_index.exists():
        raise FileNotFoundError(
            f"Fold index not found at {args.fold_index}. Run p0_1_build_harness.py first."
        )
    if not args.encoder.exists():
        raise FileNotFoundError(f"Encoder weights not found at {args.encoder}")

    print(f"[P0.3 ETL] Loading fold assignments from {args.fold_index}")
    flight_index = pd.read_parquet(args.fold_index)
    print(f"[P0.3 ETL] {len(flight_index)} flights in index.")

    if args.split_types:
        before = len(flight_index)
        flight_index = flight_index[flight_index["split_type"].isin(args.split_types)]
        print(
            f"[P0.3 ETL] Filtered to split_types={args.split_types}: "
            f"{len(flight_index)} / {before} flights remain."
        )

    if args.max_flights is not None:
        flight_index = flight_index.head(args.max_flights)
        print(f"[P0.3 ETL] SMOKE TEST: processing first {len(flight_index)} flights only.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P0.3 ETL] Loading encoder from {args.encoder} (device={args.device})")
    encoder = load_encoder(args.encoder, device=args.device)
    preprocessor = NetCDFFlightPreprocessor(preprocessing_method="filtered")
    extractor = DDMFeatureExtractor()
    schema = _build_arrow_schema()

    # One ParquetWriter per (split_type, partition file). We write one
    # Parquet file per split_type to keep things simple and fast to read back.
    writers: dict[str, pq.ParquetWriter] = {}

    t_start = time.time()
    stats = {
        "n_flights_processed": 0,
        "n_flights_failed": 0,
        "n_flights_empty": 0,
        "n_samples_total": 0,
        "n_samples_per_split": {},
        "elapsed_seconds": 0.0,
        "failures": [],
    }

    try:
        for row in flight_index.itertuples(index=False):
            flight_id = row.flight_id
            fold_id = int(row.fold_id) if row.fold_id is not None else -1
            split_type = row.split_type
            file_path = Path(row.file_path)

            try:
                samples = preprocessor.process_flight(file_path, flight_id=flight_id)
            except Exception as exc:
                stats["n_flights_failed"] += 1
                stats["failures"].append({"flight_id": flight_id, "error": str(exc)})
                print(f"[P0.3 ETL] FAIL {flight_id}: {exc}")
                continue

            if len(samples) == 0:
                stats["n_flights_empty"] += 1
                continue

            encoded = compress_array(
                samples.raw_ddm,
                encoder=encoder,
                fit_scaler_per_call=True,
                batch_size=args.batch_size,
                device=args.device,
            )
            stat_feats = extractor.extract(samples.raw_ddm)

            assert encoded.shape == (len(samples), 20), (
                f"Encoder output shape mismatch for {flight_id}: {encoded.shape}"
            )
            assert stat_feats.shape == (len(samples), len(STAT_FEATURE_NAMES)), (
                f"Stat feats shape mismatch for {flight_id}: {stat_feats.shape} "
                f"vs expected ({len(samples)}, {len(STAT_FEATURE_NAMES)})"
            )

            table = _row_table(
                flight_id=flight_id,
                fold_id=fold_id,
                split_type=split_type,
                encoded=encoded,
                stats=stat_feats,
                labels=samples.labels,
                sp_centers=samples.sp_centers,
                schema=schema,
            )

            if split_type not in writers:
                out_path = args.output_dir / f"{split_type}.parquet"
                writers[split_type] = pq.ParquetWriter(
                    out_path, schema, compression="zstd"
                )
                print(f"[P0.3 ETL] Opened writer for split '{split_type}' → {out_path}")

            writers[split_type].write_table(table)

            stats["n_flights_processed"] += 1
            stats["n_samples_total"] += len(samples)
            stats["n_samples_per_split"][split_type] = (
                stats["n_samples_per_split"].get(split_type, 0) + len(samples)
            )

            if stats["n_flights_processed"] % 10 == 0:
                elapsed = time.time() - t_start
                rate = stats["n_flights_processed"] / max(elapsed, 1e-9)
                print(
                    f"[P0.3 ETL] {stats['n_flights_processed']} flights | "
                    f"{stats['n_samples_total']} samples | "
                    f"{rate:.2f} flights/s | {elapsed:.1f}s elapsed"
                )
    finally:
        for writer in writers.values():
            writer.close()
        stats["elapsed_seconds"] = time.time() - t_start

    report_path = args.output_dir / "p0_3_etl_report.json"
    with report_path.open("w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n[P0.3 ETL] === SUMMARY ===")
    print(f"  Processed: {stats['n_flights_processed']}")
    print(f"  Empty   : {stats['n_flights_empty']}")
    print(f"  Failed  : {stats['n_flights_failed']}")
    print(f"  Samples : {stats['n_samples_total']}")
    print(f"  Per split:")
    for split, n in sorted(stats["n_samples_per_split"].items()):
        print(f"    {split:42s}: {n}")
    print(f"  Elapsed: {stats['elapsed_seconds']:.1f}s")
    print(f"\n[P0.3 ETL] Wrote report to {report_path}")


if __name__ == "__main__":
    main()
