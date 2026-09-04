from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rongowai_ddm.data.flight_index import parse_flight_filename
from rongowai_ddm.data.labels import LABEL_LAND, LABEL_UNKNOWN, LABEL_WATER, binary_label
from rongowai_ddm.data.sampling import sample_reflections
from rongowai_ddm.data.splits import (
    SPLIT_GEO,
    SPLIT_GEO_TIME,
    SPLIT_TIME,
    SPLIT_TRAIN,
    LeakError,
    assign_folds,
    assign_holdouts,
    check_flight_splits,
    check_sample_splits,
    drop_invalid_airports,
)


def test_parse_filename():
    meta = parse_flight_filename("20221026-100450_NZRO-NZAA_L1.nc")
    assert meta["flight_id"] == "20221026-100450_NZRO-NZAA"
    assert meta["origin"] == "NZRO" and meta["dest"] == "NZAA"
    assert meta["date"] == datetime(2022, 10, 26, 10, 4, 50)
    assert parse_flight_filename("notes.txt") is None


def test_binary_label():
    st = np.array([[-1, 1, 2, 3, 4, 5, 6, 7, np.nan, 0]])
    lab = binary_label(st)
    assert lab.tolist() == [
        [
            LABEL_WATER,
            LABEL_LAND,
            LABEL_LAND,
            LABEL_WATER,
            LABEL_LAND,
            LABEL_LAND,
            LABEL_LAND,
            LABEL_LAND,
            LABEL_UNKNOWN,
            LABEL_UNKNOWN,
        ]
    ]


def test_sampling_budget_and_stratification():
    rng = np.random.default_rng(0)
    valid = np.ones((100, 10), dtype=bool)
    label = np.zeros((100, 10), dtype=np.int8)
    label[:30] = 1  # 300 land, 700 water
    ep, ch = sample_reflections(valid, label, 400, rng)
    assert len(ep) == 400
    picked = label[ep, ch]
    assert (picked == 1).sum() == 200 and (picked == 0).sum() == 200
    # short class: land has only 50 -> water fills the rest
    label[:] = 0
    label[:5] = 1
    ep, ch = sample_reflections(valid, label, 400, rng)
    picked = label[ep, ch]
    assert (picked == 1).sum() == 50 and (picked == 0).sum() == 350
    # budget larger than the pool
    ep, ch = sample_reflections(valid, label, 5000, rng)
    assert len(ep) == 1000 and len(np.unique(ep * 10 + ch)) == 1000
    # nothing valid
    ep, ch = sample_reflections(np.zeros_like(valid), label, 100, rng)
    assert len(ep) == 0


def test_sampling_is_deterministic():
    valid = np.ones((50, 10), dtype=bool)
    label = (np.arange(500).reshape(50, 10) % 3 == 0).astype(np.int8)
    a = sample_reflections(valid, label, 60, np.random.default_rng(7))
    b = sample_reflections(valid, label, 60, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def _toy_flights(n=60):
    base = datetime(2023, 1, 1)
    rows = []
    routes = [("NZAA", "NZWN"), ("NZWN", "NZNS"), ("NZCH", "NZAA"), ("NZAA", "NZGS")]
    for i in range(n):
        o, d = routes[i % len(routes)]
        rows.append(
            {"flight_id": f"f{i}", "date": base + timedelta(days=10 * i), "origin": o, "dest": d}
        )
    return pd.DataFrame(rows)


def test_holdouts_and_folds():
    df = assign_holdouts(_toy_flights(), temporal_holdout_months=6)
    assert set(df["split"]) == {SPLIT_TRAIN, SPLIT_GEO, SPLIT_TIME, SPLIT_GEO_TIME}
    south = df["origin"].isin({"NZCH", "NZNS"}) | df["dest"].isin({"NZCH", "NZNS"})
    assert df.loc[south, "split"].isin({SPLIT_GEO, SPLIT_GEO_TIME}).all()
    assert (
        df.loc[df["date"] >= df["temporal_cutoff"], "split"].isin({SPLIT_TIME, SPLIT_GEO_TIME})
    ).all()
    df["land_fraction"] = np.linspace(0, 1, len(df))
    df = assign_folds(df, n_folds=3, seed=1)
    report = check_flight_splits(df, n_folds=3)
    assert report["n_flights"] == len(df)
    assert (df.loc[df["split"] != SPLIT_TRAIN, "fold_id"] == -1).all()
    assert df.loc[df["split"] == SPLIT_TRAIN, "fold_id"].between(0, 2).all()


def test_leak_checks_raise():
    df = assign_folds(assign_holdouts(_toy_flights()), n_folds=3)
    bad = pd.concat([df, df.iloc[:1]])
    with pytest.raises(LeakError):
        check_flight_splits(bad, 3)
    bad = df.copy()
    bad.loc[bad["split"] == SPLIT_GEO, "fold_id"] = 0
    with pytest.raises(LeakError):
        check_flight_splits(bad, 3)
    samples = pd.DataFrame(
        {
            "flight_id": ["f0", "f0", "f1"],
            "split": [SPLIT_TRAIN, SPLIT_TRAIN, "geo_holdout"],
            "fold_id": [0, 1, -1],
        }
    )
    with pytest.raises(LeakError):
        check_sample_splits(samples, df)


def test_drop_invalid_airports():
    df = _toy_flights(4)
    df.loc[0, "dest"] = "ZZZZ"
    kept, dropped = drop_invalid_airports(df)
    assert len(dropped) == 1 and len(kept) == 3
