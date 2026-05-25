"""Validation harness for the Rongowai DDM classifier (Phase 0.1).

Exports the public surface area used by the experiment scripts.
"""
from .adversarial import run_adversarial_validation
from .flight_index import FlightMetadata, build_flight_index
from .geographic_holdout import (
    INVALID_ICAO_CODES,
    NORTH_ISLAND_AIRPORTS,
    SOUTH_ISLAND_AIRPORTS,
    annotate_island,
    drop_invalid_icao,
    get_unknown_airports,
)
from .group_folds import (
    SPLIT_TRAIN_CV,
    SPLIT_VAL_BOTH,
    SPLIT_VAL_GEOGRAPHIC,
    SPLIT_VAL_TEMPORAL,
    assign_holdouts,
    assign_train_cv_folds,
)
from .leak_checks import LeakError, run_all_checks

__all__ = [
    "FlightMetadata",
    "build_flight_index",
    "run_adversarial_validation",
    "INVALID_ICAO_CODES",
    "SOUTH_ISLAND_AIRPORTS",
    "NORTH_ISLAND_AIRPORTS",
    "annotate_island",
    "drop_invalid_icao",
    "get_unknown_airports",
    "SPLIT_TRAIN_CV",
    "SPLIT_VAL_GEOGRAPHIC",
    "SPLIT_VAL_TEMPORAL",
    "SPLIT_VAL_BOTH",
    "assign_holdouts",
    "assign_train_cv_folds",
    "LeakError",
    "run_all_checks",
]
