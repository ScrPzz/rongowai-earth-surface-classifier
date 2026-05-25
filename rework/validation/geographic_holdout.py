"""Geographic holdout: partition NZ airports into North/South Island groups.

A flight is flagged for the geographic holdout if EITHER its origin or its
destination is in :data:`SOUTH_ISLAND_AIRPORTS`. This is intentionally
conservative — a flight that briefly crosses from one island to the other
captures DDMs over heterogeneous geography, which the classifier should not
be allowed to see during training.

Airport codes were curated from the ICAO list and cross-checked against the
filename catalogue of the Rongowai dataset (December 2022 — March 2025).
"""
from __future__ import annotations

import pandas as pd

# Curated ICAO airport codes for New Zealand. Codes are grouped by island.
# When the indexer encounters a code not in either set, it is flagged as
# "unknown" and surfaced to the user; the geographic-holdout assignment treats
# unknown codes as NOT touching the South Island (conservative — they stay in
# the train pool until manually classified).

SOUTH_ISLAND_AIRPORTS = frozenset(
    {
        "NZCH",  # Christchurch
        "NZNS",  # Nelson
        "NZHK",  # Hokitika
        "NZWB",  # Woodbourne (Blenheim)
        "NZDN",  # Dunedin
        "NZQN",  # Queenstown
        "NZIR",  # Invercargill
        "NZGT",  # Greymouth (commercial)
        "NZTU",  # Timaru
        "NZWF",  # Wanaka
        "NZNV",  # Invercargill (alt)
        "NZMK",  # Motueka
    }
)

# ICAO codes that indicate invalid / placeholder / unknown destinations.
# ZZZZ is the standard ICAO code for "unknown or unscheduled aerodrome" and is
# used in Rongowai filenames when origin or destination metadata is missing
# (likely aborted flights, diversions, or test runs). Such flights are excluded
# from the entire dataset by default; see drop_invalid_icao().
INVALID_ICAO_CODES = frozenset({"ZZZZ"})


NORTH_ISLAND_AIRPORTS = frozenset(
    {
        "NZAA",  # Auckland International
        "NZNR",  # Napier
        "NZGS",  # Gisborne
        "NZAP",  # Taupo
        "NZRO",  # Rotorua
        "NZTG",  # Tauranga
        "NZHN",  # Hamilton
        "NZNP",  # New Plymouth
        "NZWN",  # Wellington
        "NZPP",  # Paraparaumu
        "NZPM",  # Palmerston North
        "NZWP",  # Whenuapai (RNZAF / Auckland)
        "NZWR",  # Whangarei
        "NZKK",  # Kerikeri
        "NZTO",  # Tokoroa
        "NZKT",  # Kaitaia
    }
)


def _island_of(code: str) -> str:
    if code in SOUTH_ISLAND_AIRPORTS:
        return "south"
    if code in NORTH_ISLAND_AIRPORTS:
        return "north"
    return "unknown"


def annotate_island(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'origin_island', 'dest_island', 'touches_south_island' columns."""
    df = df.copy()
    df["origin_island"] = df["origin"].map(_island_of)
    df["dest_island"] = df["dest"].map(_island_of)
    df["touches_south_island"] = (
        df["origin"].isin(SOUTH_ISLAND_AIRPORTS)
        | df["dest"].isin(SOUTH_ISLAND_AIRPORTS)
    )
    return df


def get_unknown_airports(df: pd.DataFrame) -> set[str]:
    """Return airport codes present in the data but absent from either island set.

    Note: this returns unknown codes regardless of whether they are also in
    :data:`INVALID_ICAO_CODES`. Use :func:`drop_invalid_icao` to actually
    remove flights with invalid codes from the index.
    """
    all_codes = set(df["origin"]).union(set(df["dest"]))
    known = SOUTH_ISLAND_AIRPORTS | NORTH_ISLAND_AIRPORTS
    return all_codes - known


def drop_invalid_icao(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove flights whose origin or destination is in :data:`INVALID_ICAO_CODES`.

    Returns ``(kept, dropped)`` so the caller can report on what was excluded.
    """
    invalid_mask = (
        df["origin"].isin(INVALID_ICAO_CODES)
        | df["dest"].isin(INVALID_ICAO_CODES)
    )
    return df[~invalid_mask].reset_index(drop=True), df[invalid_mask].reset_index(drop=True)
