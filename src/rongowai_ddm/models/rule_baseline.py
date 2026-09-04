"""Rule-based baseline from the L1 product's own coherence flag.

``coherence_state`` (1 coherent ... 4 incoherent, 5 uncertain, 0 missing) is
mapped to a land score so that it can be compared with the learned models on
the same metrics.
"""

from __future__ import annotations

import numpy as np

COHERENCE_STATE_LAND_SCORE = {1: 0.05, 2: 0.25, 3: 0.65, 4: 0.95, 5: 0.5, 0: 0.5}


def coherence_state_score(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state).astype(int)
    out = np.full(s.shape, 0.5, dtype=np.float32)
    for k, v in COHERENCE_STATE_LAND_SCORE.items():
        out[s == k] = v
    return out
