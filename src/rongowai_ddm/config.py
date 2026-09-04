"""Configuration objects loaded from a YAML file.

Every threshold that shapes the dataset lives here, so a run is fully described
by one file under ``configs/`` plus the command line of the script.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class QualityThresholds:
    """Per-DDM quality filters (see :func:`rongowai_ddm.data.quality.quality_mask`)."""

    copol_gain_min_dbi: float = 5.0
    xpol_gain_min_dbi: float = 5.0
    slant_range_min_m: float = 2000.0
    slant_range_max_m: float = 10000.0
    snr_min_db: float = 0.0
    apply_snr_filter: bool = False


@dataclass
class SamplingConfig:
    """How many reflections are kept per flight."""

    max_reflections_per_flight: int = 300
    seed: int = 42


@dataclass
class SplitConfig:
    """Flight-level holdouts and cross-validation folds."""

    temporal_holdout_months: int = 6
    n_folds: int = 5
    seed: int = 42


@dataclass
class FeatureConfig:
    peak_region_pixels: int = 40


@dataclass
class Config:
    data_dir: Path
    work_dir: Path = Path(".")
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @property
    def samples_dir(self) -> Path:
        return self.work_dir / "data" / "samples"

    @property
    def results_dir(self) -> Path:
        return self.work_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def flight_table_path(self) -> Path:
        return self.results_dir / "flights.parquet"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["work_dir"] = str(self.work_dir)
        return d

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load ``configs/default.yaml`` (or ``path``); ``RONGOWAI_DATA_DIR`` overrides the data dir."""
        if path is None:
            path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        data_dir = os.environ.get("RONGOWAI_DATA_DIR", raw.get("data_dir", ""))
        work_dir = raw.get("work_dir", ".")
        if not Path(work_dir).is_absolute():
            work_dir = (Path(path).resolve().parents[1] / work_dir).resolve()
        return cls(
            data_dir=Path(data_dir),
            work_dir=Path(work_dir),
            quality=QualityThresholds(**raw.get("quality", {})),
            sampling=SamplingConfig(**raw.get("sampling", {})),
            split=SplitConfig(**raw.get("split", {})),
            features=FeatureConfig(**raw.get("features", {})),
        )
