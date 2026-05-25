"""Data layer for the Rongowai rework pipeline.

Vendored implementations of the production ETL classes (originally defined
inside the notebooks under ``deliver/``), refactored as importable Python
modules so the new experiment scripts can compose them without touching the
existing notebook code.
"""
from .encoder import Encoder, Decoder, load_encoder, compress_array
from .feature_extractor import DDMFeatureExtractor, STAT_FEATURE_NAMES
from .preprocessor import NetCDFFlightPreprocessor, QUALITY_THRESHOLDS

__all__ = [
    "Encoder",
    "Decoder",
    "load_encoder",
    "compress_array",
    "DDMFeatureExtractor",
    "STAT_FEATURE_NAMES",
    "NetCDFFlightPreprocessor",
    "QUALITY_THRESHOLDS",
]
