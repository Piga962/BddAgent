"""Analysis module for BddAgent experiments."""

from .statistical import (
    compute_confidence_interval,
    compute_effect_size,
    compare_models,
    StatisticalSummary,
    ModelComparison,
    ExperimentAnalyzer
)

__all__ = [
    'compute_confidence_interval',
    'compute_effect_size',
    'compare_models',
    'StatisticalSummary',
    'ModelComparison',
    'ExperimentAnalyzer'
]
