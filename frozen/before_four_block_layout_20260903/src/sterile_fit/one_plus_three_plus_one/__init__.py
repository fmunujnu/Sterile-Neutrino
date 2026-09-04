"""Parallel 1+3+1 physics and fitting layer.

This package intentionally does not alter the validated 3+1 implementation.
It reuses the same experiment templates, covariance prescription and CLs
utilities through a model-compatible probability interface.
"""

from .analysis import (
    BuiltOnePlusThreePlusOneAnalysis,
    BuiltOnePlusThreePlusOneExperiment,
    build_one_plus_three_plus_one_analysis,
)
from .fitting import (
    OnePlusThreePlusOneFitPoint,
    OnePlusThreePlusOneProfileResult,
    prefit_one_plus_three_plus_one,
    profile_at_fixed_mass_pair,
    profile_one_plus_three_plus_one,
)
from .model import OnePlusThreePlusOneVacuumModel
from .parameters import OnePlusThreePlusOneParameters

__all__ = [
    "BuiltOnePlusThreePlusOneAnalysis",
    "BuiltOnePlusThreePlusOneExperiment",
    "OnePlusThreePlusOneFitPoint",
    "OnePlusThreePlusOneParameters",
    "OnePlusThreePlusOneProfileResult",
    "OnePlusThreePlusOneVacuumModel",
    "build_one_plus_three_plus_one_analysis",
    "prefit_one_plus_three_plus_one",
    "profile_at_fixed_mass_pair",
    "profile_one_plus_three_plus_one",
]
