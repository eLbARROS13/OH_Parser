"""
OH Stats Hypothesis Testing Module
==================================

Structured hypothesis testing for the Occupational Health study.

Each hypothesis (H1-H6) has its own documented module with:
- Detailed scientific rationale
- Model specification
- Expected outcomes

Usage:
    from hypotheses import run_all, run_hypothesis, HYPOTHESES
    
    # Run all hypotheses
    results = run_all(profiles)
    
    # Run single hypothesis
    h1_result = run_hypothesis("H1", profiles)
"""

from .config import HYPOTHESES, HypothesisConfig
from .runner import (
    run_hypothesis,
    run_all,
    summarize_results,
    apply_multiplicity_correction,
    # Within-between decomposition
    decompose_within_between,
    get_within_between_variables,
)

__all__ = [
    "HYPOTHESES",
    "HypothesisConfig",
    "run_hypothesis",
    "run_all",
    "summarize_results",
    "apply_multiplicity_correction",
    # Within-between decomposition
    "decompose_within_between",
    "get_within_between_variables",
]
