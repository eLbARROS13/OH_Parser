"""
Descriptive Statistics Module
=============================

Summary statistics, normality assessment, variance checks, and
missingness reporting for AnalysisDataset dictionaries.

Architecture Note:
    This module uses dictionaries instead of classes for data structures
    to maintain consistency with the oh_parser project style.

Key functions:
- summarize_outcomes: Per-outcome descriptive statistics
- check_normality: Shapiro-Wilk and skewness assessment
- check_variance: Detect near-zero variance (degenerate) outcomes
- missingness_report: Patterns of missing data
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .prepare import AnalysisDataset
from .registry import get_outcome_info, OutcomeType, TransformType


# =============================================================================
# Summary Statistics
# =============================================================================

def summarize_outcomes(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    by_group: bool = False,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each outcome.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes to summarize (None = all)
    :param by_group: If True, compute stats within each grouping variable
    :returns: DataFrame with summary statistics
    
    Statistics computed:
    - n: Number of non-missing observations
    - n_missing: Number of missing values
    - mean, std: Mean and standard deviation
    - min, p25, median, p75, max: Percentiles
    - skewness, kurtosis: Shape statistics
    - cv: Coefficient of variation (std/mean)
    """
    outcomes = outcomes or ds["outcome_vars"]
    outcomes = [o for o in outcomes if o in ds["data"].columns]
    
    if not outcomes:
        return pd.DataFrame()
    
    if by_group and ds["grouping_vars"]:
        # Compute within groups
        results = []
        for group_vals, group_df in ds["data"].groupby(ds["grouping_vars"]):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            
            group_stats = _compute_stats(group_df, outcomes)
            
            # Add group identifiers
            for var, val in zip(ds["grouping_vars"], group_vals):
                group_stats[var] = val
            
            results.append(group_stats)
        
        df = pd.concat(results, ignore_index=True)
        # Reorder columns
        cols = ds["grouping_vars"] + [c for c in df.columns if c not in ds["grouping_vars"]]
        return df[cols]
    
    else:
        return _compute_stats(ds["data"], outcomes)


def _compute_stats(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    """Compute summary statistics for a DataFrame."""
    rows = []
    
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        
        values = df[outcome].dropna()
        n = len(values)
        n_total = len(df[outcome])
        n_missing = n_total - n
        
        if n == 0:
            row = {
                "outcome": outcome,
                "n": 0,
                "n_missing": n_missing,
                "pct_missing": 100.0,
            }
        else:
            row = {
                "outcome": outcome,
                "n": n,
                "n_missing": n_missing,
                "pct_missing": 100.0 * n_missing / n_total if n_total > 0 else 0,
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "max": values.max(),
                "skewness": stats.skew(values) if n >= 3 else np.nan,
                "kurtosis": stats.kurtosis(values) if n >= 4 else np.nan,
                "cv": values.std() / values.mean() if values.mean() != 0 else np.nan,
            }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Normality Assessment
# =============================================================================

NormalityResult = Dict[str, Any]


def check_normality(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Assess normality of outcome distributions.
    
    Uses Shapiro-Wilk test and skewness/kurtosis to evaluate normality
    and recommend transformations.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes to check (None = all)
    :param alpha: Significance level for Shapiro-Wilk test
    :returns: DataFrame with normality assessment
    
    Note: Shapiro-Wilk is conservative with large samples; always
    examine skewness and visual diagnostics alongside p-values.
    """
    outcomes = outcomes or ds["outcome_vars"]
    outcomes = [o for o in outcomes if o in ds["data"].columns]
    
    results = []
    
    for outcome in outcomes:
        values = ds["data"][outcome].dropna()
        n = len(values)
        
        if n < 3:
            results.append({
                "outcome": outcome,
                "n": n,
                "shapiro_stat": np.nan,
                "shapiro_p": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "is_normal": None,
                "recommended_transform": "insufficient_data",
                "notes": "Too few observations for normality test",
            })
            continue
        
        # Shapiro-Wilk (limited to 5000 samples due to scipy constraint)
        truncated_for_shapiro = n > 5000
        test_values = values.values[:5000] if truncated_for_shapiro else values.values
        try:
            shapiro_stat, shapiro_p = stats.shapiro(test_values)
            if truncated_for_shapiro:
                warnings.warn(
                    f"Shapiro-Wilk test for '{outcome}' used first 5000 of {n} samples. "
                    f"Consider using visual diagnostics or Jarque-Bera for large samples."
                )
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        skew = stats.skew(values)
        kurt = stats.kurtosis(values)
        
        is_normal = shapiro_p > alpha if not np.isnan(shapiro_p) else None
        
        # Determine recommended transform based on registry and distribution
        info = get_outcome_info(outcome)
        registry_transform = info["transform"]
        
        # Override based on actual distribution
        if abs(skew) > 2:
            if (values > 0).all():
                recommended = TransformType.LOG
            else:
                recommended = TransformType.LOG1P if (values >= 0).all() else TransformType.NONE
            notes = f"High skewness ({skew:.2f}); log transform recommended"
        elif abs(skew) > 1:
            if (values > 0).all():
                recommended = TransformType.SQRT
            else:
                recommended = TransformType.NONE
            notes = f"Moderate skewness ({skew:.2f}); sqrt may help"
        elif info["outcome_type"] == OutcomeType.PROPORTION:
            recommended = TransformType.LOGIT
            notes = "Proportion outcome; logit transform typical"
        else:
            recommended = registry_transform
            notes = "Distribution appears approximately normal" if is_normal else "Non-normal but mild"
        
        results.append({
            "outcome": outcome,
            "n": n,
            "shapiro_stat": shapiro_stat,
            "shapiro_p": shapiro_p,
            "skewness": skew,
            "kurtosis": kurt,
            "is_normal": is_normal,
            "recommended_transform": str(recommended),
            "notes": notes,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Variance Checks (Degenerate Outcome Detection)
# =============================================================================

VarianceCheckResult = Dict[str, Any]


def check_variance(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    min_unique: int = 3,
    max_mode_pct: float = 95.0,
    min_cv: float = 0.01,
) -> pd.DataFrame:
    """
    Detect near-zero variance (degenerate) outcomes.
    
    Outcomes are flagged as degenerate if:
    - Fewer than min_unique unique values
    - More than max_mode_pct of values at a single value
    - Coefficient of variation below min_cv
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes to check (None = all)
    :param min_unique: Minimum unique values required
    :param max_mode_pct: Maximum percentage at mode before flagging
    :param min_cv: Minimum coefficient of variation
    :returns: DataFrame with variance assessment
    """
    outcomes = outcomes or ds["outcome_vars"]
    outcomes = [o for o in outcomes if o in ds["data"].columns]
    
    results = []
    
    for outcome in outcomes:
        values = ds["data"][outcome].dropna()
        n = len(values)
        
        if n == 0:
            results.append({
                "outcome": outcome,
                "n": 0,
                "n_unique": 0,
                "std": np.nan,
                "cv": np.nan,
                "pct_mode": np.nan,
                "is_degenerate": True,
                "reason": "No non-missing values",
            })
            continue
        
        n_unique = values.nunique()
        std = values.std()
        mean = values.mean()
        
        # Calculate CV carefully:
        # - If mean is 0 but std > 0, data is centered at zero with variance (potentially valid)
        # - If both mean and std are ~0, it's degenerate
        if mean == 0:
            if std == 0:
                cv = 0.0  # Constant at zero - degenerate
            else:
                cv = np.inf  # Centered at zero with variance - flag for review
        else:
            cv = std / abs(mean)
        
        # Calculate percentage at mode
        mode_count = values.value_counts().iloc[0] if n_unique > 0 else n
        pct_mode = 100.0 * mode_count / n
        
        # Determine if degenerate
        is_degenerate = False
        reasons = []
        
        if n_unique < min_unique:
            is_degenerate = True
            reasons.append(f"Only {n_unique} unique values")
        
        if pct_mode > max_mode_pct:
            is_degenerate = True
            reasons.append(f"{pct_mode:.1f}% at single value")
        
        if cv < min_cv and mean != 0:
            is_degenerate = True
            reasons.append(f"Very low CV ({cv:.4f})")
        
        results.append({
            "outcome": outcome,
            "n": n,
            "n_unique": n_unique,
            "std": std,
            "cv": cv,
            "pct_mode": pct_mode,
            "is_degenerate": is_degenerate,
            "reason": "; ".join(reasons) if reasons else "OK",
        })
    
    return pd.DataFrame(results)


def get_non_degenerate_outcomes(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """
    Return list of outcomes that are not degenerate.
    
    Convenience wrapper around check_variance.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes to check
    :param kwargs: Additional arguments to check_variance
    :returns: List of non-degenerate outcome names
    """
    var_check = check_variance(ds, outcomes, **kwargs)
    non_degen = var_check[~var_check["is_degenerate"]]["outcome"].tolist()
    
    n_degen = len(var_check) - len(non_degen)
    if n_degen > 0:
        warnings.warn(
            f"Excluded {n_degen} degenerate outcomes with near-zero variance. "
            f"Use check_variance() for details."
        )
    
    return non_degen


# =============================================================================
# Missingness Reporting
# =============================================================================

def missingness_report(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive missingness report.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes to check (None = all)
    :returns: Dictionary with missingness statistics and patterns
    """
    outcomes = outcomes or ds["outcome_vars"]
    outcomes = [o for o in outcomes if o in ds["data"].columns]
    
    df = ds["data"]
    
    # Overall missingness by outcome
    outcome_missing = pd.DataFrame({
        "outcome": outcomes,
        "n_missing": [df[o].isna().sum() for o in outcomes],
        "n_total": [len(df) for _ in outcomes],
        "pct_missing": [100.0 * df[o].isna().sum() / len(df) for o in outcomes],
    })
    
    # Missingness by subject
    from .prepare import get_obs_per_subject
    
    subject_missing = df.groupby(ds["id_var"])[outcomes].apply(
        lambda x: x.isna().sum().sum()
    ).rename("total_missing")
    
    # Observations per subject
    obs_per_subject = get_obs_per_subject(ds)
    
    # Pattern: which outcomes tend to be missing together
    missing_matrix = df[outcomes].isna()
    missing_corr = missing_matrix.corr() if len(outcomes) > 1 else pd.DataFrame()
    
    return {
        "by_outcome": outcome_missing,
        "by_subject": pd.DataFrame({
            "subject_id": subject_missing.index,
            "total_missing": subject_missing.values,
            "n_observations": obs_per_subject.values,
        }),
        "missing_correlation": missing_corr,
        "summary": {
            "total_cells": len(df) * len(outcomes),
            "total_missing": outcome_missing["n_missing"].sum(),
            "pct_missing": outcome_missing["pct_missing"].mean(),
            "outcomes_with_missing": (outcome_missing["n_missing"] > 0).sum(),
            "subjects_with_missing": (subject_missing > 0).sum(),
        },
    }


def print_missingness_summary(ds: AnalysisDataset) -> None:
    """Print a human-readable missingness summary."""
    report = missingness_report(ds)
    summary = report["summary"]
    
    print("=" * 50)
    print("MISSINGNESS SUMMARY")
    print("=" * 50)
    print(f"Total cells: {summary['total_cells']}")
    print(f"Total missing: {summary['total_missing']} ({summary['pct_missing']:.1f}%)")
    print(f"Outcomes with missing: {summary['outcomes_with_missing']} / {len(ds['outcome_vars'])}")
    from .prepare import get_n_subjects
    print(f"Subjects with missing: {summary['subjects_with_missing']} / {get_n_subjects(ds)}")
    
    # Show top missing outcomes
    by_outcome = report["by_outcome"]
    by_outcome = by_outcome[by_outcome["n_missing"] > 0].sort_values("pct_missing", ascending=False)
    
    if len(by_outcome) > 0:
        print("\nMost missing outcomes:")
        for _, row in by_outcome.head(5).iterrows():
            print(f"  {row['outcome']}: {row['n_missing']} ({row['pct_missing']:.1f}%)")
