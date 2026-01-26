"""
Multiplicity Correction Module
==============================

Two-layer multiplicity control:
1. Across outcomes (many EMG metrics): BH-FDR
2. Within outcome (post-hoc contrasts): Holm

Architecture Note:
    This module uses pure functions and dictionaries instead of classes
    to maintain consistency with the oh_parser project style.

Functions:
- apply_fdr: Benjamini-Hochberg FDR correction
- apply_holm: Holm step-down procedure
- adjust_pvalues: General-purpose adjustment
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .lmm import LMMResult


# =============================================================================
# Core Adjustment Functions
# =============================================================================

CorrectionMethod = Literal["bonferroni", "holm", "fdr_bh", "fdr_by", "none"]


def adjust_pvalues(
    pvalues: Union[np.ndarray, List[float]],
    method: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Adjust p-values for multiple comparisons.
    
    :param pvalues: Array of raw p-values
    :param method: Correction method:
        - "bonferroni": Bonferroni (FWER control, conservative)
        - "holm": Holm step-down (FWER, less conservative)
        - "fdr_bh": Benjamini-Hochberg (FDR control, recommended default)
        - "fdr_by": Benjamini-Yekutieli (FDR under dependence)
        - "none": No adjustment
    :param alpha: Significance level (for reference, not used in adjustment)
    :returns: Array of adjusted p-values
    
    References:
        - Benjamini & Hochberg (1995): FDR control
        - Holm (1979): Step-down procedure
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    
    if n == 0:
        return np.array([])
    
    if method == "none":
        return pvalues.copy()
    
    # Handle NaN values
    nan_mask = np.isnan(pvalues)
    valid_pvalues = pvalues[~nan_mask]
    n_valid = len(valid_pvalues)
    
    if n_valid == 0:
        return pvalues.copy()
    
    if method == "bonferroni":
        adjusted_valid = np.minimum(valid_pvalues * n_valid, 1.0)
    
    elif method == "holm":
        adjusted_valid = _holm_correction(valid_pvalues)
    
    elif method == "fdr_bh":
        adjusted_valid = _bh_correction(valid_pvalues)
    
    elif method == "fdr_by":
        adjusted_valid = _by_correction(valid_pvalues)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reconstruct with NaN
    adjusted = np.full(n, np.nan)
    adjusted[~nan_mask] = adjusted_valid
    
    return adjusted


def _holm_correction(pvalues: np.ndarray) -> np.ndarray:
    """
    Holm step-down procedure for FWER control.
    
    More powerful than Bonferroni while still controlling FWER.
    """
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    
    # Holm adjustment: p_adj[i] = max(p[j] * (n - j)) for j <= i
    adjusted_sorted = np.zeros(n)
    cummax = 0
    
    for i in range(n):
        adj = sorted_pvals[i] * (n - i)
        cummax = max(cummax, adj)
        adjusted_sorted[i] = min(cummax, 1.0)
    
    # Restore original order
    adjusted = np.zeros(n)
    adjusted[sorted_idx] = adjusted_sorted
    
    return adjusted


def _bh_correction(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg procedure for FDR control.
    
    Controls the expected proportion of false discoveries.
    """
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    
    # BH adjustment: p_adj[i] = min(p[j] * n / (j+1)) for j >= i
    adjusted_sorted = np.zeros(n)
    cummin = 1.0
    
    for i in range(n - 1, -1, -1):
        adj = sorted_pvals[i] * n / (i + 1)
        cummin = min(cummin, adj)
        adjusted_sorted[i] = min(cummin, 1.0)
    
    # Restore original order
    adjusted = np.zeros(n)
    adjusted[sorted_idx] = adjusted_sorted
    
    return adjusted


def _by_correction(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Yekutieli procedure for FDR under dependence.
    
    More conservative than BH, but valid under arbitrary dependence.
    """
    n = len(pvalues)
    
    # Correction factor: sum(1/i) for i=1 to n
    c_n = np.sum(1.0 / np.arange(1, n + 1))
    
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    
    # BY adjustment: same as BH but multiply by c(n)
    adjusted_sorted = np.zeros(n)
    cummin = 1.0
    
    for i in range(n - 1, -1, -1):
        adj = sorted_pvals[i] * n * c_n / (i + 1)
        cummin = min(cummin, adj)
        adjusted_sorted[i] = min(cummin, 1.0)
    
    # Restore original order
    adjusted = np.zeros(n)
    adjusted[sorted_idx] = adjusted_sorted
    
    return adjusted


# =============================================================================
# Outcome-level FDR (across multiple outcomes)
# =============================================================================

def apply_fdr(
    results: Union[List[LMMResult], Dict[str, LMMResult]],
    term: str = "day_index",
    method: CorrectionMethod = "fdr_bh",
    alpha: float = 0.05,
    use_lrt: bool = True,
) -> pd.DataFrame:
    """
    Apply FDR correction across multiple LMM outcomes.
    
    Extracts the p-value for a specific term from each model and
    applies multiplicity correction across outcomes.
    
    :param results: List or dict of LMMResult dictionaries
    :param term: Model term to extract p-value for (partial match)
    :param method: Correction method
    :param alpha: Significance threshold
    :returns: DataFrame with outcome, p_raw, p_adjusted, significant
    
    Example:
        >>> results = fit_all_outcomes(ds)
        >>> fdr_results = apply_fdr(results, term="day_index")
        >>> significant = fdr_results[fdr_results["significant"]]
    """
    if isinstance(results, dict):
        results_list = list(results.values())
    else:
        results_list = results
    
    rows = []
    
    for result in results_list:
        if result["coefficients"].empty:
            continue

        if use_lrt:
            lrt_p = result.get("fit_stats", {}).get("lrt_pvalue", np.nan)
            if not np.isnan(lrt_p):
                rows.append({
                    "outcome": result["outcome"],
                    "term": "lrt",
                    "p_raw": lrt_p,
                })
                continue
        
        # Find the term (partial match for categorical encoding)
        term_mask = result["coefficients"]["term"].str.contains(term, case=False, na=False)
        term_rows = result["coefficients"][term_mask]
        
        if term_rows.empty:
            # Term not found - might be reference level
            rows.append({
                "outcome": result["outcome"],
                "term": term,
                "estimate": np.nan,
                "p_raw": np.nan,
                "note": "Term not in model (reference level?)",
            })
            continue
        
        # For categorical factors, take the most significant p-value
        # (representing the overall factor effect)
        min_p_idx = term_rows["p_value"].idxmin()
        best_row = term_rows.loc[min_p_idx]
        
        rows.append({
            "outcome": result["outcome"],
            "term": best_row["term"],
            "estimate": best_row["estimate"],
            "std_error": best_row["std_error"],
            "z_value": best_row["z_value"],
            "p_raw": best_row["p_value"],
            "converged": result["converged"],
        })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Apply FDR correction
    df["p_adjusted"] = adjust_pvalues(df["p_raw"].values, method=method)
    df["significant"] = df["p_adjusted"] < alpha
    df["correction"] = method
    
    # Sort by adjusted p-value
    df = df.sort_values("p_adjusted").reset_index(drop=True)
    
    return df


def apply_holm_hypotheses(
    hypotheses: Union[List[Dict[str, float]], Dict[str, Dict[str, float]]],
    pvalue_key: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Holm correction across hypothesis-level p-values.
    """
    if isinstance(hypotheses, dict):
        rows = [
            {"hypothesis": k, "p_raw": v.get(pvalue_key, np.nan)}
            for k, v in hypotheses.items()
        ]
    else:
        rows = []
        for i, h in enumerate(hypotheses, start=1):
            rows.append({
                "hypothesis": h.get("id", f"H{i}"),
                "p_raw": h.get(pvalue_key, np.nan),
            })

    df = pd.DataFrame(rows)
    df["p_adjusted"] = adjust_pvalues(df["p_raw"].values, method="holm", alpha=alpha)
    df["significant"] = df["p_adjusted"] < alpha
    return df


def apply_holm(
    results: Union[List[LMMResult], Dict[str, LMMResult]],
    term: str = "day_index",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Holm correction across outcomes (FWER control).
    
    More conservative than FDR; use for primary outcomes or when
    FWER control is required.
    
    :param results: List or dict of LMMResult dictionaries
    :param term: Model term to extract p-value for
    :param alpha: Significance threshold
    :returns: DataFrame with outcome, p_raw, p_adjusted, significant
    """
    return apply_fdr(results, term=term, method="holm", alpha=alpha)


# =============================================================================
# Summary Functions
# =============================================================================

def significant_outcomes(
    fdr_results: pd.DataFrame,
    alpha: float = 0.05,
) -> List[str]:
    """
    Extract names of significant outcomes after FDR correction.
    
    :param fdr_results: Output from apply_fdr()
    :param alpha: Significance threshold
    :returns: List of significant outcome names
    """
    if "significant" in fdr_results.columns:
        return fdr_results[fdr_results["significant"]]["outcome"].tolist()
    elif "p_adjusted" in fdr_results.columns:
        return fdr_results[fdr_results["p_adjusted"] < alpha]["outcome"].tolist()
    else:
        return []


def fdr_summary(fdr_results: pd.DataFrame) -> str:
    """
    Generate a human-readable summary of FDR results.
    
    :param fdr_results: Output from apply_fdr()
    :returns: Human-readable summary string
    """
    n_tested = len(fdr_results)
    n_sig = fdr_results["significant"].sum() if "significant" in fdr_results.columns else 0
    method = fdr_results["correction"].iloc[0] if "correction" in fdr_results.columns else "unknown"
    
    lines = [
        f"FDR Correction Summary ({method})",
        f"  Outcomes tested: {n_tested}",
        f"  Significant (q<0.05): {n_sig}",
        f"  Discovery rate: {100 * n_sig / n_tested:.1f}%" if n_tested > 0 else "",
    ]
    
    if n_sig > 0:
        sig_outcomes = significant_outcomes(fdr_results)
        lines.append("\n  Significant outcomes:")
        for outcome in sig_outcomes[:10]:
            p_adj = fdr_results[fdr_results["outcome"] == outcome]["p_adjusted"].values[0]
            lines.append(f"    - {outcome} (q={p_adj:.4f})")
        if len(sig_outcomes) > 10:
            lines.append(f"    ... and {len(sig_outcomes) - 10} more")
    
    return "\n".join(lines)
