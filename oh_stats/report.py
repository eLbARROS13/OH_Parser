"""
Report Generation Module
========================

Auto-generate summary tables for statistical reports:
- Table 1: Descriptive statistics
- Coefficient tables: Model results
- Results summary: FDR-corrected findings

Architecture Note:
    This module uses pure functions and dictionaries instead of classes
    to maintain consistency with the oh_parser project style.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from .lmm import LMMResult
from .prepare import AnalysisDataset
from .descriptive import summarize_outcomes


# =============================================================================
# Table 1: Descriptive Statistics
# =============================================================================

def descriptive_table(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    by_group: Optional[str] = None,
    format_spec: str = ".2f",
    include_n: bool = True,
) -> pd.DataFrame:
    """
    Generate a "Table 1" style descriptive statistics table.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes (None = all)
    :param by_group: Optional grouping variable (e.g., "side")
    :param format_spec: Format string for numeric values (e.g., ".2f", ".3f")
    :param include_n: Include sample size
    :returns: Formatted DataFrame suitable for publication
    
    Example output:
        |  Outcome                  |  Mean (SD)      |  Median [IQR]     |
        |---------------------------|-----------------|-------------------|
        |  EMG_intensity.mean_mvc   |  15.2 (8.3)     |  13.1 [9.2-19.4]  |
    """
    outcomes = outcomes or ds["outcome_vars"]
    
    summary = summarize_outcomes(ds, outcomes, by_group=by_group is not None)
    
    rows = []
    
    # Use format_spec directly (e.g., ".2f")
    fmt = format_spec
    
    for _, row in summary.iterrows():
        outcome = row["outcome"]
        n = row.get("n", 0)
        
        # Format mean (SD)
        mean_sd = f"{row['mean']:{fmt}} ({row['std']:{fmt}})"
        
        # Format median [IQR]
        median_iqr = f"{row['median']:{fmt}} [{row['p25']:{fmt}}-{row['p75']:{fmt}}]"
        
        # Format range
        range_str = f"{row['min']:{fmt}} - {row['max']:{fmt}}"
        
        table_row = {
            "Outcome": outcome,
            "N": int(n),
            "Mean (SD)": mean_sd,
            "Median [IQR]": median_iqr,
            "Range": range_str,
        }
        
        if by_group and by_group in row:
            table_row["Group"] = row[by_group]
        
        rows.append(table_row)
    
    df = pd.DataFrame(rows)
    
    if not include_n:
        df = df.drop(columns=["N"])
    
    return df


def descriptive_table_formatted(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    style: str = "mean_sd",
) -> pd.DataFrame:
    """
    Generate a compact descriptive table with single summary column.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Specific outcomes
    :param style: Summary style - "mean_sd", "median_iqr", or "both"
    :returns: Compact formatted table
    """
    summary = summarize_outcomes(ds, outcomes)
    
    rows = []
    for _, row in summary.iterrows():
        table_row = {"Outcome": row["outcome"]}
        
        if style == "mean_sd":
            table_row["Summary"] = f"{row['mean']:.2f} ± {row['std']:.2f}"
        elif style == "median_iqr":
            table_row["Summary"] = f"{row['median']:.2f} ({row['p25']:.2f}-{row['p75']:.2f})"
        else:
            table_row["Mean ± SD"] = f"{row['mean']:.2f} ± {row['std']:.2f}"
            table_row["Median (IQR)"] = f"{row['median']:.2f} ({row['p25']:.2f}-{row['p75']:.2f})"
        
        table_row["N"] = int(row["n"])
        rows.append(table_row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Coefficient Tables
# =============================================================================

def coefficient_table(
    result: LMMResult,
    format_estimate: str = "{:.3f}",
    format_ci: str = "{:.3f}",
    format_p: str = "{:.4f}",
) -> pd.DataFrame:
    """
    Format coefficient table for a single LMM result.
    
    :param result: Fitted LMMResult dictionary
    :param format_estimate: Format for estimates
    :param format_ci: Format for confidence intervals
    :param format_p: Format for p-values
    :returns: Formatted coefficient table
    """
    if result["coefficients"].empty:
        return pd.DataFrame({"Note": ["Model not fitted or no coefficients"]})
    
    df = result["coefficients"].copy()
    
    # Format estimate with CI
    df["Estimate (95% CI)"] = df.apply(
        lambda r: (
            f"{format_estimate.format(r['estimate'])} "
            f"({format_ci.format(r['ci_lower'])}, {format_ci.format(r['ci_upper'])})"
        ),
        axis=1,
    )
    
    # Format p-value with significance stars
    def format_pval(p):
        if np.isnan(p):
            return "NA"
        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        return f"{format_p.format(p)}{stars}"
    
    df["P-value"] = df["p_value"].apply(format_pval)
    
    # Select and rename columns
    output = df[["term", "Estimate (95% CI)", "std_error", "z_value", "P-value"]].copy()
    output = output.rename(columns={
        "term": "Term",
        "std_error": "SE",
        "z_value": "Z",
    })
    
    return output


def coefficient_table_multiple(
    results: Union[List[LMMResult], Dict[str, LMMResult]],
    term_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate coefficient table across multiple outcomes.
    
    :param results: List or dict of LMMResult dictionaries
    :param term_filter: Only include terms matching this pattern
    :returns: Combined coefficient table
    """
    if isinstance(results, dict):
        results_list = list(results.values())
    else:
        results_list = results
    
    rows = []
    
    for result in results_list:
        if result["coefficients"].empty:
            continue
        
        coef_df = result["coefficients"]
        
        if term_filter:
            coef_df = coef_df[coef_df["term"].str.contains(term_filter, na=False)]
        
        for _, row in coef_df.iterrows():
            rows.append({
                "Outcome": result["outcome"],
                "Term": row["term"],
                "Estimate": row["estimate"],
                "SE": row["std_error"],
                "95% CI": f"({row['ci_lower']:.3f}, {row['ci_upper']:.3f})",
                "P-value": row["p_value"],
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# Results Summary
# =============================================================================

def results_summary(
    results: Union[List[LMMResult], Dict[str, LMMResult]],
    fdr_results: Optional[pd.DataFrame] = None,
    include_fit_stats: bool = True,
) -> pd.DataFrame:
    """
    Generate summary table of all model results.
    
    :param results: List or dict of LMMResult dictionaries
    :param fdr_results: Optional FDR correction results from apply_fdr()
    :param include_fit_stats: Include AIC, BIC, ICC
    :returns: Summary DataFrame
    """
    if isinstance(results, dict):
        results_list = list(results.values())
    else:
        results_list = results
    
    rows = []
    
    for result in results_list:
        row = {
            "Outcome": result["outcome"],
            "N_obs": result["n_obs"],
            "N_subjects": result["n_groups"],
            "Converged": result["converged"],
        }
        
        if include_fit_stats:
            row["AIC"] = result["fit_stats"].get("aic", np.nan)
            row["BIC"] = result["fit_stats"].get("bic", np.nan)
            row["ICC"] = result["random_effects"].get("icc", np.nan)
        
        if result["warnings"]:
            row["Warnings"] = len(result["warnings"])
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Merge FDR results if provided
    if fdr_results is not None and not fdr_results.empty:
        fdr_subset = fdr_results[["outcome", "p_raw", "p_adjusted", "significant"]].copy()
        fdr_subset = fdr_subset.rename(columns={"outcome": "Outcome"})
        df = df.merge(fdr_subset, on="Outcome", how="left")
    
    return df


# =============================================================================
# Export Functions
# =============================================================================

def export_to_csv(
    table: pd.DataFrame,
    filepath: str,
    **kwargs,
) -> None:
    """Export table to CSV file."""
    table.to_csv(filepath, index=False, **kwargs)


def export_to_latex(
    table: pd.DataFrame,
    filepath: Optional[str] = None,
    caption: str = "",
    label: str = "",
) -> str:
    """
    Export table to LaTeX format.
    
    :param table: DataFrame to export
    :param filepath: Optional file path to write
    :param caption: Table caption
    :param label: LaTeX label
    :returns: LaTeX string
    """
    latex = table.to_latex(
        index=False,
        caption=caption if caption else None,
        label=label if label else None,
        escape=True,
    )
    
    if filepath:
        with open(filepath, "w") as f:
            f.write(latex)
    
    return latex


# =============================================================================
# Print Functions
# =============================================================================

def print_results_summary(
    results: Union[List[LMMResult], Dict[str, LMMResult]],
    fdr_results: Optional[pd.DataFrame] = None,
) -> None:
    """Print a human-readable results summary."""
    summary = results_summary(results, fdr_results)
    
    print("=" * 70)
    print("LMM RESULTS SUMMARY")
    print("=" * 70)
    
    n_converged = summary["Converged"].sum()
    n_total = len(summary)
    
    print(f"\nModels fitted: {n_total}")
    print(f"Converged: {n_converged} ({100*n_converged/n_total:.0f}%)")
    
    if "significant" in summary.columns:
        n_sig = summary["significant"].sum()
        print(f"Significant (FDR q<0.05): {n_sig}")
    
    print("\n" + "-" * 70)
    print(summary.to_string(index=False))
    print("-" * 70)


def print_coefficient_summary(result: LMMResult) -> None:
    """Print coefficient summary for a single model."""
    print(f"\n{'=' * 60}")
    print(f"MODEL: {result['outcome']}")
    print(f"{'=' * 60}")
    print(f"Formula: {result['formula']}")
    print(f"N obs: {result['n_obs']}, N subjects: {result['n_groups']}")
    print(f"Converged: {result['converged']}")
    print(f"AIC: {result['fit_stats'].get('aic', np.nan):.1f}, "
          f"BIC: {result['fit_stats'].get('bic', np.nan):.1f}")
    print(f"ICC: {result['random_effects'].get('icc', np.nan):.3f}")
    
    print("\nCoefficients:")
    print("-" * 60)
    
    table = coefficient_table(result)
    print(table.to_string(index=False))
    
    if result["warnings"]:
        print(f"\nWarnings: {result['warnings']}")
