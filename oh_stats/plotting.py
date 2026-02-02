"""
LMM Visualization Module
========================

Provides high-level plotting functions for visualizing Linear Mixed Model
results on longitudinal sensor data. Designed to help understand the
"spaghetti" of individual trajectories vs population-level trends.

Key visualizations:
- plot_lmm_trajectories: Raw data spaghetti plot with population means
- plot_lmm_fit: Actual LMM fitted values (marginal + conditional predictions)
- plot_random_intercepts: Random effects distribution (who you are matters)
- plot_model_diagnostics: QQ plot + residuals vs fitted

Architecture Note:
    This module follows the oh_stats convention of using dictionaries
    rather than classes. It accepts LMMResult dicts from
    the lmm module.

Example:
    >>> from oh_stats import fit_lmm, prepare_daily_emg
    >>> from oh_stats.plotting import plot_lmm_fit, plot_random_intercepts
    >>> 
    >>> ds = prepare_daily_emg(profiles, side="both")
    >>> result = fit_lmm(ds, "EMG_apdf.active.p90", fixed_effects=["C(work_type)"])
    >>> plot_lmm_fit(ds, result, group="work_type")
    >>> plot_random_intercepts(result)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if TYPE_CHECKING:
    from statsmodels.regression.mixed_linear_model import MixedLMResults
    from .lmm import LMMResult
    from .prepare import AnalysisDataset


# =============================================================================
# Style Configuration
# =============================================================================

# Color palette for FO vs BO (or any two-group comparison)
GROUP_COLORS = {
    "FO": "#E64B35",  # Coral red
    "BO": "#4DBBD5",  # Teal blue
    # Generic fallbacks
    0: "#E64B35",
    1: "#4DBBD5",
}

# Default figure style
DEFAULT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
}


def _apply_style() -> None:
    """Apply consistent plotting style."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams.update(DEFAULT_STYLE)
        sns.set_palette("colorblind")


# =============================================================================
# Main Visualization Functions
# =============================================================================

def plot_lmm_trajectories(
    df: pd.DataFrame,
    outcome: str = "emg_p90",
    group: str = "work_type",
    subject: str = "subject_id",
    day: str = "day_index",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    show_ci: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a spaghetti plot with population-level trend overlays.
    
    This is the "intuitive" visualization showing:
    - Individual subject trajectories as light background lines (random effects)
    - Group-level means as bold lines (approximation of fixed effects)
    
    Note: This uses seaborn's mean aggregation, not actual LMM predictions.
    For true LMM fitted values, use plot_lmm_fit() instead.
    
    :param df: DataFrame with longitudinal data
    :param outcome: Name of the outcome column (Y-axis)
    :param group: Name of the grouping variable (e.g., "work_type")
    :param subject: Name of the subject ID column
    :param day: Name of the day/time column (X-axis)
    :param title: Plot title (auto-generated if None)
    :param ylabel: Y-axis label (auto-generated if None)
    :param xlabel: X-axis label (auto-generated if None)
    :param show_ci: Show 95% confidence interval around population means
    :param figsize: Figure size tuple
    :param save_path: Path to save figure (None = display only)
    :returns: matplotlib Figure object
    
    Example:
        >>> plot_lmm_trajectories(
        ...     df=ds["data"],
        ...     outcome="EMG_apdf.active.p90",
        ...     group="work_type"
        ... )
    """
    _apply_style()
    
    # Validate columns exist
    required_cols = [outcome, group, subject, day]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot Individual Trajectories (Random Effects visualization)
    # Light, thin lines showing each subject's path
    sns.lineplot(
        data=df,
        x=day,
        y=outcome,
        hue=group,
        units=subject,
        estimator=None,
        alpha=0.2,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    
    # Plot Population Trends (Fixed Effects approximation)
    # Bold lines showing group means with optional CI
    errorbar = ("ci", 95) if show_ci else None
    sns.lineplot(
        data=df,
        x=day,
        y=outcome,
        hue=group,
        linewidth=3,
        marker="o",
        markersize=8,
        errorbar=errorbar,
        ax=ax,
    )
    
    # Labels and title
    ax.set_xlabel(xlabel or f"Monitoring Day ({day})")
    ax.set_ylabel(ylabel or _format_outcome_label(outcome))
    ax.set_title(title or f"Longitudinal Trajectories: {_format_outcome_label(outcome)} by {group}")
    
    # Improve legend
    ax.legend(title=group, loc="best", frameon=True, fancybox=True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_lmm_fit(
    ds: "AnalysisDataset",
    result: "LMMResult",
    group: Optional[str] = None,
    day: str = "day_index",
    subject: str = "subject_id",
    show_raw: bool = True,
    show_conditional: bool = True,
    show_marginal: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot actual LMM fitted values (not just empirical means).
    
    Shows three layers:
    1. Raw observed data points (if show_raw=True)
    2. Conditional predictions per subject (BLUPs, if show_conditional=True)
    3. Marginal population predictions (if show_marginal=True)
    
    This visualization is more faithful to what the LMM is actually doing
    compared to simple group means.
    
    :param ds: AnalysisDataset dictionary
    :param result: LMMResult from fit_lmm()
    :param group: Grouping variable for color (e.g., "work_type")
    :param day: Day/time column name
    :param subject: Subject ID column name
    :param show_raw: Show raw data points
    :param show_conditional: Show subject-specific predictions (BLUPs)
    :param show_marginal: Show population-level predictions
    :param title: Plot title
    :param figsize: Figure size
    :param save_path: Path to save figure
    :returns: matplotlib Figure object
    
    Example:
        >>> result = fit_lmm(ds, "EMG_apdf.active.p90", fixed_effects=["C(work_type)"])
        >>> plot_lmm_fit(ds, result, group="work_type")
    """
    _apply_style()
    
    model = result.get("model")
    if model is None:
        raise ValueError("LMMResult has no fitted model (model fitting may have failed)")
    
    outcome = result["outcome"]
    df = ds["data"].copy()
    
    # Get fitted values
    df = df.dropna(subset=[outcome, subject])
    
    # Align index with model's fitted values
    model_index = model.fittedvalues.index
    df_aligned = df.loc[df.index.isin(model_index)].copy()
    
    # Add fitted values (conditional = includes random effects)
    df_aligned["fitted_conditional"] = model.fittedvalues.loc[df_aligned.index]
    
    # Marginal predictions (fixed effects only, averaged over random effects)
    # This is the "population average" prediction
    df_aligned["fitted_marginal"] = model.predict(df_aligned)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color mapping
    if group and group in df_aligned.columns:
        groups = df_aligned[group].unique()
        palette = {g: GROUP_COLORS.get(g, f"C{i}") for i, g in enumerate(groups)}
    else:
        group = None
        palette = None
    
    # Layer 1: Raw data points
    if show_raw:
        sns.scatterplot(
            data=df_aligned,
            x=day,
            y=outcome,
            hue=group,
            palette=palette,
            alpha=0.3,
            s=30,
            legend=False,
            ax=ax,
        )
    
    # Layer 2: Conditional predictions (subject-specific lines)
    if show_conditional:
        for subj in df_aligned[subject].unique():
            subj_df = df_aligned[df_aligned[subject] == subj].sort_values(day)
            if len(subj_df) > 1:
                color = "gray"
                if group:
                    grp_val = subj_df[group].iloc[0]
                    color = palette.get(grp_val, "gray") if palette else "gray"
                ax.plot(
                    subj_df[day],
                    subj_df["fitted_conditional"],
                    color=color,
                    alpha=0.25,
                    linewidth=1,
                    zorder=1,
                )
    
    # Layer 3: Marginal predictions (population mean trajectory)
    if show_marginal:
        if group:
            for grp_val in df_aligned[group].unique():
                grp_df = df_aligned[df_aligned[group] == grp_val]
                # Aggregate marginal predictions by day
                marginal_means = grp_df.groupby(day)["fitted_marginal"].mean()
                color = palette.get(grp_val, "black") if palette else "black"
                ax.plot(
                    marginal_means.index,
                    marginal_means.values,
                    color=color,
                    linewidth=4,
                    marker="o",
                    markersize=10,
                    label=f"{grp_val} (marginal)",
                    zorder=10,
                )
        else:
            marginal_means = df_aligned.groupby(day)["fitted_marginal"].mean()
            ax.plot(
                marginal_means.index,
                marginal_means.values,
                color="black",
                linewidth=4,
                marker="o",
                markersize=10,
                label="Marginal mean",
                zorder=10,
            )
    
    # Labels
    ax.set_xlabel(f"Monitoring Day ({day})")
    ax.set_ylabel(_format_outcome_label(outcome))
    ax.set_title(title or f"LMM Fit: {_format_outcome_label(outcome)}")
    
    # Add ICC annotation
    icc = result.get("random_effects", {}).get("icc", np.nan)
    if not np.isnan(icc):
        ax.annotate(
            f"ICC = {icc:.2f}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    
    if show_marginal:
        ax.legend(title=group or "Prediction", loc="best")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_random_intercepts(
    result: "LMMResult",
    group_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the distribution of random intercepts (BLUPs).
    
    This shows "who you are matters" - how much each subject deviates
    from the population mean. Subjects far from zero have consistently
    higher or lower values than average.
    
    :param result: LMMResult from fit_lmm()
    :param group_labels: Optional dict mapping subject IDs to group labels
    :param title: Plot title
    :param figsize: Figure size
    :param save_path: Path to save figure
    :returns: matplotlib Figure object
    
    Example:
        >>> result = fit_lmm(ds, "EMG_apdf.active.p90")
        >>> # Color by work_type
        >>> group_map = ds["data"].groupby("subject_id")["work_type"].first().to_dict()
        >>> plot_random_intercepts(result, group_labels=group_map)
    """
    _apply_style()
    
    model = result.get("model")
    if model is None:
        raise ValueError("LMMResult has no fitted model")
    
    # Extract random effects (BLUPs)
    random_effects = model.random_effects
    
    # Build DataFrame for plotting
    re_data = []
    for subj, effects in random_effects.items():
        intercept = effects.iloc[0] if hasattr(effects, "iloc") else effects[0]
        re_data.append({
            "subject_id": subj,
            "random_intercept": intercept,
            "group": group_labels.get(subj, "Unknown") if group_labels else "All",
        })
    
    re_df = pd.DataFrame(re_data).sort_values("random_intercept")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})
    
    # Left panel: Caterpillar plot (sorted random intercepts)
    ax1 = axes[0]
    
    if group_labels:
        groups = re_df["group"].unique()
        palette = {g: GROUP_COLORS.get(g, f"C{i}") for i, g in enumerate(groups)}
        colors = re_df["group"].map(palette)
    else:
        colors = "#4DBBD5"
    
    ax1.barh(
        range(len(re_df)),
        re_df["random_intercept"],
        color=colors,
        alpha=0.7,
        edgecolor="white",
    )
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Random Intercept (deviation from population mean)")
    ax1.set_ylabel("Subjects (sorted)")
    ax1.set_yticks([])
    
    # Add group legend if applicable
    if group_labels:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=palette[g], label=g, alpha=0.7)
            for g in groups
        ]
        ax1.legend(handles=legend_elements, loc="lower right")
    
    # Right panel: Distribution histogram
    ax2 = axes[1]
    
    if group_labels:
        for grp in groups:
            grp_data = re_df[re_df["group"] == grp]["random_intercept"]
            ax2.hist(
                grp_data,
                bins=15,
                alpha=0.5,
                label=grp,
                color=palette[grp],
                orientation="horizontal",
            )
    else:
        ax2.hist(
            re_df["random_intercept"],
            bins=15,
            alpha=0.7,
            color="#4DBBD5",
            orientation="horizontal",
        )
    
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    
    # Overall title
    outcome = result["outcome"]
    icc = result.get("random_effects", {}).get("icc", np.nan)
    suptitle = title or f"Random Intercepts: {_format_outcome_label(outcome)}"
    if not np.isnan(icc):
        suptitle += f" (ICC = {icc:.2f})"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_model_diagnostics(
    result: "LMMResult",
    title_prefix: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate diagnostic plots for LMM residuals.
    
    Creates a two-panel figure:
    1. QQ plot: Checks normality assumption of residuals
    2. Residuals vs Fitted: Checks homoscedasticity
    
    :param result: LMMResult from fit_lmm()
    :param title_prefix: Prefix for plot titles
    :param figsize: Figure size
    :param save_path: Path to save figure
    :returns: matplotlib Figure object
    
    Example:
        >>> result = fit_lmm(ds, "EMG_apdf.active.p90")
        >>> plot_model_diagnostics(result)
    """
    _apply_style()
    
    model = result.get("model")
    if model is None:
        raise ValueError("LMMResult has no fitted model")
    
    outcome = result["outcome"]
    prefix = title_prefix or _format_outcome_label(outcome)
    
    # Get residuals and fitted values
    residuals = model.resid
    fitted = model.fittedvalues
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: QQ Plot
    ax1 = axes[0]
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title(f"{prefix}: Q-Q Plot")
    ax1.get_lines()[0].set_markerfacecolor("#4DBBD5")
    ax1.get_lines()[0].set_markeredgecolor("#4DBBD5")
    ax1.get_lines()[0].set_alpha(0.6)
    
    # Panel 2: Residuals vs Fitted
    ax2 = axes[1]
    ax2.scatter(fitted, residuals, alpha=0.5, s=30, c="#E64B35")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    
    # Add LOWESS smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.3)
        ax2.plot(smoothed[:, 0], smoothed[:, 1], color="blue", linewidth=2, label="LOWESS")
    except Exception:
        pass
    
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title(f"{prefix}: Residuals vs Fitted")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_group_comparison(
    ds: "AnalysisDataset",
    result: "LMMResult",
    group: str = "work_type",
    show_emmeans: bool = True,
    show_raw: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a focused comparison plot between groups.
    
    Shows:
    - Box/violin plots of raw data per group
    - Estimated marginal means (EMMs) with CIs from the model
    - Effect size annotation
    
    :param ds: AnalysisDataset dictionary
    :param result: LMMResult from fit_lmm()
    :param group: Grouping variable name
    :param show_emmeans: Overlay estimated marginal means
    :param show_raw: Show raw data points
    :param figsize: Figure size
    :param save_path: Path to save figure
    :returns: matplotlib Figure object
    """
    _apply_style()
    
    outcome = result["outcome"]
    df = ds["data"].copy()
    
    if group not in df.columns:
        raise ValueError(f"Group column '{group}' not found in data")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = df[group].dropna().unique()
    palette = {g: GROUP_COLORS.get(g, f"C{i}") for i, g in enumerate(groups)}
    
    # Violin plot with raw points
    sns.violinplot(
        data=df,
        x=group,
        y=outcome,
        hue=group,
        palette=palette,
        alpha=0.3,
        inner=None,
        legend=False,
        ax=ax,
    )
    
    if show_raw:
        sns.stripplot(
            data=df,
            x=group,
            y=outcome,
            hue=group,
            palette=palette,
            alpha=0.4,
            size=4,
            jitter=True,
            legend=False,
            ax=ax,
        )
    
    # Add estimated marginal means if model available
    model = result.get("model")
    if show_emmeans and model is not None:
        # Compute group means from fixed effects
        coef = result.get("coefficients", pd.DataFrame())
        if not coef.empty:
            intercept = coef[coef["term"] == "Intercept"]["estimate"].values
            if len(intercept) > 0:
                base_mean = intercept[0]
                
                # Plot EMM points
                for i, grp in enumerate(groups):
                    # Find coefficient for this group level
                    grp_term = f"C({group})[T.{grp}]"
                    grp_coef = coef[coef["term"] == grp_term]["estimate"].values
                    
                    if len(grp_coef) > 0:
                        emm = base_mean + grp_coef[0]
                    else:
                        emm = base_mean  # Reference level
                    
                    ax.scatter(
                        [i], [emm],
                        s=200,
                        c=palette.get(grp, "black"),
                        marker="D",
                        edgecolor="white",
                        linewidth=2,
                        zorder=10,
                        label=f"EMM ({grp})" if i == 0 else None,
                    )
    
    ax.set_xlabel(group)
    ax.set_ylabel(_format_outcome_label(outcome))
    ax.set_title(f"Group Comparison: {_format_outcome_label(outcome)}")
    
    # Add ICC annotation
    icc = result.get("random_effects", {}).get("icc", np.nan)
    if not np.isnan(icc):
        ax.annotate(
            f"ICC = {icc:.2f}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_continuous_relationship(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    add_regression: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot relationship between a continuous predictor and outcome.

    Shows scatter points and optional regression lines (overall or per group).
    """
    _apply_style()

    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Missing columns: {[c for c in [x, y] if c not in df.columns]}")

    fig, ax = plt.subplots(figsize=figsize)

    if hue and hue in df.columns:
        groups = df[hue].dropna().unique()
        palette = {g: GROUP_COLORS.get(g, f"C{i}") for i, g in enumerate(groups)}
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            alpha=0.5,
            s=40,
            ax=ax,
        )
        if add_regression:
            for grp in groups:
                grp_df = df[df[hue] == grp]
                if len(grp_df) > 2:
                    sns.regplot(
                        data=grp_df,
                        x=x,
                        y=y,
                        scatter=False,
                        color=palette.get(grp, "black"),
                        ax=ax,
                    )
    else:
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            alpha=0.5,
            s=40,
            ax=ax,
        )
        if add_regression:
            sns.regplot(
                data=df,
                x=x,
                y=y,
                scatter=False,
                color="black",
                ax=ax,
            )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or _format_outcome_label(y))
    ax.set_title(title or f"{y} vs {x}")

    if hue and hue in df.columns:
        ax.legend(title=hue, loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ols_diagnostics(
    model: Any,
    title_prefix: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate diagnostic plots for OLS residuals.

    Creates a two-panel figure:
    1. QQ plot for residual normality
    2. Residuals vs fitted for homoscedasticity
    """
    _apply_style()

    residuals = model.resid
    fitted = model.fittedvalues
    prefix = title_prefix or "OLS"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title(f"{prefix}: Q-Q Plot")
    ax1.get_lines()[0].set_markerfacecolor("#4DBBD5")
    ax1.get_lines()[0].set_alpha(0.6)

    ax2 = axes[1]
    ax2.scatter(fitted, residuals, alpha=0.5, s=30, c="#E64B35")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.3)
        ax2.plot(smoothed[:, 0], smoothed[:, 1], color="blue", linewidth=2, label="LOWESS")
    except Exception:
        pass

    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title(f"{prefix}: Residuals vs Fitted")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def _format_outcome_label(outcome: str) -> str:
    """
    Format outcome column name for display.
    
    Converts: "EMG_apdf.active.p90" → "EMG Active P90"
    """
    # Remove common prefixes
    label = outcome
    label = label.replace("EMG_", "EMG ")
    label = label.replace("apdf.", "")
    label = label.replace("intensity.", "")
    label = label.replace("_", " ")
    label = label.replace(".", " ")
    
    # Title case
    label = label.title()
    
    # Fix common abbreviations
    label = label.replace("P90", "P90")
    label = label.replace("P50", "P50")
    label = label.replace("Mvc", "MVC")
    label = label.replace("Emg", "EMG")
    
    return label


def create_lmm_summary_figure(
    ds: "AnalysisDataset",
    result: "LMMResult",
    group: str = "work_type",
    day: str = "day_index",
    subject: str = "subject_id",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive 4-panel summary figure for an LMM.
    
    Panels:
    1. Top-left: Spaghetti plot with trajectories
    2. Top-right: Random intercepts distribution
    3. Bottom-left: QQ plot
    4. Bottom-right: Residuals vs fitted
    
    :param ds: AnalysisDataset dictionary
    :param result: LMMResult from fit_lmm()
    :param group: Grouping variable
    :param day: Day column name
    :param subject: Subject ID column name
    :param figsize: Figure size
    :param save_path: Path to save figure
    :returns: matplotlib Figure object
    """
    _apply_style()
    
    model = result.get("model")
    if model is None:
        raise ValueError("LMMResult has no fitted model")
    
    outcome = result["outcome"]
    df = ds["data"].copy()
    
    fig = plt.figure(figsize=figsize)
    
    # Panel 1: Spaghetti plot
    ax1 = fig.add_subplot(2, 2, 1)
    
    if group in df.columns:
        groups = df[group].unique()
        palette = {g: GROUP_COLORS.get(g, f"C{i}") for i, g in enumerate(groups)}
        
        sns.lineplot(
            data=df, x=day, y=outcome, hue=group, units=subject,
            estimator=None, alpha=0.2, linewidth=0.8, legend=False, ax=ax1,
        )
        sns.lineplot(
            data=df, x=day, y=outcome, hue=group,
            linewidth=3, marker="o", errorbar=("ci", 95), ax=ax1,
        )
    else:
        sns.lineplot(
            data=df, x=day, y=outcome, units=subject,
            estimator=None, alpha=0.2, linewidth=0.8, ax=ax1,
        )
    
    ax1.set_title("Longitudinal Trajectories")
    ax1.set_xlabel("Day")
    ax1.set_ylabel(_format_outcome_label(outcome))
    
    # Panel 2: Random intercepts
    ax2 = fig.add_subplot(2, 2, 2)
    
    random_effects = model.random_effects
    re_values = [effects.iloc[0] if hasattr(effects, "iloc") else effects[0] 
                 for effects in random_effects.values()]
    re_df = pd.DataFrame({
        "subject": list(random_effects.keys()),
        "intercept": re_values,
    }).sort_values("intercept")
    
    if group in df.columns:
        subj_groups = df.groupby(subject)[group].first()
        re_df["group"] = re_df["subject"].map(subj_groups)
        colors = re_df["group"].map(
            {g: GROUP_COLORS.get(g, f"C{i}") for i, g in enumerate(groups)}
        )
    else:
        colors = "#4DBBD5"
    
    ax2.barh(range(len(re_df)), re_df["intercept"], color=colors, alpha=0.7)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Random Intercept")
    ax2.set_ylabel("Subjects (sorted)")
    ax2.set_yticks([])
    ax2.set_title("Random Effects Distribution")
    
    # Add ICC annotation
    icc = result.get("random_effects", {}).get("icc", np.nan)
    if not np.isnan(icc):
        ax2.annotate(f"ICC = {icc:.2f}", xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Panel 3: QQ plot
    ax3 = fig.add_subplot(2, 2, 3)
    
    residuals = model.resid
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Normality Check)")
    ax3.get_lines()[0].set_markerfacecolor("#4DBBD5")
    ax3.get_lines()[0].set_alpha(0.6)
    
    # Panel 4: Residuals vs Fitted
    ax4 = fig.add_subplot(2, 2, 4)
    
    fitted = model.fittedvalues
    ax4.scatter(fitted, residuals, alpha=0.5, s=30, c="#E64B35")
    ax4.axhline(0, color="black", linestyle="--", linewidth=1)
    
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.3)
        ax4.plot(smoothed[:, 0], smoothed[:, 1], color="blue", linewidth=2)
    except Exception:
        pass
    
    ax4.set_xlabel("Fitted Values")
    ax4.set_ylabel("Residuals")
    ax4.set_title("Residuals vs Fitted (Homoscedasticity)")
    
    # Overall title
    fig.suptitle(
        f"LMM Summary: {_format_outcome_label(outcome)}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "plot_lmm_trajectories",
    "plot_lmm_fit",
    "plot_random_intercepts",
    "plot_model_diagnostics",
    "plot_group_comparison",
    "plot_continuous_relationship",
    "plot_ols_diagnostics",
    "create_lmm_summary_figure",
    "GROUP_COLORS",
]
