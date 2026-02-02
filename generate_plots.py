#!/usr/bin/env python3
"""
Generate key plots for hypothesis testing and diagnostics.

Outputs figures into the specified folder (default: plots/).
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from oh_parser import load_profiles
from hypotheses.config import HYPOTHESES, get_hypothesis
from hypotheses.runner import (
    _prepare_data,
    _run_lmm,
    _apply_logit_transform,
    _apply_log_transform,
    _apply_sqrt_transform,
)
from oh_stats.plotting import (
    create_lmm_summary_figure,
    plot_group_comparison,
    plot_continuous_relationship,
    plot_ols_diagnostics,
)


def _ensure_transformed_outcome(ds: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Ensure transformed outcome column exists in ds and return its name."""
    outcome = config["outcome"]
    transformed = config.get("_transformed_outcome")
    if not transformed:
        return outcome

    if transformed in ds["data"].columns:
        return transformed

    if transformed.endswith("_logit"):
        ds["data"] = _apply_logit_transform(ds["data"], outcome)
    elif transformed.endswith("_log"):
        ds["data"] = _apply_log_transform(ds["data"], outcome)
    elif transformed.endswith("_sqrt"):
        ds["data"] = _apply_sqrt_transform(ds["data"], outcome)

    return transformed


def _build_h4_ols_model(single_ds: Dict[str, Any], daily_ds: Dict[str, Any]) -> Optional[tuple[pd.DataFrame, Any]]:
    """Recreate H4 OLS model for plotting and diagnostics."""
    if "ospaq_sitting_frac" not in single_ds["data"].columns:
        return None
    if "har_sentado_prop" not in daily_ds["data"].columns:
        return None

    obj = daily_ds["data"].dropna(subset=["har_sentado_prop"])

    if {"har_sentado_duration_sec", "har_total_duration_sec"}.issubset(obj.columns):
        obj_subject = (
            obj.groupby("subject_id")[
                ["har_sentado_duration_sec", "har_total_duration_sec"]
            ]
            .sum(min_count=1)
            .reset_index()
        )
        obj_subject["har_sentado_prop"] = (
            obj_subject["har_sentado_duration_sec"]
            / obj_subject["har_total_duration_sec"]
        )
        obj_subject = obj_subject[["subject_id", "har_sentado_prop"]]
    elif "har_total_duration_sec" in obj.columns:
        def weighted_mean(group):
            weights = group["har_total_duration_sec"]
            if weights.sum() > 0:
                return np.average(group["har_sentado_prop"], weights=weights)
            return group["har_sentado_prop"].mean()
        obj_subject = obj.groupby("subject_id").apply(weighted_mean).reset_index()
        obj_subject.columns = ["subject_id", "har_sentado_prop"]
    else:
        obj_subject = (
            obj.groupby("subject_id")["har_sentado_prop"]
            .mean()
            .reset_index()
        )

    df = single_ds["data"][["subject_id", "ospaq_sitting_frac"]].copy()
    if "work_type" in single_ds["data"].columns:
        df["work_type"] = single_ds["data"]["work_type"]
    elif "work_type" in daily_ds["data"].columns:
        wt = daily_ds["data"].groupby("subject_id")["work_type"].first().reset_index()
        df = df.merge(wt, on="subject_id", how="left")

    df = df.merge(obj_subject, on="subject_id", how="inner")
    df = df.dropna(subset=["ospaq_sitting_frac", "har_sentado_prop"])
    if df.empty:
        return None

    eps = 1e-6
    p_clipped = df["har_sentado_prop"].clip(eps, 1 - eps)
    df["har_sentado_prop_logit"] = np.log(p_clipped / (1 - p_clipped))

    X = df[["ospaq_sitting_frac"]].copy().astype(float)
    if "work_type" in df.columns:
        work_dummies = pd.get_dummies(df["work_type"], drop_first=True, dtype=float)
        X = pd.concat([X, work_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df["har_sentado_prop_logit"].astype(float)
    model = sm.OLS(y, X).fit()

    return df, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate hypothesis plots")
    parser.add_argument(
        "--profiles-path",
        default=os.getenv("OH_PROFILES_PATH", "/Users/goncalobarros/Documents/projects/OH_profiles"),
        help="Path to OH profiles directory",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    hyp_dir = os.path.join(out_dir, "hypotheses")
    os.makedirs(hyp_dir, exist_ok=True)

    profiles = load_profiles(args.profiles_path)

    for h_id in HYPOTHESES.keys():
        config = dict(get_hypothesis(h_id))
        config["_id"] = h_id

        data, config, error = _prepare_data(profiles, config)
        if error:
            print(f"[{h_id}] Skipped plot generation: {error}")
            continue

        if config.get("model") == "lmm":
            result = _run_lmm(data, config)
            model_result = result.get("model_result")
            if not model_result or not model_result.get("model"):
                print(f"[{h_id}] No model available for plotting")
                continue

            ds = data
            outcome_for_plot = _ensure_transformed_outcome(ds, result["config"])
            model_result["outcome"] = outcome_for_plot

            summary_path = os.path.join(hyp_dir, f"{h_id}_summary.png")
            create_lmm_summary_figure(
                ds,
                model_result,
                group="work_type" if "work_type" in ds["data"].columns else "work_type",
                save_path=summary_path,
            )

            if result["config"].get("primary_predictor") == "work_type":
                group_path = os.path.join(hyp_dir, f"{h_id}_group_comparison.png")
                plot_group_comparison(
                    ds,
                    model_result,
                    group="work_type",
                    save_path=group_path,
                )

            if h_id == "H3":
                rel_path = os.path.join(hyp_dir, f"{h_id}_workload_vs_sitting.png")
                plot_continuous_relationship(
                    ds["data"],
                    x="workload_mean",
                    y=config["outcome"],
                    hue="work_type" if "work_type" in ds["data"].columns else None,
                    title="Workload vs Sitting (Daily)",
                    save_path=rel_path,
                )

            if h_id == "H5":
                rel_path = os.path.join(hyp_dir, f"{h_id}_posture_vs_emg.png")
                plot_continuous_relationship(
                    ds["data"],
                    x="posture_95_confidence_ellipse_area",
                    y=config["outcome"],
                    hue="work_type" if "work_type" in ds["data"].columns else None,
                    title="Posture vs EMG (Daily)",
                    save_path=rel_path,
                )

        elif config.get("model") == "ols" and h_id == "H4":
            if not isinstance(data, dict) or "single" not in data or "daily" not in data:
                print(f"[{h_id}] Missing data for OLS plotting")
                continue

            h4_tuple = _build_h4_ols_model(data["single"], data["daily"])
            if h4_tuple is None:
                print(f"[{h_id}] No data available for OLS plotting")
                continue

            df, model = h4_tuple
            scatter_path = os.path.join(hyp_dir, f"{h_id}_ospaq_vs_objective.png")
            plot_continuous_relationship(
                df,
                x="ospaq_sitting_frac",
                y="har_sentado_prop",
                hue="work_type" if "work_type" in df.columns else None,
                title="OSPAQ vs Objective Sitting",
                save_path=scatter_path,
            )

            diag_path = os.path.join(hyp_dir, f"{h_id}_ols_diagnostics.png")
            plot_ols_diagnostics(
                model,
                title_prefix="H4 OLS",
                save_path=diag_path,
            )

    print(f"Plots saved to: {hyp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
