"""
OH Stats Comprehensive Analysis Entry Point
===========================================

This script is the **main, step-by-step entry point** for running the full
oh_stats analysis pipeline on OH profiles. It is intentionally verbose and
heavily commented so you can follow **every step** of the workflow.

Pipeline overview:
1) Load OH profiles
2) Discover available data
3) Prepare analysis datasets
4) Run data quality checks
5) Fit statistical models (LMM)
6) Post-hoc contrasts and effect sizes
7) Multiplicity correction (FDR + Holm)
8) Diagnostics
9) Reporting + export
10) Registry inspection

Run from the project root:
    python testing_stats.py
"""
import os
import warnings

import numpy as np
import pandas as pd

from oh_parser import load_profiles, list_subjects

from oh_stats import (
    discover_sensors,
    discover_questionnaires,
    get_profile_summary,
    prepare_daily_emg,
    prepare_daily_metrics,
    prepare_single_instance_metrics,
    prepare_daily_questionnaires,
    add_subject_metadata,
    aggregate_daily_to_subject,
    describe_dataset,
    validate_dataset,
    subset_dataset,
    get_n_subjects,
    get_n_observations,
    get_date_range,
    get_obs_per_subject,
    summarize_outcomes,
    check_normality,
    check_variance,
    missingness_report,
    fit_lmm,
    fit_all_outcomes,
    summarize_lmm_result,
    apply_transform,
    prepare_from_dataframe,
    compare_models,
    get_residuals,
    get_fitted_values,
    get_random_effects,
    pairwise_contrasts,
    compute_emmeans,
    compute_effect_size,
    test_linear_trend,
    apply_fdr,
    apply_holm,
    apply_holm_hypotheses,
    adjust_pvalues,
    significant_outcomes,
    fdr_summary,
    residual_diagnostics,
    summarize_diagnostics,
    descriptive_table,
    coefficient_table,
    coefficient_table_multiple,
    descriptive_table_formatted,
    results_summary,
    export_to_csv,
    export_to_latex,
    print_results_summary,
    print_coefficient_summary,
)

from oh_stats.registry import (
    OutcomeType,
    TransformType,
    list_outcomes,
    get_outcome_info,
)

import statsmodels.api as sm

# Suppress some convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


def print_section(title: str) -> None:
    """Utility to print a consistent section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    """Run the full OH stats pipeline end-to-end."""
    # ---------------------------------------------------------------------
    # STEP 1: Load OH profiles
    # ---------------------------------------------------------------------
    print_section("[1] LOADING OH PROFILES")

    # Path can be overridden by environment variable
    oh_profiles_path = os.getenv("OH_PROFILES_PATH", r"E:\Backup PrevOccupAI_PLUS Data\OH_profiles_2")

    print(f"Using OH profile path: {oh_profiles_path}")
    profiles = load_profiles(oh_profiles_path)
    subjects = list_subjects(profiles)
    print(f"Loaded {len(subjects)} subjects: {subjects[:5]}...")

    # ---------------------------------------------------------------------
    # STEP 2: Data discovery
    # ---------------------------------------------------------------------
    print_section("[2] DATA DISCOVERY")

    print("\n--- Available Sensors ---")
    sensors = discover_sensors(profiles)
    for sensor, metrics in sensors.items():
        print(f"  {sensor}: {len(metrics)} metrics")
        if metrics:
            print(f"    Sample: {list(metrics)[:3]}...")

    print("\n--- Available Questionnaires ---")
    questionnaires = discover_questionnaires(profiles)
    for q_type, q_names in questionnaires.items():
        print(f"  {q_type}: {q_names}")

    print("\n--- Profile Summary (First 500 chars) ---")
    print(get_profile_summary(profiles)[:500])

    # ---------------------------------------------------------------------
    # STEP 3: Data preparation
    # ---------------------------------------------------------------------
    print_section("[3] DATA PREPARATION")

    # 3A. EMG daily (right side only for analysis)
    print("\n[3A] Daily EMG (side=right)")
    emg_ds = prepare_daily_emg(profiles, side="right")
    print(f"  Shape: {emg_ds['data'].shape}")
    print(f"  Outcomes: {len(emg_ds['outcome_vars'])}")
    print(f"  Grouping: {emg_ds['grouping_vars']}")
    print(emg_ds["data"].head(5).to_string())

    # 3B. Daily questionnaire data (if present)
    print("\n[3B] Daily questionnaires")
    qs_ds = prepare_daily_questionnaires(profiles)
    if qs_ds is None:
        print("  Daily questionnaire data not available")
    else:
        print(f"  Shape: {qs_ds['data'].shape}")

    # 3C. Unified daily metrics (HR + noise + HAR + workload + EMG p90/p50)
    print("\n[3C] Unified daily metrics (custom)")
    daily_ds = prepare_daily_metrics(profiles)
    print(f"  Shape: {daily_ds['data'].shape}")
    print(f"  Outcomes: {len(daily_ds['outcome_vars'])}")
    print(daily_ds["data"].head(5).to_string())

    # 3D. Single-instance metrics (metadata + IPAQ/OSPAQ + weekly HAR)
    print("\n[3D] Single-instance metrics")
    single_ds = prepare_single_instance_metrics(profiles)
    print(f"  Shape: {single_ds['data'].shape}")
    print(f"  Outcomes: {len(single_ds['outcome_vars'])}")
    print(single_ds["data"].head(5).to_string())

    # Add metadata to daily dataset (e.g., work_type)
    print("\n[3E] Add subject metadata (work_type)")
    daily_ds = add_subject_metadata(daily_ds, profiles, fields=["work_type"])
    print(f"  Columns now include work_type: {'work_type' in daily_ds['data'].columns}")

    # ---------------------------------------------------------------------
    # STEP 4: Dataset utilities + QA checks
    # ---------------------------------------------------------------------
    print_section("[4] DATASET UTILITIES + QA CHECKS")

    print("\n--- describe_dataset(daily_ds) ---")
    print(describe_dataset(daily_ds))

    print("\n--- validate_dataset(daily_ds) ---")
    try:
        validate_dataset(daily_ds)
        print("  Valid: True")
    except ValueError as e:
        print(f"  Valid: False -> {e}")

    print("\n--- Quick Accessors ---")
    print(f"  N subjects: {get_n_subjects(daily_ds)}")
    print(f"  N observations: {get_n_observations(daily_ds)}")
    print(f"  Date range: {get_date_range(daily_ds)}")
    print(f"  Obs per subject (first 3): {dict(list(get_obs_per_subject(daily_ds).items())[:3])}")

    print("\n--- summarize_outcomes(daily_ds) ---")
    summary = summarize_outcomes(daily_ds)
    print(summary.head(10).to_string(index=False))

    print("\n--- check_variance(daily_ds) ---")
    variance = check_variance(daily_ds)
    print(variance.head(10).to_string(index=False))

    print("\n--- missingness_report(daily_ds) ---")
    miss = missingness_report(daily_ds)
    print(f"Total missing: {miss['summary']['total_missing']} cells ({miss['summary']['pct_missing']:.1f}%)")

    # ---------------------------------------------------------------------
    # STEP 5: Fit Linear Mixed Models (example H1/H2 patterns)
    # ---------------------------------------------------------------------
    print_section("[5] LINEAR MIXED MODELS")

    # Example outcome (if present)
    outcome = "emg_apdf_active_p90"
    if outcome in daily_ds["data"].columns:
        print(f"\n--- Fit LMM: {outcome} ~ work_type + C(day_index) ---")
        result = fit_lmm(
            daily_ds,
            outcome=outcome,
            fixed_effects=["work_type", "C(day_index)"],
            random_intercept="subject_id",
        )
        print(summarize_lmm_result(result))
        print(result["coefficients"].to_string(index=False))
    else:
        print(f"\n[Skip] Outcome not found: {outcome}")
        result = None

    # Demonstrate transforms
    print("\n--- Transform demo (LOG, SQRT) ---")
    if outcome in daily_ds["data"].columns:
        values = daily_ds["data"][outcome].dropna()
        if not values.empty:
            log_vals = apply_transform(values, TransformType.LOG)
            sqrt_vals = apply_transform(values, TransformType.SQRT)
            print(f"  Original range: {values.min():.4f} - {values.max():.4f}")
            print(f"  LOG range: {log_vals.min():.4f} - {log_vals.max():.4f}")
            print(f"  SQRT range: {sqrt_vals.min():.4f} - {sqrt_vals.max():.4f}")

    # Batch fit a few outcomes
    print("\n--- Batch fitting (first 5 numeric outcomes) ---")
    if daily_ds["outcome_vars"]:
        results = fit_all_outcomes(daily_ds, outcomes=daily_ds["outcome_vars"][:5])
        for name, r in results.items():
            status = "✓" if r["converged"] else "✗"
            print(f"  {status} {name} (AIC={r['fit_stats'].get('aic', 'NA')})")
    else:
        results = {}

    # ---------------------------------------------------------------------
    # STEP 6: Post-hoc contrasts + effect sizes
    # ---------------------------------------------------------------------
    print_section("[6] POST-HOC CONTRASTS + EFFECT SIZES")

    if result and result["converged"]:
        print("\n--- Pairwise contrasts on day_index ---")
        contrast_result = pairwise_contrasts(result, factor="day_index", ds=daily_ds, correction="holm")
        contrasts_df = contrast_result.get("contrasts")
        if contrasts_df is not None and not contrasts_df.empty:
            print(contrasts_df.head(10).to_string(index=False))
        else:
            print("  No contrasts available")

        print("\n--- Estimated marginal means ---")
        emmeans_df = compute_emmeans(result, factor="day_index", ds=daily_ds)
        print(emmeans_df.to_string(index=False))

        print("\n--- Effect size ---")
        effect = compute_effect_size(result, ds=daily_ds)
        print(effect)

        print("\n--- Linear trend test ---")
        trend = test_linear_trend(result, factor="day_index")
        print(trend)

    # ---------------------------------------------------------------------
    # STEP 7: Multiplicity correction
    # ---------------------------------------------------------------------
    print_section("[7] MULTIPLICITY CORRECTION")

    if results:
        fdr_results = apply_fdr(results, term="day_index", method="fdr_bh")
        print("\n--- FDR (BH) ---")
        print(fdr_results.head(10).to_string(index=False))
        print(fdr_summary(fdr_results))

        holm_results = apply_holm(results, term="day_index")
        print("\n--- Holm (per-outcome) ---")
        print(holm_results.head(10).to_string(index=False))

        print("\n--- Significant outcomes (FDR) ---")
        print(significant_outcomes(fdr_results))

    print("\n--- Hypothesis-level Holm demo ---")
    hypotheses = {
        "H1": {"p_value": 0.01},
        "H2": {"p_value": 0.03},
        "H3": {"p_value": 0.20},
        "H4": {"p_value": 0.04},
        "H6": {"p_value": 0.50},
    }
    print(apply_holm_hypotheses(hypotheses))

    # ---------------------------------------------------------------------
    # STEP 8: Diagnostics
    # ---------------------------------------------------------------------
    print_section("[8] MODEL DIAGNOSTICS")

    if result and result["converged"]:
        diag = residual_diagnostics(result)
        print(summarize_diagnostics(diag))
        print(diag)

    # ---------------------------------------------------------------------
    # STEP 9: Reporting + export
    # ---------------------------------------------------------------------
    print_section("[9] REPORTING + EXPORT")

    if daily_ds["outcome_vars"]:
        table1 = descriptive_table(daily_ds, outcomes=daily_ds["outcome_vars"][:3])
        print("\n--- Table 1 ---")
        print(table1.to_string(index=False))

        if result and result["converged"]:
            coef_table = coefficient_table(result)
            print("\n--- Coefficients ---")
            print(coef_table.to_string(index=False))

        if results:
            res_summary = results_summary(results, apply_fdr(results))
            print("\n--- Results summary ---")
            print(res_summary.head(10).to_string(index=False))

            coef_multi = coefficient_table_multiple(results)
            print("\n--- Multiple coefficients ---")
            print(coef_multi.head(10).to_string(index=False))

        formatted = descriptive_table_formatted(daily_ds, outcomes=daily_ds["outcome_vars"][:2])
        print("\n--- Formatted Table 1 ---")
        print(formatted.to_string(index=False))

    # Export examples
    import tempfile

    if results:
        res_summary = results_summary(results, apply_fdr(results))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            export_to_csv(res_summary, f.name)
            print(f"\nExported CSV: {f.name}")

    if daily_ds["outcome_vars"]:
        table1 = descriptive_table(daily_ds, outcomes=daily_ds["outcome_vars"][:3])
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
            try:
                export_to_latex(table1, f.name)
                print(f"Exported LaTeX: {f.name}")
            except ImportError as e:
                print(f"LaTeX export skipped (missing dependency): {e}")

    if results:
        print("\n--- print_results_summary() ---")
        print_results_summary(results, apply_fdr(results))

    if result and result["converged"]:
        print("\n--- print_coefficient_summary() ---")
        print_coefficient_summary(result)

    # ---------------------------------------------------------------------
    # STEP 10: Registry inspection
    # ---------------------------------------------------------------------
    print_section("[10] OUTCOME REGISTRY")

    print(f"Total outcomes: {len(list_outcomes())}")
    print(f"Continuous: {len(list_outcomes(outcome_type=OutcomeType.CONTINUOUS))}")
    print(f"Proportion: {len(list_outcomes(outcome_type=OutcomeType.PROPORTION))}")
    print(f"Count: {len(list_outcomes(outcome_type=OutcomeType.COUNT))}")

    info = get_outcome_info("EMG_apdf.active.p50")
    print("\n--- Outcome Info: EMG_apdf.active.p50 ---")
    if info:
        print(f"  Type: {info['outcome_type'].name}")
        print(f"  Level: {info['level']}")
        print(f"  Transform: {info['transform'].name}")
        print(f"  Is Primary: {info.get('is_primary', False)}")
        print(f"  Description: {info.get('description', 'N/A')}")
    else:
        print("  (Not found)")

    print_section("PIPELINE COMPLETE")


    # ---------------------------------------------------------------------
    # STEP 11: Hypothesis Testing (H1–H6)
    # ---------------------------------------------------------------------
    print_section("[11] HYPOTHESIS TESTING (H1–H6)")

    hypothesis_results = {}

    # H1: FO vs BO differ in daily EMG p90 (right side only)
    if "emg_apdf_active_p90" in daily_ds["data"].columns and "work_type" in daily_ds["data"].columns:
        print("\n[H1] FO vs BO on EMG p90 (daily)")
        h1_result = fit_lmm(
            daily_ds,
            outcome="emg_apdf_active_p90",
            fixed_effects=["work_type", "C(day_index)"],
            random_intercept="subject_id",
        )
        h1_p = h1_result["coefficients"].loc[
            h1_result["coefficients"]["term"].str.contains("work_type", case=False, na=False),
            "p_value",
        ]
        h1_p = float(h1_p.iloc[0]) if not h1_p.empty else np.nan
        hypothesis_results["H1"] = {"p_value": h1_p, "note": "FO vs BO on EMG p90"}
        print(summarize_lmm_result(h1_result))
    else:
        print("\n[H1] Skipped (missing emg_apdf_active_p90 or work_type)")

    # H2: FO vs BO differ in daily workload (stress proxy)
    if "workload_mean" in daily_ds["data"].columns and "work_type" in daily_ds["data"].columns:
        print("\n[H2] FO vs BO on daily workload mean")
        if daily_ds["data"]["workload_mean"].dropna().empty:
            print("[H2] Skipped (workload_mean has no non-missing values)")
        else:
            h2_result = fit_lmm(
                daily_ds,
                outcome="workload_mean",
                fixed_effects=["work_type", "C(day_index)"],
                random_intercept="subject_id",
            )
            if h2_result["coefficients"].empty:
                print("[H2] Skipped (model produced no coefficients)")
            else:
                h2_p = h2_result["coefficients"].loc[
                    h2_result["coefficients"]["term"].str.contains("work_type", case=False, na=False),
                    "p_value",
                ]
                h2_p = float(h2_p.iloc[0]) if not h2_p.empty else np.nan
                hypothesis_results["H2"] = {"p_value": h2_p, "note": "FO vs BO on workload mean"}
                print(summarize_lmm_result(h2_result))
    else:
        print("\n[H2] Skipped (missing workload_mean or work_type)")

    # H3: Daily stress predicts daily sitting proportion
    if "workload_mean" in daily_ds["data"].columns and "har_sentado_prop" in daily_ds["data"].columns:
        print("\n[H3] Daily workload predicts daily sitting proportion")
        df_h3 = daily_ds["data"].copy()
        if df_h3["workload_mean"].dropna().empty:
            print("[H3] Skipped (workload_mean has no non-missing values)")
        else:
            eps = 1e-6
            df_h3["har_sit_logit"] = np.log(
                df_h3["har_sentado_prop"].clip(eps, 1 - eps) /
                (1 - df_h3["har_sentado_prop"].clip(eps, 1 - eps))
            )
            h3_ds = prepare_from_dataframe(
                df=df_h3[["subject_id", "date", "har_sit_logit", "workload_mean", "day_index"]].dropna(),
                sensor="har",
                level="daily",
                id_col="subject_id",
                date_col="date",
                outcome_cols=["har_sit_logit"],
            )
            if h3_ds["data"].empty:
                print("[H3] Skipped (no valid rows after logit + dropna)")
            else:
                h3_result = fit_lmm(
                    h3_ds,
                    outcome="har_sit_logit",
                    fixed_effects=["workload_mean", "C(day_index)"],
                    random_intercept="subject_id",
                )
                if h3_result["coefficients"].empty:
                    print("[H3] Skipped (model produced no coefficients)")
                else:
                    h3_p = h3_result["coefficients"].loc[
                        h3_result["coefficients"]["term"].str.contains("workload_mean", case=False, na=False),
                        "p_value",
                    ]
                    h3_p = float(h3_p.iloc[0]) if not h3_p.empty else np.nan
                    hypothesis_results["H3"] = {"p_value": h3_p, "note": "workload -> sitting proportion"}
                    print(summarize_lmm_result(h3_result))
    else:
        print("\n[H3] Skipped (missing workload_mean or har_sentado_prop)")

    # H4: OSPAQ sitting predicts objective sitting (subject-level)
    if "ospaq_sitting_frac" in single_ds["data"].columns and "har_sentado_prop" in daily_ds["data"].columns:
        print("\n[H4] OSPAQ sitting vs objective sitting (subject-level)")
        obj = daily_ds["data"].copy()
        obj = obj.dropna(subset=["har_sentado_prop"])
        obj_subject = obj.groupby("subject_id")["har_sentado_prop"].mean().reset_index()
        df_h4 = single_ds["data"][["subject_id", "ospaq_sitting_frac", "work_type"]].merge(
            obj_subject,
            on="subject_id",
            how="inner",
        )
        df_h4 = df_h4.dropna(subset=["ospaq_sitting_frac", "har_sentado_prop"])
        if not df_h4.empty:
            X = df_h4[["ospaq_sitting_frac"]]
            if "work_type" in df_h4.columns:
                X = pd.concat([X, pd.get_dummies(df_h4["work_type"], drop_first=True)], axis=1)
            X = sm.add_constant(X, has_constant="add")
            y = df_h4["har_sentado_prop"]
            model = sm.OLS(y, X).fit()
            h4_p = model.pvalues.get("ospaq_sitting_frac", np.nan)
            hypothesis_results["H4"] = {"p_value": float(h4_p), "note": "OSPAQ -> objective sitting"}
            print(model.summary().as_text().splitlines()[0])
        else:
            print("[H4] Skipped (no overlap between OSPAQ and objective sitting)")
    else:
        print("\n[H4] Skipped (missing OSPAQ or objective sitting data)")

    # H5: Daily physiological predictors of EMG p90 (exploratory)
    predictors = [c for c in ["hr_ratio_mean", "noise_mean"] if c in daily_ds["data"].columns]
    if "emg_apdf_active_p90" in daily_ds["data"].columns and predictors:
        print("\n[H5] Physiological predictors of EMG p90 (exploratory)")
        h5_fixed = predictors + ["C(day_index)"]
        h5_result = fit_lmm(
            daily_ds,
            outcome="emg_apdf_active_p90",
            fixed_effects=h5_fixed,
            random_intercept="subject_id",
        )
        coeffs = h5_result.get("coefficients")
        if isinstance(coeffs, pd.DataFrame) and "term" in coeffs.columns:
            h5_p = coeffs.loc[
                coeffs["term"].str.contains(predictors[0], case=False, na=False),
                "p_value",
            ]
            h5_p = float(h5_p.iloc[0]) if not h5_p.empty else np.nan
            hypothesis_results["H5"] = {"p_value": h5_p, "note": "Physiological predictors"}
        else:
            hypothesis_results["H5"] = {"p_value": np.nan, "note": "No coefficients returned"}
        print(summarize_lmm_result(h5_result))
    else:
        print("\n[H5] Skipped (missing EMG p90 or predictors)")

    # H6: FO vs BO differ in posture metric (if available)
    posture_cols = [c for c in daily_ds["data"].columns if c.startswith("posture_")]
    if posture_cols and "work_type" in daily_ds["data"].columns:
        posture_outcome = posture_cols[0]
        print(f"\n[H6] FO vs BO on posture metric: {posture_outcome}")
        h6_result = fit_lmm(
            daily_ds,
            outcome=posture_outcome,
            fixed_effects=["work_type", "C(day_index)"],
            random_intercept="subject_id",
        )
        h6_p = h6_result["coefficients"].loc[
            h6_result["coefficients"]["term"].str.contains("work_type", case=False, na=False),
            "p_value",
        ]
        h6_p = float(h6_p.iloc[0]) if not h6_p.empty else np.nan
        hypothesis_results["H6"] = {"p_value": h6_p, "note": "FO vs BO on posture"}
        print(summarize_lmm_result(h6_result))
    else:
        print("\n[H6] Skipped (no posture metric available)")

    if hypothesis_results:
        print("\n--- Holm correction across hypotheses ---")
        print(apply_holm_hypotheses(hypothesis_results))


if __name__ == "__main__":
    main()
