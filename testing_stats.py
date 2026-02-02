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
    # Plotting / Visualization
    plot_lmm_trajectories,
    plot_lmm_fit,
    plot_random_intercepts,
    plot_model_diagnostics,
    plot_group_comparison,
    create_lmm_summary_figure,
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
    oh_profiles_path = os.getenv("OH_PROFILES_PATH", "/Users/goncalobarros/Documents/projects/OH_profiles")

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
        outcome_type = info.get("outcome_type")
        transform = info.get("transform")
        outcome_type_label = outcome_type.name if hasattr(outcome_type, "name") else str(outcome_type)
        transform_label = transform.name if hasattr(transform, "name") else str(transform)
        print(f"  Type: {outcome_type_label}")
        print(f"  Level: {info.get('level')}")
        print(f"  Transform: {transform_label}")
        print(f"  Is Primary: {info.get('is_primary', False)}")
        print(f"  Description: {info.get('description', 'N/A')}")
    else:
        print("  (Not found)")

    print_section("PIPELINE COMPLETE")


    # ---------------------------------------------------------------------
    # STEP 11: Hypothesis Testing (H1–H6)
    # ---------------------------------------------------------------------
    print_section("[11] HYPOTHESIS TESTING (H1–H6)")

    from hypotheses.runner import run_all as run_hypotheses
    from hypotheses.runner import apply_multiplicity_correction
    from hypotheses.runner import summarize_results

    hypothesis_results = run_hypotheses(profiles, verbose=True)
    if hypothesis_results:
        corrected = apply_multiplicity_correction(hypothesis_results, method="holm", verbose=True)
        summary = summarize_results(hypothesis_results)
        print("\n--- Hypothesis Summary ---")
        print(summary.to_string(index=False))

    # =========================================================================
    # STEP 11: LMM VISUALIZATION (NEW)
    # =========================================================================
    # This section demonstrates the new plotting module for understanding
    # LMM results visually.
    print_section("STEP 11: LMM Visualization")
    
    # We'll use the first successful EMG model from STEP 5 for visualization
    # Let's refit one model specifically for plotting
    print("\nGenerating LMM visualizations...")
    
    # Find a suitable EMG outcome (check both uppercase and lowercase patterns)
    emg_outcomes = [c for c in daily_ds["data"].columns if c.lower().startswith("emg")]
    if emg_outcomes:
        viz_outcome = emg_outcomes[0]
        print(f"\n[11.1] Fitting model for visualization: {viz_outcome}")
        
        # Fit with work_type as fixed effect (if available)
        if "work_type" in daily_ds["data"].columns:
            viz_result = fit_lmm(
                daily_ds,
                outcome=viz_outcome,
                fixed_effects=["C(work_type)", "C(day_index)"],
                random_intercept="subject_id",
            )
        else:
            viz_result = fit_lmm(
                daily_ds,
                outcome=viz_outcome,
                fixed_effects=["C(day_index)"],
                random_intercept="subject_id",
            )
        
        if viz_result.get("model") is not None:
            print(summarize_lmm_result(viz_result))
            
            # Create output directory for figures
            output_dir = os.path.join(os.path.dirname(oh_profiles_path), "figures")
            os.makedirs(output_dir, exist_ok=True)
            
            # [11.2] Spaghetti Plot - Raw trajectories with population means
            print("\n[11.2] Generating spaghetti plot (raw trajectories)...")
            try:
                fig_traj = plot_lmm_trajectories(
                    df=daily_ds["data"],
                    outcome=viz_outcome,
                    group="work_type" if "work_type" in daily_ds["data"].columns else None,
                    subject="subject_id",
                    day="day_index",
                    save_path=os.path.join(output_dir, "lmm_trajectories.png"),
                )
                print(f"   Saved: {output_dir}/lmm_trajectories.png")
            except Exception as e:
                print(f"   Skipped spaghetti plot: {e}")
            
            # [11.3] LMM Fit Plot - Actual model predictions
            print("\n[11.3] Generating LMM fit plot (model predictions)...")
            try:
                fig_fit = plot_lmm_fit(
                    ds=daily_ds,
                    result=viz_result,
                    group="work_type" if "work_type" in daily_ds["data"].columns else None,
                    day="day_index",
                    save_path=os.path.join(output_dir, "lmm_fit.png"),
                )
                print(f"   Saved: {output_dir}/lmm_fit.png")
            except Exception as e:
                print(f"   Skipped LMM fit plot: {e}")
            
            # [11.4] Random Intercepts - "Who you are matters"
            print("\n[11.4] Generating random intercepts plot...")
            try:
                # Build group mapping for coloring
                if "work_type" in daily_ds["data"].columns:
                    group_map = (
                        daily_ds["data"]
                        .groupby("subject_id")["work_type"]
                        .first()
                        .to_dict()
                    )
                else:
                    group_map = None
                
                fig_re = plot_random_intercepts(
                    result=viz_result,
                    group_labels=group_map,
                    save_path=os.path.join(output_dir, "lmm_random_intercepts.png"),
                )
                print(f"   Saved: {output_dir}/lmm_random_intercepts.png")
            except Exception as e:
                print(f"   Skipped random intercepts plot: {e}")
            
            # [11.5] Model Diagnostics - QQ + Residuals vs Fitted
            print("\n[11.5] Generating diagnostic plots...")
            try:
                fig_diag = plot_model_diagnostics(
                    result=viz_result,
                    save_path=os.path.join(output_dir, "lmm_diagnostics.png"),
                )
                print(f"   Saved: {output_dir}/lmm_diagnostics.png")
            except Exception as e:
                print(f"   Skipped diagnostic plots: {e}")
            
            # [11.6] Comprehensive 4-panel summary
            print("\n[11.6] Generating comprehensive summary figure...")
            try:
                fig_summary = create_lmm_summary_figure(
                    ds=daily_ds,
                    result=viz_result,
                    group="work_type" if "work_type" in daily_ds["data"].columns else None,
                    save_path=os.path.join(output_dir, "lmm_summary.png"),
                )
                print(f"   Saved: {output_dir}/lmm_summary.png")
            except Exception as e:
                print(f"   Skipped summary figure: {e}")
            
            print(f"\n✓ All figures saved to: {output_dir}/")
        else:
            print("   Model fitting failed - skipping visualizations")
    else:
        print("   No EMG outcomes available for visualization")


if __name__ == "__main__":
    main()
