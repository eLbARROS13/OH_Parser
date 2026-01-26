"""
OH Stats - Statistical Analysis Package for OH Profiles
========================================================

A multilevel inference engine for Occupational Health profile data.
Designed to work with oh_parser output and support heterogeneous outcome types.

Architecture Note:
    This package uses dictionaries (TypedDicts) instead of classes for data 
    structures to maintain consistency with the oh_parser project style.
    All data containers are plain Python dicts with documented keys.

Architecture:
- registry: Outcome type registry (continuous/ordinal/proportion/count)
- prepare: Data preparation and harmonization
- descriptive: Summary statistics and QA checks
- lmm: Linear Mixed Models for continuous outcomes
- posthoc: Pairwise contrasts and effect sizes
- multiplicity: FDR/Holm corrections
- diagnostics: Model assumption checks
- report: Auto-generated summary tables

Usage:
    from oh_parser import load_profiles
    from oh_stats import (
        prepare_daily_emg,
        describe_dataset,
        summarize_outcomes,
        fit_lmm,
        pairwise_contrasts,
        apply_fdr,
    )
    
    profiles = load_profiles("/path/to/OH_profiles")
    ds = prepare_daily_emg(profiles, side="both")
    print(describe_dataset(ds))
    summary = summarize_outcomes(ds)
    result = fit_lmm(ds, outcome="EMG_intensity.mean_percent_mvc")
"""
from __future__ import annotations

__version__ = "0.3.0"

# Registry - Enums and TypedDicts
from .registry import (
    OutcomeType,
    AnalysisLevel,
    TransformType,
    OutcomeInfo,
    create_outcome_info,
    OUTCOME_REGISTRY,
    get_outcome_info,
    register_outcome,
    list_outcomes,
    get_primary_outcomes,
    get_continuous_outcomes,
    get_proportion_outcomes,
    get_count_outcomes,
    get_questionnaire_outcomes,
    get_emg_outcomes,
    get_daily_outcomes,
    get_single_instance_outcomes,
    reset_registry,
)

# Data preparation - TypedDict and helper functions
from .prepare import (
    AnalysisDataset,
    create_analysis_dataset,
    validate_dataset,
    describe_dataset,
    subset_dataset,
    get_n_subjects,
    get_n_observations,
    get_date_range,
    get_obs_per_subject,
    # Core preparation (from pre-extracted DataFrame)
    prepare_from_dataframe,
    # Generic preparation (from profiles)
    prepare_sensor_data,
    # Discovery functions
    discover_sensors,
    discover_questionnaires,
    get_profile_summary,
    # EMG convenience wrappers (common use case)
    prepare_daily_emg,
    prepare_weekly_emg,
    prepare_daily_metrics,
    prepare_single_instance_metrics,
    add_subject_metadata,
    aggregate_daily_to_subject,
    # Questionnaire preparation (special nested structure)
    prepare_daily_questionnaires,
    prepare_baseline_questionnaires,
    prepare_daily_workload,
    prepare_daily_pain,
    # Utilities
    compute_composite_score,
    align_sensor_questionnaire,
    parse_date,
)

# Descriptive statistics
from .descriptive import (
    NormalityResult,
    VarianceCheckResult,
    summarize_outcomes,
    check_normality,
    check_variance,
    get_non_degenerate_outcomes,
    missingness_report,
    print_missingness_summary,
)

# Linear Mixed Models - TypedDict and functions
from .lmm import (
    LMMResult,
    create_lmm_result,
    summarize_lmm_result,
    apply_transform,
    fit_lmm,
    fit_all_outcomes,
    get_residuals,
    get_fitted_values,
    get_random_effects,
    compare_models,
)

# Post-hoc comparisons - TypedDict and functions
from .posthoc import (
    ContrastResult,
    create_contrast_result,
    summarize_contrast_result,
    compute_emmeans,
    pairwise_contrasts,
    compute_effect_size,
    test_linear_trend,
)

# Multiplicity corrections
from .multiplicity import (
    CorrectionMethod,
    adjust_pvalues,
    apply_fdr,
    apply_holm,
    apply_holm_hypotheses,
    significant_outcomes,
    fdr_summary,
)

# Diagnostics - TypedDict and functions
from .diagnostics import (
    DiagnosticsResult,
    create_diagnostics_result,
    summarize_diagnostics,
    residual_diagnostics,
    check_assumptions,
    get_diagnostic_data,
)

# Reporting
from .report import (
    descriptive_table,
    descriptive_table_formatted,
    coefficient_table,
    coefficient_table_multiple,
    results_summary,
    export_to_csv,
    export_to_latex,
    print_results_summary,
    print_coefficient_summary,
)

__all__ = [
    # Version
    "__version__",
    
    # Registry - Enums
    "OutcomeType",
    "AnalysisLevel", 
    "TransformType",
    
    # Registry - TypedDict and functions
    "OutcomeInfo",
    "create_outcome_info",
    "OUTCOME_REGISTRY",
    "get_outcome_info",
    "register_outcome",
    "list_outcomes",
    "get_primary_outcomes",
    "get_continuous_outcomes",
    "get_proportion_outcomes",
    "get_count_outcomes",
    "get_questionnaire_outcomes",
    "get_emg_outcomes",
    "get_daily_outcomes",
    "get_single_instance_outcomes",
    "reset_registry",
    
    # Prepare - TypedDict and functions
    "AnalysisDataset",
    "create_analysis_dataset",
    "validate_dataset",
    "describe_dataset",
    "subset_dataset",
    "get_n_subjects",
    "get_n_observations",
    "get_date_range",
    "get_obs_per_subject",
    # Core preparation (from pre-extracted DataFrame)
    "prepare_from_dataframe",
    # Generic preparation (from profiles)
    "prepare_sensor_data",
    # Discovery functions
    "discover_sensors",
    "discover_questionnaires",
    "get_profile_summary",
    # EMG convenience wrappers
    "prepare_daily_emg",
    "prepare_weekly_emg",
    "prepare_daily_metrics",
    "prepare_single_instance_metrics",
    "add_subject_metadata",
    "aggregate_daily_to_subject",
    # Questionnaire preparation
    "prepare_daily_questionnaires",
    "prepare_baseline_questionnaires",
    "prepare_daily_workload",
    "prepare_daily_pain",
    # Utilities
    "compute_composite_score",
    "align_sensor_questionnaire",
    "parse_date",
    
    # Descriptive - TypedDicts and functions
    "NormalityResult",
    "VarianceCheckResult",
    "summarize_outcomes",
    "check_normality",
    "check_variance",
    "get_non_degenerate_outcomes",
    "missingness_report",
    "print_missingness_summary",
    
    # LMM - TypedDict and functions
    "LMMResult",
    "create_lmm_result",
    "summarize_lmm_result",
    "apply_transform",
    "fit_lmm",
    "fit_all_outcomes",
    "get_residuals",
    "get_fitted_values",
    "get_random_effects",
    "compare_models",
    
    # Post-hoc - TypedDict and functions
    "ContrastResult",
    "create_contrast_result",
    "summarize_contrast_result",
    "compute_emmeans",
    "pairwise_contrasts",
    "compute_effect_size",
    "test_linear_trend",
    
    # Multiplicity
    "CorrectionMethod",
    "adjust_pvalues",
    "apply_fdr",
    "apply_holm",
    "apply_holm_hypotheses",
    "significant_outcomes",
    "fdr_summary",
    
    # Diagnostics - TypedDict and functions
    "DiagnosticsResult",
    "create_diagnostics_result",
    "summarize_diagnostics",
    "residual_diagnostics",
    "check_assumptions",
    "get_diagnostic_data",
    
    # Reporting
    "descriptive_table",
    "descriptive_table_formatted",
    "coefficient_table",
    "coefficient_table_multiple",
    "results_summary",
    "export_to_csv",
    "export_to_latex",
    "print_results_summary",
    "print_coefficient_summary",
]
