"""
Hypothesis Configuration
========================

Declarative definitions for all study hypotheses (H1-H6).

This module centralizes hypothesis specifications to:
1. Eliminate code repetition across hypothesis files
2. Enable batch processing via the runner
3. Document the statistical design in one place

Each hypothesis is defined as a HypothesisConfig dict with:
- name: Short descriptive name
- description: Full scientific rationale
- level: "daily" (repeated measures) or "subject" (cross-sectional)
- outcome: Dependent variable column name
- predictors: List of fixed effect predictors
- model: "lmm" (Linear Mixed Model) or "ols" (Ordinary Least Squares)
- transform: Optional transformation for the outcome ("logit", "log", "sqrt")
- covariates: Additional covariates to control for
"""

from typing import Dict, Any, List, Optional, Literal


HypothesisConfig = Dict[str, Any]


HYPOTHESES: dict[str, HypothesisConfig] = {
    # =========================================================================
    # H1: FO vs BO on Trapezius EMG Activity
    # =========================================================================
    "H1": {
        "name": "FO vs BO on EMG p90",
        "description": """
        Do Front-Office (FO) and Back-Office (BO) workers differ in trapezius 
        muscle activity (EMG p90)?
        
        Rationale: FO workers may exhibit higher sustained low-level muscle 
        contractions due to computer-intensive tasks requiring static postures.
        
        Model: log(emg_apdf_active_p90) ~ work_type + day_index + (1|subject_id)
        
        Note: Log-transform applied to stabilize variance (heteroscedasticity
        detected in untransformed model - variance increases with mean).
        """,
        "level": "daily",
        "outcome": "emg_apdf_active_p90",
        "predictors": ["work_type"],
        "covariates": ["C(day_index)"],
        "model": "lmm",
        "transform": "log",  # Variance stabilization for heteroscedasticity
        "random_intercept": "subject_id",
        "primary_predictor": "work_type",
        "exploratory": False,
        "auto_correct": True,
        "use_lrt_primary": True,
        "bootstrap_on_violation": True,
        "bootstrap_iterations": 200,
    },
    
    # =========================================================================
    # H2: FO vs BO on Perceived Workload
    # =========================================================================
    "H2": {
        "name": "FO vs BO on Workload",
        "description": """
        Do Front-Office and Back-Office workers differ in daily perceived 
        workload (stress proxy)?
        
        Rationale: Different job demands may lead to different stress levels.
        FO workers with customer interactions may report higher workload.
        
        Model: workload_mean ~ work_type + day_index + (1|subject_id)
        """,
        "level": "daily",
        "outcome": "workload_mean",
        "predictors": ["work_type"],
        "covariates": ["C(day_index)"],
        "model": "lmm",
        "random_intercept": "subject_id",
        "primary_predictor": "work_type",
        "exploratory": False,
        "auto_correct": True,
        "use_lrt_primary": True,
        "bootstrap_on_violation": True,
        "bootstrap_iterations": 200,
    },
    
    # =========================================================================
    # H3: Daily Workload Predicts Sitting Behavior
    # =========================================================================
    "H3": {
        "name": "Workload → Sitting",
        "description": """
        Does daily perceived workload predict daily sitting proportion?
        
        Rationale: Higher workload days may lead to more sedentary behavior
        as workers remain at their desks to complete tasks.
        
        Model: logit(har_sentado_prop) ~ workload_mean + work_type + day_index + (1|subject_id)
        
        Note: Sitting proportion is logit-transformed since it's bounded [0,1].
        """,
        "level": "daily",
        "outcome": "har_sentado_prop",
        "predictors": ["workload_mean"],
        "covariates": ["work_type", "C(day_index)"],
        "model": "lmm",
        "transform": "logit",
        "random_intercept": "subject_id",
        "primary_predictor": "workload_mean",
        "exploratory": False,
        "auto_correct": True,
        "use_lrt_primary": True,
        "bootstrap_on_violation": True,
        "bootstrap_iterations": 200,
    },
    
    # =========================================================================
    # H4: Self-Reported vs Objective Sitting (Validation)
    # =========================================================================
    "H4": {
        "name": "OSPAQ Validation",
        "description": """
        Does self-reported sitting (OSPAQ questionnaire) correlate with 
        objectively measured sitting (HAR sensor)?
        
        Rationale: Validates the OSPAQ questionnaire against objective 
        accelerometer-derived sitting time. Important for epidemiological
        studies relying on self-report.
        
        Model: logit(har_sentado_prop_mean) ~ ospaq_sitting_frac + work_type
        
        Note: Subject-level analysis (one observation per person).
        Logit transform applied because outcome is a proportion [0,1].
        """,
        "level": "subject",
        "outcome": "har_sentado_prop",  # Aggregated to subject mean
        "predictors": ["ospaq_sitting_frac"],
        "covariates": ["work_type"],
        "model": "ols",
        "transform": "logit",  # Proportion requires logit transform
        "primary_predictor": "ospaq_sitting_frac",
        "exploratory": False,
        "auto_correct": True,
        "use_lrt_primary": False,
        "bootstrap_on_violation": False,
    },
    
    # =========================================================================
    # H5: Physiological Predictors of EMG (Exploratory)
    # =========================================================================
    "H5": {
        "name": "Physiological → EMG",
        "description": """
        Do physiological and environmental factors predict trapezius EMG activity?
        
        Predictors (with within-between decomposition):
        - Heart rate ratio (cardiovascular strain)
        - Noise level (environmental stressor)
        - Posture sway area (postural stability/fatigue)
        
        Within-Between Decomposition:
        - Within-subject: "On days when X is higher than this person's average..."
        - Between-subject: "People who generally have higher X..."
        
        Rationale: Exploratory analysis to identify which daily factors
        are associated with muscle tension. Decomposition separates 
        day-to-day fluctuations from stable individual differences.
        
        Model: emg_p90 ~ hr_within + hr_between + noise_within + noise_between 
                        + posture_within + posture_between + work_type + day_index + (1|subject_id)
        """,
        "level": "daily",
        "outcome": "emg_apdf_active_p90",
        "predictors": ["hr_ratio_mean", "noise_mean", "posture_95_confidence_ellipse_area"],
        "covariates": ["work_type", "C(day_index)"],
        "model": "lmm",
        "random_intercept": "subject_id",
        "primary_predictor": "posture_95_confidence_ellipse_area",  # Primary interest
        "exploratory": True,  # Excluded from multiplicity correction
        "decompose_predictors": True,  # Split into within/between components
        "auto_correct": True,
        "use_lrt_primary": True,
        "bootstrap_on_violation": True,
        "bootstrap_iterations": 200,
    },
    
    # =========================================================================
    # H6: FO vs BO on Postural Stability
    # =========================================================================
    "H6": {
        "name": "FO vs BO on Posture",
        "description": """
        Do Front-Office and Back-Office workers differ in postural stability
        (95% confidence ellipse area)?
        
        Rationale: Different work tasks may affect postural control differently.
        FO workers with prolonged sitting at computer workstations may show
        different sway patterns compared to BO workers with more varied tasks.
        
        Larger ellipse area = more postural sway = less stability
        
        Model: posture_95_confidence_ellipse_area ~ work_type + day_index + (1|subject_id)
        """,
        "level": "daily",
        "outcome": "posture_95_confidence_ellipse_area",
        "predictors": ["work_type"],
        "covariates": ["C(day_index)"],
        "model": "lmm",
        "random_intercept": "subject_id",
        "primary_predictor": "work_type",
        "exploratory": False,
        "auto_correct": True,
        "use_lrt_primary": True,
        "bootstrap_on_violation": True,
        "bootstrap_iterations": 200,
    },
}


def get_hypothesis(hypothesis_id: str) -> HypothesisConfig:
    """Get configuration for a specific hypothesis."""
    if hypothesis_id not in HYPOTHESES:
        raise ValueError(f"Unknown hypothesis: {hypothesis_id}. Available: {list(HYPOTHESES.keys())}")
    return HYPOTHESES[hypothesis_id]


def list_hypotheses() -> List[str]:
    """List all available hypothesis IDs."""
    return list(HYPOTHESES.keys())
