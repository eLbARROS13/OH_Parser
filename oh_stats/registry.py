"""
Outcome Type Registry
=====================

Maps outcomes to their appropriate statistical model families, transforms,
and analysis levels. This is the core dispatch mechanism that makes the
pipeline outcome-type aware rather than assuming everything is Gaussian.

Model Families:
- continuous: LMM (Gaussian) with optional transforms
- ordinal: Ordered logistic mixed model (future: PyMC)
- proportion: Beta/logit-transformed LMM for bounded [0,1] outcomes
- count: Poisson/NegBin for event counts (future: GLMM)

Architecture Note:
    This module uses dictionaries instead of classes for data structures
    to maintain consistency with the oh_parser project style.

Usage:
    from oh_stats.registry import get_outcome_info, OutcomeType
    
    info = get_outcome_info("EMG_intensity.mean_percent_mvc")
    if info["outcome_type"] == OutcomeType.CONTINUOUS:
        result = fit_lmm(...)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Literal
from types import SimpleNamespace
import warnings


# =============================================================================
# Enums (Type constants - these are acceptable)
# =============================================================================

OutcomeTypeValue = Literal[
    "continuous",
    "ordinal",
    "proportion",
    "count",
    "binary",
    "unknown",
]

AnalysisLevelValue = Literal[
    "session",
    "daily",
    "weekly",
    "single",
]

TransformTypeValue = Literal[
    "none",
    "log",
    "log1p",
    "sqrt",
    "logit",
    "arcsine",
]

# Namespace-style constants (no class definitions)
OutcomeType = SimpleNamespace(
    CONTINUOUS="continuous",
    ORDINAL="ordinal",
    PROPORTION="proportion",
    COUNT="count",
    BINARY="binary",
    UNKNOWN="unknown",
)

AnalysisLevel = SimpleNamespace(
    SESSION="session",
    DAILY="daily",
    WEEKLY="weekly",
    SINGLE="single",
)

TransformType = SimpleNamespace(
    NONE="none",
    LOG="log",
    LOG1P="log1p",
    SQRT="sqrt",
    LOGIT="logit",
    ARCSINE="arcsine",
)


# =============================================================================
# OutcomeInfo (dict)
# =============================================================================

OutcomeInfo = Dict[str, Any]


def create_outcome_info(
    name: str,
    outcome_type: OutcomeTypeValue = "continuous",
    level: AnalysisLevelValue = "daily",
    transform: TransformTypeValue = "none",
    description: str = "",
    unit: str = "",
    valid_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    is_primary: bool = False,
    sensor: str = "emg",
    requires_both_sides: bool = False,
) -> OutcomeInfo:
    """
    Create an OutcomeInfo dictionary with all metadata.
    
    :param name: Outcome variable name (dot-notation path)
    :param outcome_type: Statistical model family
    :param level: Temporal aggregation level
    :param transform: Recommended transform (can be overridden)
    :param description: Human-readable description
    :param unit: Measurement unit (e.g., "%MVC", "seconds", "count")
    :param valid_range: (min, max) tuple for sanity checks
    :param is_primary: Whether this is a primary (vs exploratory) outcome
    :param sensor: Source sensor (e.g., "emg", "heart_rate", "questionnaire")
    :param requires_both_sides: Whether left/right must both exist
    :returns: OutcomeInfo dictionary
    """
    return {
        "name": name,
        "outcome_type": outcome_type,
        "level": level,
        "transform": transform,
        "description": description,
        "unit": unit,
        "valid_range": valid_range,
        "is_primary": is_primary,
        "sensor": sensor,
        "requires_both_sides": requires_both_sides,
    }


# =============================================================================
# DEFAULT OUTCOME REGISTRY (EMG metrics)
# =============================================================================

_DEFAULT_EMG_OUTCOMES: Dict[str, OutcomeInfo] = {
    # --- EMG Intensity metrics ---
    "EMG_intensity.mean_percent_mvc": create_outcome_info(
        name="EMG_intensity.mean_percent_mvc",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,  # Often roughly normal
        description="Mean muscle activation as percentage of MVC",
        unit="%MVC",
        valid_range=(0, 500),  # Can exceed 100% for brief peaks
        is_primary=True,
        sensor="emg",
    ),
    "EMG_intensity.max_percent_mvc": create_outcome_info(
        name="EMG_intensity.max_percent_mvc",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG,  # Often right-skewed
        description="Maximum muscle activation as percentage of MVC",
        unit="%MVC",
        valid_range=(0, 1000),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_intensity.min_percent_mvc": create_outcome_info(
        name="EMG_intensity.min_percent_mvc",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG1P,
        description="Minimum muscle activation as percentage of MVC",
        unit="%MVC",
        valid_range=(0, 100),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_intensity.iemg_percent_seconds": create_outcome_info(
        name="EMG_intensity.iemg_percent_seconds",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG,  # Cumulative, often skewed
        description="Integrated EMG (cumulative activation × time)",
        unit="%MVC·s",
        valid_range=(0, None),
        is_primary=False,
        sensor="emg",
    ),
    
    # --- EMG APDF (Amplitude Probability Distribution Function) ---
    "EMG_apdf.active.p10": create_outcome_info(
        name="EMG_apdf.active.p10",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="10th percentile of activation during active periods",
        unit="%MVC",
        valid_range=(0, 100),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_apdf.active.p50": create_outcome_info(
        name="EMG_apdf.active.p50",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Median activation during active periods",
        unit="%MVC",
        valid_range=(0, 100),
        is_primary=True,  # Key outcome
        sensor="emg",
    ),
    "EMG_apdf.active.p90": create_outcome_info(
        name="EMG_apdf.active.p90",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="90th percentile of activation during active periods",
        unit="%MVC",
        valid_range=(0, 200),
        is_primary=False,
        sensor="emg",
    ),
    
    # --- EMG Rest/Recovery metrics ---
    "EMG_rest_recovery.rest_percent": create_outcome_info(
        name="EMG_rest_recovery.rest_percent",
        outcome_type=OutcomeType.PROPORTION,  # Bounded [0, 1] as fraction
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,  # For proportions
        description="Percentage of time in rest state",
        unit="%",
        valid_range=(0, 100),
        is_primary=False,  # Often near-zero, may be degenerate
        sensor="emg",
    ),
    "EMG_rest_recovery.gap_frequency_per_minute": create_outcome_info(
        name="EMG_rest_recovery.gap_frequency_per_minute",
        outcome_type=OutcomeType.CONTINUOUS,  # Rate, treat as continuous
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG1P,
        description="Frequency of rest gaps per minute",
        unit="gaps/min",
        valid_range=(0, None),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_rest_recovery.max_sustained_activity_s": create_outcome_info(
        name="EMG_rest_recovery.max_sustained_activity_s",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG,  # Duration, often skewed
        description="Maximum sustained activity duration",
        unit="seconds",
        valid_range=(0, None),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_rest_recovery.gap_count": create_outcome_info(
        name="EMG_rest_recovery.gap_count",
        outcome_type=OutcomeType.COUNT,  # Integer count
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,  # Use appropriate count model
        description="Number of rest gaps",
        unit="count",
        valid_range=(0, None),
        is_primary=False,
        sensor="emg",
    ),
    
    # --- EMG Relative Bins ---
    "EMG_relative_bins.below_usual_pct": create_outcome_info(
        name="EMG_relative_bins.below_usual_pct",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,
        description="Percentage of time below usual activation level",
        unit="%",
        valid_range=(0, 100),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_relative_bins.typical_low_pct": create_outcome_info(
        name="EMG_relative_bins.typical_low_pct",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,
        description="Percentage of time in typical-low activation",
        unit="%",
        valid_range=(0, 100),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_relative_bins.typical_high_pct": create_outcome_info(
        name="EMG_relative_bins.typical_high_pct",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,
        description="Percentage of time in typical-high activation",
        unit="%",
        valid_range=(0, 100),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_relative_bins.high_for_you_pct": create_outcome_info(
        name="EMG_relative_bins.high_for_you_pct",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,
        description="Percentage of time in high-for-individual activation",
        unit="%",
        valid_range=(0, 100),
        is_primary=False,
        sensor="emg",
    ),
    
    # --- EMG Session metrics ---
    "EMG_session.duration_s": create_outcome_info(
        name="EMG_session.duration_s",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG,
        description="Total recording duration",
        unit="seconds",
        valid_range=(0, None),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_session.active_duration_s": create_outcome_info(
        name="EMG_session.active_duration_s",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG,
        description="Active recording duration",
        unit="seconds",
        valid_range=(0, None),
        is_primary=False,
        sensor="emg",
    ),
    "EMG_session.session_count": create_outcome_info(
        name="EMG_session.session_count",
        outcome_type=OutcomeType.COUNT,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Number of recording sessions",
        unit="count",
        valid_range=(1, None),
        is_primary=False,
        sensor="emg",
    ),
}


# =============================================================================
# DEFAULT QUESTIONNAIRE OUTCOMES
# =============================================================================

_DEFAULT_QUESTIONNAIRE_OUTCOMES: Dict[str, OutcomeInfo] = {
    # --- COPSOQ II Dimension Scores (0-100, treat as continuous) ---
    "copsoq.Exigências Quantitativas": create_outcome_info(
        name="copsoq.Exigências Quantitativas",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="COPSOQ: Quantitative demands",
        unit="score 0-100",
        valid_range=(0, 100),
        is_primary=False,
        sensor="questionnaire",
    ),
    "copsoq.Ritmo de Trabalho": create_outcome_info(
        name="copsoq.Ritmo de Trabalho",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="COPSOQ: Work pace",
        unit="score 0-100",
        valid_range=(0, 100),
        is_primary=False,
        sensor="questionnaire",
    ),
    "copsoq.Burnout": create_outcome_info(
        name="copsoq.Burnout",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="COPSOQ: Burnout",
        unit="score 0-100",
        valid_range=(0, 100),
        is_primary=True,
        sensor="questionnaire",
    ),
    "copsoq.Stress": create_outcome_info(
        name="copsoq.Stress",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="COPSOQ: Stress",
        unit="score 0-100",
        valid_range=(0, 100),
        is_primary=True,
        sensor="questionnaire",
    ),
    "copsoq.Satisfação Laboral": create_outcome_info(
        name="copsoq.Satisfação Laboral",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="COPSOQ: Job satisfaction",
        unit="score 0-100",
        valid_range=(0, 100),
        is_primary=False,
        sensor="questionnaire",
    ),
    
    # --- MUEQ Domain Scores (continuous by aggregation) ---
    "mueq.Autonomia": create_outcome_info(
        name="mueq.Autonomia",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="MUEQ: Autonomy",
        unit="score 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="questionnaire",
    ),
    "mueq.Qualidade das Pausas": create_outcome_info(
        name="mueq.Qualidade das Pausas",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="MUEQ: Break quality",
        unit="score 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="questionnaire",
    ),
    
    # --- ROSA (Rapid Office Strain Assessment) ---
    "ROSA_final": create_outcome_info(
        name="ROSA_final",
        outcome_type=OutcomeType.ORDINAL,  # Discrete 1-10, quasi-continuous
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="ROSA: Final ergonomic risk score",
        unit="score 1-10",
        valid_range=(1, 10),
        is_primary=True,
        sensor="questionnaire",
    ),
    "ROSA_final_normalized": create_outcome_info(
        name="ROSA_final_normalized",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,
        description="ROSA: Normalized final score",
        unit="proportion 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="questionnaire",
    ),
    
    # --- IPAQ-SF (Physical Activity) ---
    "ipaq.total_met": create_outcome_info(
        name="ipaq.total_met",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOG1P,  # Highly skewed
        description="IPAQ: Total MET-minutes/week",
        unit="MET-min/week",
        valid_range=(0, None),
        is_primary=False,
        sensor="questionnaire",
    ),
    "ipaq.category": create_outcome_info(
        name="ipaq.category",
        outcome_type=OutcomeType.ORDINAL,  # Low/Moderate/High
        level=AnalysisLevel.SINGLE,
        transform=TransformType.NONE,
        description="IPAQ: Activity category",
        unit="category",
        valid_range=None,
        is_primary=False,
        sensor="questionnaire",
    ),
    
    # --- OSPAQ (Occupational Sitting and Physical Activity) ---
    "ospaq.percentagem_sentado": create_outcome_info(
        name="ospaq.percentagem_sentado",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,  # Compositional, but analyzed separately
        description="OSPAQ: Percentage of time sitting",
        unit="proportion 0-1",
        valid_range=(0, 100),  # Stored as percentage, convert to 0-1
        is_primary=False,
        sensor="questionnaire",
    ),
    "ospaq.percentagem_pe": create_outcome_info(
        name="ospaq.percentagem_pe",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,
        description="OSPAQ: Percentage of time standing",
        unit="proportion 0-1",
        valid_range=(0, 100),
        is_primary=False,
        sensor="questionnaire",
    ),
    "ospaq.percentagem_caminhar": create_outcome_info(
        name="ospaq.percentagem_caminhar",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,
        description="OSPAQ: Percentage of time walking",
        unit="proportion 0-1",
        valid_range=(0, 100),
        is_primary=False,
        sensor="questionnaire",
    ),
    
    # --- Pain (NPRS 0-10) ---
    "pain.intensity": create_outcome_info(
        name="pain.intensity",
        outcome_type=OutcomeType.ORDINAL,  # 0-10 scale, often treated as continuous
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="NPRS: Pain intensity",
        unit="score 0-10",
        valid_range=(0, 10),
        is_primary=True,
        sensor="questionnaire",
    ),
    
    # --- Daily Workload Questionnaire (Likert 1-5) ---
    "workload.focus_and_mental_strain": create_outcome_info(
        name="workload.focus_and_mental_strain",
        outcome_type=OutcomeType.ORDINAL,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily workload: Focus and mental strain",
        unit="Likert 1-5",
        valid_range=(1, 5),
        is_primary=False,
        sensor="questionnaire",
    ),
    "workload.rushed_and_under_pressure": create_outcome_info(
        name="workload.rushed_and_under_pressure",
        outcome_type=OutcomeType.ORDINAL,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily workload: Rushed and under pressure",
        unit="Likert 1-5",
        valid_range=(1, 5),
        is_primary=False,
        sensor="questionnaire",
    ),
    "workload.frequent_interruptions": create_outcome_info(
        name="workload.frequent_interruptions",
        outcome_type=OutcomeType.ORDINAL,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily workload: Frequent interruptions",
        unit="Likert 1-5",
        valid_range=(1, 5),
        is_primary=False,
        sensor="questionnaire",
    ),
    "workload.more_effort_than_resources": create_outcome_info(
        name="workload.more_effort_than_resources",
        outcome_type=OutcomeType.ORDINAL,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily workload: More effort than resources",
        unit="Likert 1-5",
        valid_range=(1, 5),
        is_primary=False,
        sensor="questionnaire",
    ),
    "workload.heavy_workload": create_outcome_info(
        name="workload.heavy_workload",
        outcome_type=OutcomeType.ORDINAL,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily workload: Heavy workload",
        unit="Likert 1-5",
        valid_range=(1, 5),
        is_primary=False,
        sensor="questionnaire",
    ),
    
    # --- Environmental Quality ---
    "environmental.Nível de Iluminação": create_outcome_info(
        name="environmental.Nível de Iluminação",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,
        description="Environmental: Illumination level satisfaction",
        unit="proportion 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="questionnaire",
    ),
    "environmental.Ruído": create_outcome_info(
        name="environmental.Ruído",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,
        description="Environmental: Noise satisfaction",
        unit="proportion 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="questionnaire",
    ),
}


# =============================================================================
# PROJECT-SPECIFIC (underscore) OUTCOMES
# =============================================================================

_PROJECT_OUTCOMES: Dict[str, OutcomeInfo] = {
    # EMG APDF (underscore naming from prepare.py)
    "emg_apdf_active_p90": create_outcome_info(
        name="emg_apdf_active_p90",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="90th percentile of activation during active periods",
        unit="fraction of %MVC (0-1)",
        valid_range=(0, 1),
        is_primary=True,
        sensor="emg",
    ),
    "emg_apdf_active_p90_log": create_outcome_info(
        name="emg_apdf_active_p90_log",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG,
        description="Log-transformed EMG p90 (fraction of %MVC)",
        unit="log(fraction of %MVC)",
        valid_range=(None, None),
        is_primary=False,
        sensor="emg",
    ),
    "emg_apdf_active_p50": create_outcome_info(
        name="emg_apdf_active_p50",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Median activation during active periods",
        unit="fraction of %MVC (0-1)",
        valid_range=(0, 1),
        is_primary=False,
        sensor="emg",
    ),
    # Daily questionnaire/workload
    "workload_mean": create_outcome_info(
        name="workload_mean",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily perceived workload (mean score)",
        unit="score",
        valid_range=(None, None),
        is_primary=False,
        sensor="questionnaire",
    ),
    # HAR sitting proportion (daily)
    "har_sentado_prop": create_outcome_info(
        name="har_sentado_prop",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,
        description="Daily sitting proportion from HAR",
        unit="proportion 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="har",
    ),
    "har_sentado_prop_logit": create_outcome_info(
        name="har_sentado_prop_logit",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOGIT,
        description="Logit-transformed sitting proportion",
        unit="log-odds",
        valid_range=(None, None),
        is_primary=False,
        sensor="har",
    ),
    # HAR durations (daily)
    "har_total_duration_sec": create_outcome_info(
        name="har_total_duration_sec",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG1P,
        description="Total monitored duration (seconds)",
        unit="seconds",
        valid_range=(0, None),
        is_primary=False,
        sensor="har",
    ),
    "har_sentado_duration_sec": create_outcome_info(
        name="har_sentado_duration_sec",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.LOG1P,
        description="Sitting duration (seconds)",
        unit="seconds",
        valid_range=(0, None),
        is_primary=False,
        sensor="har",
    ),
    # Heart rate
    "hr_ratio_mean": create_outcome_info(
        name="hr_ratio_mean",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily mean HR ratio",
        unit="ratio",
        valid_range=(0, None),
        is_primary=False,
        sensor="heart_rate",
    ),
    "hr_ratio_std": create_outcome_info(
        name="hr_ratio_std",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily SD HR ratio",
        unit="ratio",
        valid_range=(0, None),
        is_primary=False,
        sensor="heart_rate",
    ),
    # Noise
    "noise_mean": create_outcome_info(
        name="noise_mean",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily mean noise level",
        unit="dB",
        valid_range=(None, None),
        is_primary=False,
        sensor="noise",
    ),
    "noise_std": create_outcome_info(
        name="noise_std",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Daily SD noise level",
        unit="dB",
        valid_range=(0, None),
        is_primary=False,
        sensor="noise",
    ),
    # Posture
    "posture_95_confidence_ellipse_area": create_outcome_info(
        name="posture_95_confidence_ellipse_area",
        outcome_type=OutcomeType.CONTINUOUS,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description="Posture sway area (95% confidence ellipse)",
        unit="cm^2",
        valid_range=(0, None),
        is_primary=False,
        sensor="posture",
    ),
    # OSPAQ sitting (subject-level)
    "ospaq_sitting_frac": create_outcome_info(
        name="ospaq_sitting_frac",
        outcome_type=OutcomeType.PROPORTION,
        level=AnalysisLevel.SINGLE,
        transform=TransformType.LOGIT,
        description="OSPAQ self-reported sitting proportion",
        unit="proportion 0-1",
        valid_range=(0, 1),
        is_primary=False,
        sensor="questionnaire",
    ),
}


# Global mutable registry (populated from defaults)
OUTCOME_REGISTRY: Dict[str, OutcomeInfo] = {
    **_DEFAULT_EMG_OUTCOMES,
    **_DEFAULT_QUESTIONNAIRE_OUTCOMES,
    **_PROJECT_OUTCOMES,
}


# =============================================================================
# Registry API
# =============================================================================

def get_outcome_info(name: str) -> OutcomeInfo:
    """
    Get outcome metadata from registry.
    
    If outcome is not registered, returns a default UNKNOWN entry with a warning.
    
    :param name: Outcome variable name (dot-notation)
    :returns: OutcomeInfo dict with type, transform, and metadata
    """
    if name in OUTCOME_REGISTRY:
        return OUTCOME_REGISTRY[name]
    
    # Unknown outcome - return default with warning
    warnings.warn(
        f"Outcome '{name}' not in registry. Using default CONTINUOUS type. "
        f"Consider registering it with register_outcome() for proper handling.",
        UserWarning,
    )
    return create_outcome_info(
        name=name,
        outcome_type=OutcomeType.UNKNOWN,
        level=AnalysisLevel.DAILY,
        transform=TransformType.NONE,
        description=f"Unregistered outcome: {name}",
    )


def register_outcome(
    name: str,
    outcome_type: OutcomeType = OutcomeType.CONTINUOUS,
    level: AnalysisLevel = AnalysisLevel.DAILY,
    transform: TransformType = TransformType.NONE,
    description: str = "",
    unit: str = "",
    valid_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    is_primary: bool = False,
    sensor: str = "unknown",
    requires_both_sides: bool = False,
    overwrite: bool = False,
) -> OutcomeInfo:
    """
    Register a new outcome in the registry.
    
    :param name: Outcome variable name (dot-notation)
    :param outcome_type: Statistical model family
    :param level: Temporal aggregation level
    :param transform: Recommended transform
    :param description: Human-readable description
    :param unit: Measurement unit
    :param valid_range: (min, max) for sanity checks
    :param is_primary: Whether primary outcome (affects multiplicity)
    :param sensor: Source sensor
    :param requires_both_sides: Whether both sides must exist
    :param overwrite: Allow overwriting existing entry
    :returns: The registered OutcomeInfo dict
    """
    if name in OUTCOME_REGISTRY and not overwrite:
        raise ValueError(
            f"Outcome '{name}' already registered. Use overwrite=True to replace."
        )
    
    info = create_outcome_info(
        name=name,
        outcome_type=outcome_type,
        level=level,
        transform=transform,
        description=description,
        unit=unit,
        valid_range=valid_range,
        is_primary=is_primary,
        sensor=sensor,
        requires_both_sides=requires_both_sides,
    )
    OUTCOME_REGISTRY[name] = info
    return info


def list_outcomes(
    outcome_type: Optional[OutcomeType] = None,
    level: Optional[AnalysisLevel] = None,
    sensor: Optional[str] = None,
    primary_only: bool = False,
) -> List[str]:
    """
    List registered outcomes, optionally filtered.
    
    :param outcome_type: Filter by model family
    :param level: Filter by analysis level
    :param sensor: Filter by sensor source
    :param primary_only: Only return primary outcomes
    :returns: List of outcome names matching criteria
    """
    results = []
    for name, info in OUTCOME_REGISTRY.items():
        if outcome_type is not None and info["outcome_type"] != outcome_type:
            continue
        if level is not None and info["level"] != level:
            continue
        if sensor is not None and info["sensor"] != sensor:
            continue
        if primary_only and not info["is_primary"]:
            continue
        results.append(name)
    return sorted(results)


def get_primary_outcomes() -> List[str]:
    """Get list of primary outcome names."""
    return list_outcomes(primary_only=True)


def get_continuous_outcomes() -> List[str]:
    """Get list of continuous (LMM-suitable) outcome names."""
    return list_outcomes(outcome_type=OutcomeType.CONTINUOUS)


def get_proportion_outcomes() -> List[str]:
    """Get list of proportion [0,1] outcome names."""
    return list_outcomes(outcome_type=OutcomeType.PROPORTION)


def get_count_outcomes() -> List[str]:
    """Get list of count outcome names."""
    return list_outcomes(outcome_type=OutcomeType.COUNT)


def get_questionnaire_outcomes() -> List[str]:
    """Get list of questionnaire outcome names."""
    return list_outcomes(sensor="questionnaire")


def get_emg_outcomes() -> List[str]:
    """Get list of EMG outcome names."""
    return list_outcomes(sensor="emg")


def get_daily_outcomes() -> List[str]:
    """Get list of daily-level outcome names."""
    return list_outcomes(level=AnalysisLevel.DAILY)


def get_single_instance_outcomes() -> List[str]:
    """Get list of single-instance (baseline) outcome names."""
    return list_outcomes(level=AnalysisLevel.SINGLE)


def reset_registry() -> None:
    """Reset registry to default outcomes (EMG + questionnaire)."""
    global OUTCOME_REGISTRY
    OUTCOME_REGISTRY.clear()
    OUTCOME_REGISTRY.update(_DEFAULT_EMG_OUTCOMES)
    OUTCOME_REGISTRY.update(_DEFAULT_QUESTIONNAIRE_OUTCOMES)
    OUTCOME_REGISTRY.update(_PROJECT_OUTCOMES)
