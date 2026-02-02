"""
H5: Physiological and Postural Predictors of EMG Activity
=========================================================

Research Question
-----------------
Do physiological and environmental factors predict trapezius EMG activity?

Background
----------
Muscle tension (EMG) may be influenced by multiple factors beyond job type:

1. CARDIOVASCULAR STRAIN (hr_ratio_mean)
   - Elevated heart rate indicates physiological stress
   - Stress → muscle tension (fight-or-flight response)
   - May reflect workload demands

2. ENVIRONMENTAL NOISE (noise_mean)
   - Noise is a known occupational stressor
   - Can trigger startle responses and sustained muscle tension
   - May interfere with concentration → compensatory effort

3. POSTURAL STABILITY (posture_95_confidence_ellipse_area) ⭐ PRIMARY
   - Larger sway area = less postural stability
   - May indicate fatigue or discomfort
   - Postural instability could require compensatory muscle activation
   - Poor seated posture → increased trapezius load

Hypothesis
----------
H5 (Exploratory): Daily physiological and postural factors will predict
EMG p90 activity, with posture sway area being the primary predictor
of interest.

Statistical Model
-----------------
Linear Mixed Model with within-between decomposition:

    emg_apdf_active_p90 ~ hr_ratio_mean_within + hr_ratio_mean_between
                        + noise_mean_within + noise_mean_between
                        + posture_95_confidence_ellipse_area_within
                        + posture_95_confidence_ellipse_area_between
                        + work_type + day_index + (1|subject_id)

Where:
- emg_apdf_active_p90: Trapezius muscle activity (outcome)
- *_within: Day-to-day deviation from each subject's own mean
- *_between: Subject's mean deviation from the grand mean
- work_type: Covariate to control for FO/BO group differences
- day_index: Day of measurement (covariate)
- (1|subject_id): Random intercepts for between-person differences

About Posture 95% Confidence Ellipse Area
-----------------------------------------
This metric from the posturography assessment represents:
- The area containing 95% of center-of-pressure movements
- Measured in cm² or similar units
- LARGER area = MORE sway = LESS stable posture

Physiological interpretation:
- Increased sway may indicate muscular fatigue
- Poor seated balance requires compensatory stabilization
- Trunk instability → increased upper trapezius activation

Why Exploratory?
----------------
This hypothesis is exploratory because:
1. Multiple predictors increase Type I error risk
2. Relationships may be bidirectional (EMG ↔ posture)
3. Effect sizes are unknown from literature
4. Primary interest (posture) is hypothesis-generating

Interpretation
--------------
For each predictor:
- β > 0: Higher predictor value → higher EMG
- β < 0: Higher predictor value → lower EMG
- p < 0.05: Statistically significant association

Primary focus: posture_95_confidence_ellipse_area coefficient
- If β > 0 and p < 0.05: Postural instability predicts higher muscle tension
- Could suggest: postural training might reduce trapezius load

Effect Size Considerations
--------------------------
- Standardized coefficients for comparing predictor importance
- Partial R² for each predictor's unique contribution
- Consider multicollinearity if predictors are correlated

Why This Matters
----------------
Identifying modifiable predictors of EMG activity informs interventions:
- If HR predicts EMG → stress management interventions
- If noise predicts EMG → workplace acoustic improvements
- If posture predicts EMG → ergonomic/postural training

Usage
-----
    from hypotheses import run_hypothesis
    result = run_hypothesis("H5", profiles)
    
    # Check all predictor coefficients
    if result.model_result:
        print(result.model_result["coefficients"])

Notes
-----
- Temporality: Predictors from same day as EMG outcome
- Missing data: Posture measured once/day; may limit sample
- Confounders: Consider adding work_type as covariate
- Follow-up: If significant, consider mediation analysis
"""

from typing import Dict, Any
from .runner import run_hypothesis as _run
from .config import get_hypothesis

CONFIG = get_hypothesis("H5")


def run(profiles: Dict[str, dict], verbose: bool = True) -> Any:
    """Run H5 hypothesis test."""
    return _run("H5", profiles, verbose=verbose)


def describe() -> str:
    """Return the full hypothesis description."""
    return __doc__
