"""
H6: FO vs BO on Postural Stability
==================================

Research Question
-----------------
Do Front-Office (FO) and Back-Office (BO) workers differ in postural 
stability during standing balance assessments?

Background
----------
Postural stability, measured via center-of-pressure analysis during 
standing, reflects neuromuscular control and can indicate:
- Fatigue accumulation
- Musculoskeletal discomfort
- Work-related physical strain

The 95% Confidence Ellipse Area
-------------------------------
This metric represents the area containing 95% of center-of-pressure
movements during a standing balance test.

Interpretation:
- SMALLER area = BETTER stability = less sway
- LARGER area = WORSE stability = more sway

Factors affecting sway area:
- Muscle fatigue (increases sway)
- Proprioceptive impairment (increases sway)
- Attention demands (can increase or decrease sway)
- Discomfort/pain (often increases sway)

Why FO vs BO Might Differ
-------------------------
FO workers:
- Prolonged static sitting at computer
- May develop postural muscle fatigue
- Less movement variety throughout day
- Hypothesis: May show INCREASED sway (worse stability)

BO workers:
- Potentially more varied physical tasks
- More opportunities for position changes
- Different postural loading patterns
- Hypothesis: May show DECREASED sway (better stability)

Hypothesis
----------
H6: FO and BO workers will differ in postural stability (95% confidence
    ellipse area), with FO workers potentially showing larger sway area
    indicating reduced postural control.

Statistical Model
-----------------
Linear Mixed Model:

    posture_95_confidence_ellipse_area ~ work_type + day_index + (1|subject_id)

Where:
- posture_95_confidence_ellipse_area: Sway area in cm² (outcome)
- work_type: Fixed effect (FO vs BO) - effect of interest
- day_index: Fixed effect controlling for day-of-week patterns
- (1|subject_id): Random intercepts for individual differences

Why Random Intercepts?
----------------------
Individual differences in postural stability arise from:
- Age and physical fitness
- Previous injury/pain history
- Natural variation in balance ability
- Habitual posture patterns

Interpretation
--------------
- β > 0 for work_type[T.FO]: FO workers have LARGER sway (WORSE stability)
- β < 0 for work_type[T.FO]: FO workers have SMALLER sway (BETTER stability)
- p < 0.05: Job type is significantly associated with postural stability

Clinical Significance
---------------------
Typical normative values for young adults:
- 95% ellipse area: ~1-5 cm² (eyes open)
- Differences >20% between groups may be clinically meaningful

Why This Matters
----------------
If job type affects postural stability:
- Identifies workers at risk for balance-related issues
- Informs workplace ergonomic interventions
- May explain differential musculoskeletal complaint rates
- Suggests need for postural training programs

Connection to H5
----------------
H6 examines whether job type predicts posture (posture as OUTCOME)
H5 examines whether posture predicts EMG (posture as PREDICTOR)

Together, these hypotheses test:
- Is posture affected by work type?
- Does posture affect muscle tension?

This could reveal a pathway: Job Type → Postural Changes → Muscle Tension

Usage
-----
    from hypotheses import run_hypothesis
    result = run_hypothesis("H6", profiles)
    
    # Check effect direction
    if result.model_result:
        coeffs = result.model_result["coefficients"]
        fo_effect = coeffs[coeffs["term"].str.contains("work_type")]
        print(f"FO effect on sway: {fo_effect['estimate'].values[0]:.3f} cm²")

Notes
-----
- Measurement: Standing posturography (typically 30-60 seconds)
- Timing: Usually measured at start of shift (less fatigue)
- Consider: Adding time-of-day as covariate
- Limitation: Single daily measurement may miss intra-day changes
"""

from typing import Dict, Any
from .runner import run_hypothesis as _run
from .config import get_hypothesis

CONFIG = get_hypothesis("H6")


def run(profiles: Dict[str, dict], verbose: bool = True) -> Any:
    """Run H6 hypothesis test."""
    return _run("H6", profiles, verbose=verbose)


def describe() -> str:
    """Return the full hypothesis description."""
    return __doc__
