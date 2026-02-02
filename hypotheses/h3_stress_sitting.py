"""
H3: Daily Workload Predicts Sitting Behavior
============================================

Research Question
-----------------
Does daily perceived workload predict daily sitting proportion?

Background
----------
The relationship between work demands and physical activity is complex:

Theory 1 (Compensation):
- High workload → fatigue → less movement → more sitting
- Workers stay at desk to "push through" demanding tasks

Theory 2 (Activity-Stress buffering):
- Movement breaks reduce stress
- Workers may take more breaks on stressful days

This hypothesis tests Theory 1 (most supported in literature).

Hypothesis
----------
H3: Higher daily workload will predict higher sitting proportion,
    as workers remain sedentary to complete demanding tasks.

Statistical Model
-----------------
Linear Mixed Model with logit-transformed outcome:

    logit(har_sentado_prop) ~ workload_mean + work_type + day_index + (1|subject_id)

Where:
- har_sentado_prop: Proportion of work time spent sitting [0, 1]
- logit transform: log(p / (1-p)) to handle bounded outcome
- workload_mean: Daily perceived workload (continuous predictor)
- (1|subject_id): Random intercepts

Why Logit Transform?
--------------------
Sitting proportion is bounded between 0 and 1:
- Linear model could predict values outside [0, 1]
- Logit maps (0,1) → (-∞, +∞) for linear modeling
- Coefficients interpret as change in log-odds

Interpretation
--------------
- β > 0 for workload: Higher workload → higher sitting (log-odds)
- Exponentiate coefficient for odds ratio interpretation
- Example: β = 0.15 means 10-unit increase in workload multiplies 
  odds of sitting by exp(0.15 × 10) = 4.48

Why This Matters
----------------
If workload drives sedentary behavior:
- Interventions should target high-workload periods
- Stress management may reduce sitting time
- Workplace redesign could break the workload-sitting link

Usage
-----
    from hypotheses import run_hypothesis
    result = run_hypothesis("H3", profiles)
    
    # Note: outcome is transformed to har_sentado_prop_logit internally

Notes
-----
- Data source: Workload from questionnaire, sitting from HAR sensor
- Temporal alignment: Both measures from same day
- Confounders: Day of week controlled; consider meeting schedules
"""

from typing import Dict, Any
from .runner import run_hypothesis as _run
from .config import get_hypothesis

CONFIG = get_hypothesis("H3")


def run(profiles: Dict[str, dict], verbose: bool = True) -> Any:
    """Run H3 hypothesis test."""
    return _run("H3", profiles, verbose=verbose)


def describe() -> str:
    """Return the full hypothesis description."""
    return __doc__
