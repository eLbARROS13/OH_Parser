"""
H4: OSPAQ Questionnaire Validation
==================================

Research Question
-----------------
Does self-reported occupational sitting time (OSPAQ) correlate with
objectively measured sitting (HAR accelerometer)?

Background
----------
The Occupational Sitting and Physical Activity Questionnaire (OSPAQ) is
widely used in epidemiological studies because it's:
- Low cost
- Easy to administer
- Can be used in large populations

However, self-report measures are subject to:
- Recall bias
- Social desirability bias
- Difficulty estimating time accurately

Validating OSPAQ against objective measures is essential for:
- Interpreting existing epidemiological studies
- Informing future study design decisions
- Understanding measurement error magnitude

Hypothesis
----------
H4: Self-reported sitting percentage (OSPAQ) will positively correlate
    with objectively measured sitting proportion (HAR), indicating
    acceptable criterion validity.

Statistical Model
-----------------
Ordinary Least Squares (subject-level) with logit-transformed outcome:

    logit(har_sentado_prop_mean) ~ ospaq_sitting_frac + work_type

Where:
- har_sentado_prop_mean: Subject's mean sitting proportion across all days
- ospaq_sitting_frac: Self-reported sitting as fraction [0, 1]
- work_type: Covariate controlling for job type differences

Why Subject-Level (OLS)?
------------------------
OSPAQ is a single-instance questionnaire (one response per subject),
so we cannot use repeated measures. We aggregate daily HAR to subject
means for comparison.

Interpretation
--------------
- β > 0 for OSPAQ: Positive association (expected)
- p < 0.05: Significant correlation
- Coefficients are on the log-odds scale (back-transform for proportions)
- R² value: Proportion of variance explained

Typical Validity Thresholds:
- r > 0.70: Excellent validity
- r = 0.50-0.70: Good validity
- r = 0.30-0.50: Moderate validity
- r < 0.30: Poor validity

Expected Result
---------------
Literature suggests moderate correlation (r ≈ 0.40-0.60) between
self-report and objective sitting measures.

Why This Matters
----------------
If OSPAQ validation succeeds:
- Supports using OSPAQ in larger studies where sensors aren't feasible
- Quantifies expected measurement error for sample size calculations
- Enables comparison with previous OSPAQ-based research

If validation fails:
- Suggests OSPAQ may not be appropriate for this population
- Highlights need for objective measurement in future studies

Usage
-----
    from hypotheses import run_hypothesis
    result = run_hypothesis("H4", profiles)
    
    # Access R² from OLS summary if needed
    if result.model_result:
        print(result.model_result["ols_summary"])

Notes
-----
- Sample size: One observation per subject limits power
- Consider: Bland-Altman plots for agreement analysis
- Limitation: Can only validate at person-level, not day-level
"""

from typing import Dict, Any
from .runner import run_hypothesis as _run
from .config import get_hypothesis

CONFIG = get_hypothesis("H4")


def run(profiles: Dict[str, dict], verbose: bool = True) -> Any:
    """Run H4 hypothesis test."""
    return _run("H4", profiles, verbose=verbose)


def describe() -> str:
    """Return the full hypothesis description."""
    return __doc__
