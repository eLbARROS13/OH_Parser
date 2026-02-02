"""
H1: FO vs BO on Trapezius EMG Activity
======================================

Research Question
-----------------
Do Front-Office (FO) and Back-Office (BO) workers differ in trapezius 
muscle activity during the work week?

Background
----------
Trapezius muscle activity, measured via surface EMG, is a key indicator of 
musculoskeletal load in office workers. Sustained low-level muscle contractions 
("Cinderella fibers") are associated with neck/shoulder pain development.

FO workers typically:
- Perform computer-intensive tasks
- Maintain static postures for extended periods
- Have customer interaction demands

BO workers typically:
- Have more varied tasks
- May have more movement opportunities
- Different cognitive demands

Hypothesis
----------
H1: FO workers will exhibit higher EMG p90 values compared to BO workers,
    reflecting greater sustained muscle activation.

Statistical Model
-----------------
Linear Mixed Model with:

    emg_apdf_active_p90 ~ work_type + day_index + (1|subject_id)

Where:
- emg_apdf_active_p90: 90th percentile of amplitude probability distribution
                       during active muscle use, stored as fraction of %MVC (0–1)
                       (multiply by 100 for %MVC reporting)
- work_type: Fixed effect (FO vs BO) - the effect of interest
- day_index: Fixed effect controlling for day-of-week effects
- (1|subject_id): Random intercepts for subjects

Why Random Intercepts?
----------------------
Each worker has their own "baseline" muscle activity level due to:
- Individual physiology (muscle fiber composition, body composition)
- Personal work habits
- Workstation setup differences

The random intercept captures: "Some people are just more tense than others"

Interpretation
--------------
- Coefficient for work_type[T.FO]: Difference in EMG p90 between FO and BO
- p < 0.05: Reject null hypothesis; work type is associated with muscle activity
- Effect size (Cohen's d): Practical significance of the difference

Expected Result
---------------
Based on literature: FO workers may show 10-20% higher EMG p90 values.

Example Output
--------------
    LMM: emg_apdf_active_p90 ~ work_type + C(day_index)
    
    Subjects: 38 | Observations: 161 | ICC: 0.47
    
    Coefficients:
      term               estimate    SE       z       p
      Intercept          12.45       1.23     10.12   <0.001
      work_type[T.FO]     3.82       1.45      2.64    0.008  ***
      C(day_index)[T.2]  -0.34       0.89     -0.38    0.704
      ...

Usage
-----
    from hypotheses import run_hypothesis
    from oh_parser import load_profiles
    
    profiles = load_profiles("/path/to/profiles")
    result = run_hypothesis("H1", profiles)
    
    print(f"p-value: {result.p_value:.4f}")
    print(f"Significant: {result.p_value < 0.05}")

References
----------
1. Veiersted KB, et al. (1993). Sustained low-level muscle activity and 
   musculoskeletal disorders. Ergonomics.
2. Hägg GM (1991). Static work loads and occupational myalgia. 
   European Journal of Applied Physiology.
"""

from typing import Dict, Any
from .runner import run_hypothesis as _run
from .config import get_hypothesis

# Export hypothesis configuration
CONFIG = get_hypothesis("H1")


def run(profiles: Dict[str, dict], verbose: bool = True) -> Any:
    """
    Run H1 hypothesis test.
    
    Args:
        profiles: Loaded OH profiles
        verbose: Print detailed output
    
    Returns:
        HypothesisResult object
    """
    return _run("H1", profiles, verbose=verbose)


def describe() -> str:
    """Return the full hypothesis description."""
    return __doc__
