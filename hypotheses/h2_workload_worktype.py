"""
H2: FO vs BO on Perceived Workload
==================================

Research Question
-----------------
Do Front-Office (FO) and Back-Office (BO) workers differ in daily 
perceived workload (stress proxy)?

Background
----------
Perceived workload reflects cognitive demands, time pressure, and effort
required to complete work tasks. It serves as a proxy for occupational stress.

FO workers may experience:
- Direct customer interactions (emotional labor)
- Time pressure from service demands
- Unpredictable workflow

BO workers may experience:
- More predictable task flow
- Different cognitive demands (data processing, analysis)
- Less direct interpersonal demands

Hypothesis
----------
H2: FO and BO workers will differ in mean daily perceived workload,
    with FO workers potentially reporting higher levels due to
    customer-facing demands.

Statistical Model
-----------------
Linear Mixed Model:

    workload_mean ~ work_type + day_index + (1|subject_id)

Where:
- workload_mean: Daily mean workload score (e.g., from NASA-TLX or similar)
- work_type: Fixed effect (FO vs BO)
- day_index: Fixed effect controlling for day-of-week patterns
- (1|subject_id): Random intercepts for individual baseline differences

Why This Matters
----------------
If workload differs by job type, it could explain:
- Differences in muscle tension (H1)
- Differences in sedentary behavior (H3)
- Risk factors for burnout and musculoskeletal complaints

Interpretation
--------------
- p < 0.05 for work_type: Job type is associated with perceived workload
- Direction of coefficient: Which group reports higher workload
- Magnitude: Practical significance depends on scale range

Usage
-----
    from hypotheses import run_hypothesis
    result = run_hypothesis("H2", profiles)

Notes
-----
- Data source: Daily questionnaire responses
- Missing data: Common if workers don't complete daily surveys
- Consider: Workload may vary more within-person than between groups
"""

from typing import Dict, Any
from .runner import run_hypothesis as _run
from .config import get_hypothesis

CONFIG = get_hypothesis("H2")


def run(profiles: Dict[str, dict], verbose: bool = True) -> Any:
    """Run H2 hypothesis test."""
    return _run("H2", profiles, verbose=verbose)


def describe() -> str:
    """Return the full hypothesis description."""
    return __doc__
