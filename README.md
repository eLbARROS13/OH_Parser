# OH Toolkit

A comprehensive toolkit for Occupational Health (OH) data analysis, consisting of two integrated packages:

| Package | Purpose |
|---------|---------|
| **`oh_parser`** | Extract and structure data from OH profile JSON files |
| **`oh_stats`** | Statistical analysis pipeline with Linear Mixed Models |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Parse OH Profiles

```python
from oh_parser import load_profiles, extract_nested

# Load all profiles
profiles = load_profiles("path/to/OH_profiles/")

# Extract nested EMG data (one row per session)
df = extract_nested(
    profiles,
    base_path="sensor_metrics.emg",
    level_names=["date", "session", "side"],
    value_paths=["EMG_intensity.*", "EMG_rest_recovery.*"],
)
```

### 2. Statistical Analysis

```python
from oh_parser import load_profiles
from oh_stats import (
    prepare_daily_emg,
    fit_all_outcomes,
    apply_fdr,
    results_summary,
)

# Load & prepare
profiles = load_profiles("path/to/OH_profiles/")
ds = prepare_daily_emg(profiles, side="both")

# Fit Linear Mixed Models
results = fit_all_outcomes(ds, skip_degenerate=True)

# Apply FDR correction
fdr = apply_fdr(results)

# Generate summary
report = results_summary(results, fdr)
print(report)
```

## Hypotheses + Reports + Plots

This repository provides two entry points for hypothesis testing:

- **run_hypotheses.py**: production runner for the preregistered H1–H6 suite. Uses the shared hypotheses engine and applies Holm correction to the confirmatory set.
- **testing_stats.py**: step-by-step, verbose tutorial workflow that runs data discovery, QA checks, LMM examples, reporting, and then calls the same H1–H6 runner.

### Run H1–H6 only (production)

```bash
python run_hypotheses.py
```

### Run full step-by-step workflow

```bash
python testing_stats.py
```

### Generate plots for the supervisor report

```bash
python generate_plots.py
```

Outputs:
- Plots: `plots/hypotheses/`
- Supervisor report (Markdown): `docs/OH_SUPERVISOR_REPORT.md`
- Supervisor report (LaTeX/PDF): `docs/OH_SUPERVISOR_REPORT.tex` / `docs/OH_SUPERVISOR_REPORT.pdf`

## Packages

### oh_parser

A generic parser for extracting data from OH profile JSON files into pandas DataFrames.

**Features:**
- Load OH profile JSON files from a directory
- Inspect profile structure without knowing the schema
- Extract data using flexible dot-notation paths
- Filter by subjects, date ranges, or data availability
- Wildcard support for iterating over dynamic keys

See [Full Documentation](docs/OH_PARSER_CONTEXT.md) | [PDF](docs/OH_PARSER_DOCUMENTATION.pdf)

### oh_stats

Statistical analysis pipeline for repeated-measures Occupational Health data using Linear Mixed Models. Supports multiple sensor types and outcome categories.

**Features:**
- **Multi-modal data support**: EMG, questionnaires, and extensible to other sensors
- **Outcome type registry**: Automatic handling of continuous, ordinal, proportion, count, and binary outcomes
- Data preparation with laterality and time-series handling
- Descriptive statistics and assumption checking
- Linear Mixed Models (statsmodels)
- Post-hoc contrasts with effect sizes
- Two-layer multiplicity correction (FDR + Holm)
- Model diagnostics and residual analysis
- Publication-ready report generation

See [Complete Guide](docs/OH_STATS_GUIDE.md) | [PDF](docs/OH_STATS_GUIDE.pdf)

## Project Structure

```
OH_Toolkit/
├── README.md
├── requirements.txt
├── oh_parser/                    # Data extraction package
│   ├── __init__.py
│   ├── loader.py
│   ├── path_resolver.py
│   ├── filters.py
│   ├── extract.py
│   └── utils.py
├── oh_stats/                     # Statistical analysis package
│   ├── __init__.py
│   ├── registry.py              # Outcome type registry
│   ├── prepare.py               # Data preparation
│   ├── descriptive.py           # Summary statistics
│   ├── lmm.py                   # Linear Mixed Models
│   ├── posthoc.py               # Post-hoc contrasts
│   ├── multiplicity.py          # FDR/FWER corrections
│   ├── diagnostics.py           # Model diagnostics
│   └── report.py                # Report generation
├── docs/                         # Documentation
│   ├── OH_PARSER_CONTEXT.md
│   ├── OH_PARSER_DOCUMENTATION.pdf
│   ├── OH_STATS_GUIDE.md
│   └── OH_STATS_GUIDE.pdf
│   ├── OH_SUPERVISOR_REPORT.md
│   ├── OH_SUPERVISOR_REPORT.tex
│   └── OH_SUPERVISOR_REPORT.pdf
├── plots/                        # Generated figures
│   └── hypotheses/
├── run_hypotheses.py             # H1–H6 runner (production)
├── generate_plots.py             # Plot generation for reports
├── testing_parser.py             # Parser tests
└── testing_stats.py              # Stats pipeline demo
```

## Dependencies

- Python 3.9+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0

## License

MIT
