"""
Data Preparation Module
=======================

Transforms OH Parser output into analysis-ready datasets with:
- Date harmonization (DD-MM-YYYY to datetime)
- Day index computation (ordinal within-subject)
- Side handling (separate, average, or both)
- Weekday extraction (for exploratory use)

Architecture Note:
    This module uses dictionaries instead of classes for data structures
    to maintain consistency with the oh_parser project style.
    
    AnalysisDataset is a TypedDict containing:
    - data: pandas DataFrame with tidy long-format data
    - outcome_vars: list of outcome column names
    - id_var, time_var: identifier columns
    - grouping_vars: additional grouping columns
    - sensor, level: metadata
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
import warnings

import pandas as pd
import numpy as np


# =============================================================================
# Date Parsing
# =============================================================================

def parse_date(date_str: str) -> Optional[pd.Timestamp]:
    """
    Parse date strings in multiple formats.
    
    Supports:
    - DD-MM-YYYY (EMG dates)
    - YYYY-MM-DD (questionnaire dates, ISO format)
    
    :param date_str: Date string to parse
    :returns: pandas Timestamp or None if parsing fails
    """
    formats = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    
    # Last resort: let pandas try to infer
    try:
        return pd.to_datetime(date_str, errors="coerce")
    except Exception:
        return None


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Parse a series of date strings to datetime."""
    return series.apply(parse_date)


# =============================================================================
# Analysis Dataset TypedDict
# =============================================================================

class AnalysisDataset(TypedDict):
    """
    Container for analysis-ready data with metadata.
    
    Keys:
        data: Tidy long-format DataFrame
        outcome_vars: List of outcome column names
        id_var: Subject identifier column (default: "subject_id")
        time_var: Time/date column (default: "date")
        grouping_vars: Additional grouping columns (e.g., ["side"])
        sensor: Source sensor (e.g., "emg", "heart_rate")
        level: Analysis level (e.g., "daily", "session")
    """
    data: pd.DataFrame
    outcome_vars: List[str]
    id_var: str
    time_var: str
    grouping_vars: List[str]
    sensor: str
    level: str


def create_analysis_dataset(
    data: pd.DataFrame,
    outcome_vars: List[str],
    id_var: str = "subject_id",
    time_var: str = "date",
    grouping_vars: Optional[List[str]] = None,
    sensor: str = "emg",
    level: str = "daily",
) -> AnalysisDataset:
    """
    Create an AnalysisDataset dictionary with validation.
    
    :param data: Tidy long-format DataFrame
    :param outcome_vars: List of outcome column names
    :param id_var: Subject identifier column (default: "subject_id")
    :param time_var: Time/date column (default: "date")
    :param grouping_vars: Additional grouping columns (e.g., ["side"])
    :param sensor: Source sensor (e.g., "emg", "heart_rate")
    :param level: Analysis level (e.g., "daily", "session")
    :returns: Validated AnalysisDataset dictionary
    """
    grouping_vars = grouping_vars or []
    
    # Validate
    ds: AnalysisDataset = {
        "data": data,
        "outcome_vars": outcome_vars,
        "id_var": id_var,
        "time_var": time_var,
        "grouping_vars": grouping_vars,
        "sensor": sensor,
        "level": level,
    }
    
    ds = validate_dataset(ds)
    return ds


def validate_dataset(ds: AnalysisDataset) -> AnalysisDataset:
    """
    Validate the dataset structure.
    
    :param ds: AnalysisDataset dictionary
    :returns: Validated AnalysisDataset (outcome_vars may be filtered)
    :raises ValueError: If required columns are missing
    """
    data = ds["data"]
    id_var = ds["id_var"]
    time_var = ds["time_var"]
    outcome_vars = ds["outcome_vars"]
    
    if id_var not in data.columns:
        raise ValueError(f"ID variable '{id_var}' not found in data")
    if time_var not in data.columns:
        raise ValueError(f"Time variable '{time_var}' not found in data")
    
    missing_outcomes = [v for v in outcome_vars if v not in data.columns]
    if missing_outcomes:
        warnings.warn(f"Outcome variables not found in data: {missing_outcomes}")
        ds["outcome_vars"] = [v for v in outcome_vars if v in data.columns]
    
    return ds


def get_n_subjects(ds: AnalysisDataset) -> int:
    """Get number of unique subjects in dataset."""
    return ds["data"][ds["id_var"]].nunique()


def get_n_observations(ds: AnalysisDataset) -> int:
    """Get total number of observations in dataset."""
    return len(ds["data"])


def get_date_range(ds: AnalysisDataset) -> Tuple[Any, Any]:
    """Get date range (min, max) from dataset."""
    dates = ds["data"][ds["time_var"]]
    return (dates.min(), dates.max())


def get_obs_per_subject(ds: AnalysisDataset) -> pd.Series:
    """Get number of observations per subject."""
    return ds["data"].groupby(ds["id_var"]).size()


def subset_dataset(
    ds: AnalysisDataset,
    outcomes: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None,
    sides: Optional[List[str]] = None,
) -> AnalysisDataset:
    """
    Create a subset of the dataset.
    
    :param ds: AnalysisDataset dictionary
    :param outcomes: Subset of outcome variables
    :param subjects: Subset of subject IDs
    :param sides: Subset of sides (if "side" in grouping_vars)
    :returns: New AnalysisDataset with filtered data
    """
    df = ds["data"].copy()
    
    if subjects is not None:
        df = df[df[ds["id_var"]].isin(subjects)]
    
    if sides is not None and "side" in ds["grouping_vars"]:
        df = df[df["side"].isin(sides)]
    
    new_outcomes = outcomes if outcomes is not None else ds["outcome_vars"]
    
    return create_analysis_dataset(
        data=df,
        outcome_vars=new_outcomes,
        id_var=ds["id_var"],
        time_var=ds["time_var"],
        grouping_vars=ds["grouping_vars"],
        sensor=ds["sensor"],
        level=ds["level"],
    )


def describe_dataset(ds: AnalysisDataset) -> str:
    """
    Return a summary description of the dataset.
    
    :param ds: AnalysisDataset dictionary
    :returns: Human-readable summary string
    """
    date_range = get_date_range(ds)
    lines = [
        f"AnalysisDataset: {ds['sensor']} ({ds['level']} level)",
        f"  Subjects: {get_n_subjects(ds)}",
        f"  Observations: {get_n_observations(ds)}",
        f"  Date range: {date_range[0]} to {date_range[1]}",
        f"  Outcomes: {len(ds['outcome_vars'])} variables",
        f"  Grouping: {ds['grouping_vars']}",
    ]
    return "\n".join(lines)


# =============================================================================
# Core DataFrame Preparation (No oh_parser dependency)
# =============================================================================

SideOption = Literal["left", "right", "both", "average"]


def prepare_from_dataframe(
    df: pd.DataFrame,
    sensor: str = "unknown",
    level: str = "daily",
    id_col: str = "subject_id",
    date_col: Optional[str] = "date",
    side: SideOption = "both",
    outcome_cols: Optional[List[str]] = None,
    add_day_index: bool = True,
    add_weekday: bool = True,
) -> AnalysisDataset:
    """
    Prepare an already-extracted DataFrame for statistical analysis.
    
    Use this function when you have already extracted data using oh_parser
    (e.g., via extract_nested() or extract_flat()) and want to convert it
    to an AnalysisDataset for use with oh_stats modeling functions.
    
    This is the recommended approach for maximum flexibility, as it avoids
    redundant data extraction and allows you to pre-process the DataFrame
    as needed before analysis.
    
    :param df: DataFrame with subject data (from oh_parser or any source)
    :param sensor: Sensor/data type name for metadata (e.g., "emg", "heart_rate")
    :param level: Analysis level for metadata (e.g., "daily", "session", "weekly")
    :param id_col: Column name for subject identifier (default: "subject_id")
    :param date_col: Column name for date/time (None if no date column)
    :param side: How to handle sides if "side" column exists:
        - "left": Keep only left side
        - "right": Keep only right side
        - "both": Keep both sides as separate rows (default)
        - "average": Average across sides (only where both exist)
    :param outcome_cols: Specific columns to treat as outcomes (None = auto-detect)
    :param add_day_index: Add within-subject day index (1, 2, 3, ...)
    :param add_weekday: Add weekday name column
    :returns: AnalysisDataset dictionary ready for oh_stats functions
    
    Example:
        >>> from oh_parser import extract_nested
        >>> from oh_stats import prepare_from_dataframe, fit_lmm
        >>> 
        >>> # Step 1: Extract data with oh_parser (you control this)
        >>> df = extract_nested(
        ...     profiles,
        ...     base_path="sensor_metrics.emg",
        ...     level_names=["date", "level", "side"],
        ...     value_paths=["EMG_intensity.*"],
        ...     flatten_values=True,
        ... )
        >>> 
        >>> # Step 2: Filter/transform as needed
        >>> df = df[df["level"] == "EMG_daily_metrics"]
        >>> 
        >>> # Step 3: Convert to AnalysisDataset (no redundant extraction!)
        >>> ds = prepare_from_dataframe(df, sensor="emg", side="average")
        >>> 
        >>> # Step 4: Use oh_stats as normal
        >>> result = fit_lmm(ds, "EMG_intensity.mean_percent_mvc")
    """
    if df.empty:
        warnings.warn(f"Empty DataFrame provided")
        return create_analysis_dataset(
            data=pd.DataFrame(),
            outcome_vars=[],
            sensor=sensor,
            level=level,
        )
    
    df = df.copy()
    
    # Validate id column exists
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame. "
                        f"Available columns: {list(df.columns)}")
    
    # Parse dates if date column provided
    if date_col and date_col in df.columns:
        df["date"] = _parse_date_column(df[date_col])
        if date_col != "date":
            df = df.drop(columns=[date_col])
        
        # Remove rows with unparseable dates
        n_before = len(df)
        df = df.dropna(subset=["date"])
        if len(df) < n_before:
            warnings.warn(f"Dropped {n_before - len(df)} rows with unparseable dates")
    
    # Handle sides if present
    grouping_vars: List[str] = []
    if "side" in df.columns:
        df, grouping_vars = _handle_sides(df, side)
    
    # Add day index (within-subject ordinal)
    if add_day_index and "date" in df.columns:
        df = _add_day_index(df)
    
    # Add weekday
    if add_weekday and "date" in df.columns:
        df["weekday"] = df["date"].dt.day_name()
    
    # Sort for reproducibility
    sort_cols = [id_col]
    if "date" in df.columns:
        sort_cols.append("date")
    if "side" in df.columns:
        sort_cols.append("side")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Identify outcome columns
    meta_cols = {id_col, "date", "side", "day_index", "weekday"}
    if outcome_cols is None:
        outcome_vars = [c for c in df.columns if c not in meta_cols]
    else:
        # Validate provided outcome columns exist
        missing = [c for c in outcome_cols if c not in df.columns]
        if missing:
            warnings.warn(f"Requested outcome columns not found: {missing}")
        outcome_vars = [c for c in outcome_cols if c in df.columns]
    
    # Rename id column to standard name if needed
    if id_col != "subject_id":
        df = df.rename(columns={id_col: "subject_id"})
    
    return create_analysis_dataset(
        data=df,
        outcome_vars=outcome_vars,
        id_var="subject_id",
        time_var="date" if "date" in df.columns else "day_index",
        grouping_vars=grouping_vars,
        sensor=sensor,
        level=level,
    )


# =============================================================================
# Generic Sensor Data Preparation (from profiles - uses prepare_from_dataframe)
# =============================================================================

def prepare_sensor_data(
    profiles: Dict[str, dict],
    sensor: str,
    base_path: str,
    level_names: List[str],
    value_paths: List[str],
    level_filter: Optional[Dict[str, str]] = None,
    side: SideOption = "both",
    add_day_index: bool = True,
    add_weekday: bool = True,
) -> AnalysisDataset:
    """
    Generic preparation function for any sensor data from OH profiles.
    
    This function extracts data from OH profiles using oh_parser and then
    converts it to an AnalysisDataset. For more control, consider using
    oh_parser directly followed by prepare_from_dataframe().
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param sensor: Sensor name (e.g., "emg", "accelerometer", "heart_rate")
    :param base_path: Base JSON path to sensor data (e.g., "sensor_metrics.emg")
    :param level_names: Names for nested levels (e.g., ["date", "level", "side"])
    :param value_paths: Glob patterns for values to extract (e.g., ["EMG_intensity.*"])
    :param level_filter: Filter dict to apply (e.g., {"level": "EMG_daily_metrics"})
    :param side: How to handle sides ("left", "right", "both", "average")
    :param add_day_index: Add within-subject day index (1, 2, 3, ...)
    :param add_weekday: Add weekday name column
    :returns: AnalysisDataset dictionary
    
    Example:
        >>> # Prepare accelerometer data
        >>> ds = prepare_sensor_data(
        ...     profiles,
        ...     sensor="accelerometer",
        ...     base_path="sensor_metrics.accelerometer",
        ...     level_names=["date", "placement"],
        ...     value_paths=["activity.*", "posture.*"],
        ...     add_day_index=True
        ... )
    
    Note:
        For more control over data extraction and transformation, use:
        >>> df = extract_nested(profiles, ...)  # oh_parser
        >>> ds = prepare_from_dataframe(df, ...)  # oh_stats
    """
    from oh_parser import extract_nested
    
    # Extract data using oh_parser
    df = extract_nested(
        profiles,
        base_path=base_path,
        level_names=level_names,
        value_paths=value_paths,
        flatten_values=True,
    )
    
    if df.empty:
        warnings.warn(f"No {sensor} data found in profiles")
        return create_analysis_dataset(
            data=pd.DataFrame(),
            outcome_vars=[],
            sensor=sensor,
            level="daily",
        )
    
    # Apply level filter if provided (before passing to prepare_from_dataframe)
    if level_filter:
        for col, val in level_filter.items():
            if col in df.columns:
                df = df[df[col] == val].copy()
                df = df.drop(columns=[col])
    
    # Detect date column
    date_col = None
    for col in ["date", "Date", "DATE", "timestamp"]:
        if col in df.columns:
            date_col = col
            break
    
    # Use prepare_from_dataframe for the actual preparation
    return prepare_from_dataframe(
        df=df,
        sensor=sensor,
        level="daily",
        id_col="subject_id",
        date_col=date_col,
        side=side,
        add_day_index=add_day_index,
        add_weekday=add_weekday,
    )


# =============================================================================
# EMG Data Preparation (Convenience Wrapper)
# =============================================================================

def prepare_daily_emg(
    profiles: Dict[str, dict],
    side: SideOption = "both",
    add_day_index: bool = True,
    add_weekday: bool = True,
) -> AnalysisDataset:
    """
    Prepare daily EMG metrics for analysis.
    
    Extracts EMG_daily_metrics from OH profiles and returns an analysis-ready
    dataset with parsed dates, day indices, and optional side handling.
    
    This is a convenience wrapper. For more control, use oh_parser directly
    followed by prepare_from_dataframe().
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param side: How to handle sides:
        - "left": Only left side data
        - "right": Only right side data
        - "both": Keep both sides as separate rows (default)
        - "average": Average across sides (only when both exist)
    :param add_day_index: Add within-subject day index (1, 2, 3, ...)
    :param add_weekday: Add weekday name column
    :returns: AnalysisDataset dictionary with daily EMG metrics
    
    Example:
        >>> from oh_parser import load_profiles
        >>> profiles = load_profiles("/path/to/OH_profiles")
        >>> ds = prepare_daily_emg(profiles, side="both")
        >>> print(describe_dataset(ds))
    
    Note:
        For more control, use oh_parser directly:
        >>> df = extract_nested(profiles, base_path="sensor_metrics.emg", ...)
        >>> df = df[df["level"] == "EMG_daily_metrics"]  # Your custom filtering
        >>> ds = prepare_from_dataframe(df, sensor="emg", side="average")
    """
    from oh_parser import extract_nested
    
    # Extract daily EMG metrics
    df = extract_nested(
        profiles,
        base_path="sensor_metrics.emg",
        level_names=["date", "level", "side"],
        value_paths=[
            "EMG_session.*",
            "EMG_intensity.*",
            "EMG_apdf.full.*",
            "EMG_apdf.active.*",
            "EMG_rest_recovery.*",
            "EMG_relative_bins.*",
        ],
        flatten_values=True,
    )
    
    if df.empty:
        warnings.warn("No EMG data found in profiles")
        return create_analysis_dataset(
            data=pd.DataFrame(),
            outcome_vars=[],
            sensor="emg",
            level="daily",
        )
    
    # Filter to daily metrics only (pre-processing before prepare_from_dataframe)
    df = df[df["level"] == "EMG_daily_metrics"].copy()
    df = df.drop(columns=["level"])
    
    # Use prepare_from_dataframe for the actual preparation
    return prepare_from_dataframe(
        df=df,
        sensor="emg",
        level="daily",
        id_col="subject_id",
        date_col="date",
        side=side,
        add_day_index=add_day_index,
        add_weekday=add_weekday,
    )


def _handle_sides(
    df: pd.DataFrame,
    side: SideOption,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Handle side filtering/averaging.
    
    :param df: DataFrame with "side" column
    :param side: Side handling option
    :returns: (processed_df, grouping_vars)
    """
    if "side" not in df.columns:
        return df, []
    
    if side == "left":
        df = df[df["side"] == "left"].copy()
        df = df.drop(columns=["side"])
        return df, []
    
    elif side == "right":
        df = df[df["side"] == "right"].copy()
        df = df.drop(columns=["side"])
        return df, []
    
    elif side == "both":
        return df, ["side"]
    
    elif side == "average":
        # Average across sides only when both exist
        meta_cols = ["subject_id", "date"]
        
        # Check which subject×date combinations have both sides
        side_counts = df.groupby(["subject_id", "date"])["side"].nunique()
        has_both = side_counts[side_counts == 2].index
        
        if len(has_both) == 0:
            warnings.warn("No subject×date combinations have both sides. Returning all data.")
            return df, ["side"]
        
        # Filter to only rows with both sides
        df_both = df.set_index(["subject_id", "date"])
        df_both = df_both.loc[df_both.index.isin(has_both)].reset_index()
        
        # Identify numeric columns for averaging
        numeric_cols = df_both.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in meta_cols]
        
        # Group and average
        df_avg = df_both.groupby(meta_cols)[numeric_cols].mean().reset_index()
        
        n_dropped = len(df) - len(df_both)
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} rows where only one side existed. "
                f"Kept {len(df_avg)} averaged observations."
            )
        
        return df_avg, []
    
    else:
        raise ValueError(f"Unknown side option: {side}")


def _add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-subject day index (ordinal: 1, 2, 3, ...).
    
    Days are ordered chronologically within each subject.
    """
    df = df.copy()
    
    # Compute day_index per subject using transform instead of apply
    # This avoids the FutureWarning about grouping columns
    df = df.sort_values(["subject_id", "date"])
    
    # Create a mapping of date to day index within each subject
    day_indices = []
    for _, group in df.groupby("subject_id", sort=False):
        unique_dates = group["date"].unique()
        date_to_idx = {d: i + 1 for i, d in enumerate(sorted(unique_dates))}
        day_indices.extend(group["date"].map(date_to_idx).tolist())
    
    df["day_index"] = day_indices
    df["day_index"] = df["day_index"].astype(int)
    
    return df


# =============================================================================
# Questionnaire Data Preparation (Conditional)
# =============================================================================

def prepare_daily_questionnaires(
    profiles: Dict[str, dict],
    domain: Optional[str] = None,
    add_day_index: bool = True,
    add_weekday: bool = True,
) -> Optional[AnalysisDataset]:
    """
    Prepare daily questionnaire data for analysis.
    
    Returns None if no questionnaire data is available (conditionally activated).
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param domain: Questionnaire domain ("workload", "pain", or None for all)
    :param add_day_index: Add within-subject day index
    :param add_weekday: Add weekday name column
    :returns: AnalysisDataset dict or None if no data
    """
    # Import here to avoid circular dependency
    from oh_parser import extract_nested
    
    # Check if any profile has questionnaire data
    has_data = False
    for profile in profiles.values():
        dq = profile.get("daily_questionnaires", {})
        if domain:
            if dq.get(domain):
                has_data = True
                break
        else:
            if any(bool(v) for v in dq.values() if isinstance(v, dict)):
                has_data = True
                break
    
    if not has_data:
        # Conditionally deactivated - no data available
        return None
    
    # Build base path
    base_path = f"daily_questionnaires.{domain}" if domain else "daily_questionnaires"
    
    # Extract
    level_names = ["date"] if domain else ["domain", "date"]
    
    df = extract_nested(
        profiles,
        base_path=base_path,
        level_names=level_names,
        value_paths=["*"],
        flatten_values=True,
    )
    
    if df.empty:
        return None
    
    # Parse dates
    df["date"] = _parse_date_column(df["date"])
    df = df.dropna(subset=["date"])
    
    if df.empty:
        return None
    
    # Add day index
    if add_day_index:
        df = _add_day_index(df)
    
    # Add weekday
    if add_weekday:
        df["weekday"] = df["date"].dt.day_name()
    
    # Identify outcome columns
    meta_cols = {"subject_id", "date", "domain", "day_index", "weekday"}
    outcome_vars = [c for c in df.columns if c not in meta_cols]
    
    grouping_vars = ["domain"] if "domain" in df.columns else []
    
    return create_analysis_dataset(
        data=df,
        outcome_vars=outcome_vars,
        id_var="subject_id",
        time_var="date",
        grouping_vars=grouping_vars,
        sensor="questionnaire",
        level="daily",
    )


# =============================================================================
# Weekly Data Preparation
# =============================================================================

def prepare_weekly_emg(
    profiles: Dict[str, dict],
    side: SideOption = "both",
) -> AnalysisDataset:
    """
    Prepare weekly EMG aggregates for analysis.
    
    Note: Weekly data has only one observation per subject×side,
    so it's suitable for between-subject comparisons only.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param side: How to handle sides
    :returns: AnalysisDataset dict with weekly EMG metrics
    """
    from oh_parser import extract_flat
    
    df = extract_flat(profiles, base_path="sensor_metrics.emg.EMG_weekly_metrics")
    
    if df.empty:
        warnings.warn("No weekly EMG data found in profiles")
        return create_analysis_dataset(
            data=pd.DataFrame(),
            outcome_vars=[],
            sensor="emg",
            level="weekly",
        )
    
    # Reshape from wide to long (one row per side)
    # Current: columns like "left.EMG_apdf.active.p50", "right.EMG_apdf.active.p50"
    
    left_cols = [c for c in df.columns if c.startswith("left.")]
    right_cols = [c for c in df.columns if c.startswith("right.")]
    
    rows = []
    for _, row in df.iterrows():
        subject_id = row["subject_id"]
        
        # Left side
        if left_cols:
            left_row = {"subject_id": subject_id, "side": "left"}
            for c in left_cols:
                new_name = c.replace("left.", "")
                left_row[new_name] = row[c]
            rows.append(left_row)
        
        # Right side
        if right_cols:
            right_row = {"subject_id": subject_id, "side": "right"}
            for c in right_cols:
                new_name = c.replace("right.", "")
                right_row[new_name] = row[c]
            rows.append(right_row)
    
    df_long = pd.DataFrame(rows)
    
    # Handle sides
    df_long, grouping_vars = _handle_sides(df_long, side)
    
    # Identify outcome columns
    meta_cols = {"subject_id", "side"}
    outcome_vars = [c for c in df_long.columns if c not in meta_cols]
    
    return create_analysis_dataset(
        data=df_long,
        outcome_vars=outcome_vars,
        id_var="subject_id",
        time_var="subject_id",  # No time dimension for weekly
        grouping_vars=grouping_vars,
        sensor="emg",
        level="weekly",
    )


# =============================================================================
# Profile Discovery Functions
# =============================================================================

def discover_sensors(profiles: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Discover available sensors and their data keys from OH profiles.
    
    Inspects the sensor_metrics section of profiles to find what sensors
    have data available. This helps users know what data can be extracted.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :returns: Dict mapping sensor names to list of available metric keys
    
    Example:
        >>> profiles = load_profiles("/path/to/OH_profiles")
        >>> sensors = discover_sensors(profiles)
        >>> print(sensors)
        {'heart_rate': ['HR_BPM_stats', 'HR_ratio_stats', ...],
         'noise': ['Noise_statistics', 'Noise_distributions', ...],
         'emg': ['EMG_intensity', 'EMG_apdf', ...]}
    """
    all_sensors: Dict[str, set] = {}
    
    for subject_id, profile in profiles.items():
        sensor_metrics = profile.get("sensor_metrics", {})
        
        for sensor_name, sensor_data in sensor_metrics.items():
            if sensor_name == "sensor_timeline":
                continue  # Skip metadata
                
            if not isinstance(sensor_data, dict):
                continue
            
            if sensor_name not in all_sensors:
                all_sensors[sensor_name] = set()
            
            # Collect keys from the sensor data
            _collect_keys(sensor_data, all_sensors[sensor_name])
    
    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in all_sensors.items()}


def _collect_keys(data: dict, keys: set, depth: int = 0, max_depth: int = 3) -> None:
    """Recursively collect keys from nested dict structure."""
    if depth >= max_depth:
        return
    
    for key, value in data.items():
        # Skip date-like keys (they're the time dimension, not metrics)
        if _looks_like_date(key):
            if isinstance(value, dict):
                _collect_keys(value, keys, depth + 1, max_depth)
            continue
        
        keys.add(key)
        
        if isinstance(value, dict) and depth < max_depth - 1:
            # For nested dicts like HR_BPM_stats, add the nested keys too
            for nested_key in value.keys():
                if not _looks_like_date(nested_key):
                    keys.add(f"{key}.{nested_key}")


def _looks_like_date(s: str) -> bool:
    """Check if a string looks like a date or time key."""
    import re
    # Patterns: DD-MM-YYYY, YYYY-MM-DD, HH-MM-SS, etc.
    date_patterns = [
        r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d{2}-\d{2}-\d{2}$',  # HH-MM-SS
    ]
    return any(re.match(p, s) for p in date_patterns)


def discover_questionnaires(profiles: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Discover available questionnaire domains and their fields from OH profiles.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :returns: Dict with 'single_instance' and 'daily' keys, each mapping
              domain names to lists of field names
    
    Example:
        >>> profiles = load_profiles("/path/to/OH_profiles")
        >>> quests = discover_questionnaires(profiles)
        >>> print(quests['single_instance'].keys())
        dict_keys(['personal', 'biomechanical', 'psychosocial', 'environmental'])
    """
    result = {
        "single_instance": {},
        "daily": {},
    }
    
    for subject_id, profile in profiles.items():
        # Single-instance questionnaires
        siq = profile.get("single_instance_questionnaires", {})
        for domain, domain_data in siq.items():
            if not isinstance(domain_data, dict):
                continue
            if domain not in result["single_instance"]:
                result["single_instance"][domain] = set()
            result["single_instance"][domain].update(domain_data.keys())
        
        # Daily questionnaires
        daily_q = profile.get("daily_questionnaires", {})
        for domain, domain_data in daily_q.items():
            if not isinstance(domain_data, dict):
                continue
            if domain not in result["daily"]:
                result["daily"][domain] = set()
            
            for date_key, day_data in domain_data.items():
                if isinstance(day_data, dict):
                    result["daily"][domain].update(day_data.keys())
    
    # Convert sets to sorted lists
    return {
        "single_instance": {k: sorted(v) for k, v in result["single_instance"].items()},
        "daily": {k: sorted(v) for k, v in result["daily"].items()},
    }


def get_profile_summary(profiles: Dict[str, dict]) -> str:
    """
    Generate a summary of available data in OH profiles.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :returns: Human-readable summary string
    
    Example:
        >>> profiles = load_profiles("/path/to/OH_profiles")
        >>> print(get_profile_summary(profiles))
    """
    sensors = discover_sensors(profiles)
    quests = discover_questionnaires(profiles)
    
    lines = [
        f"OH Profile Summary ({len(profiles)} subjects)",
        "=" * 50,
        "",
        "SENSOR DATA:",
    ]
    
    if sensors:
        for sensor, keys in sorted(sensors.items()):
            lines.append(f"  {sensor}: {len(keys)} metrics")
    else:
        lines.append("  No sensor data found")
    
    lines.extend(["", "SINGLE-INSTANCE QUESTIONNAIRES:"])
    if quests["single_instance"]:
        for domain, fields in sorted(quests["single_instance"].items()):
            lines.append(f"  {domain}: {len(fields)} fields")
    else:
        lines.append("  No single-instance questionnaires found")
    
    lines.extend(["", "DAILY QUESTIONNAIRES:"])
    if quests["daily"]:
        for domain, fields in sorted(quests["daily"].items()):
            lines.append(f"  {domain}: {len(fields)} fields")
    else:
        lines.append("  No daily questionnaires found")
    
    return "\n".join(lines)


# =============================================================================
# Single-Instance Questionnaire Preparation
# =============================================================================

def prepare_baseline_questionnaires(
    profiles: Dict[str, dict],
    domains: Optional[List[str]] = None,
    convert_percentages: bool = True,
) -> AnalysisDataset:
    """
    Prepare single-instance (baseline) questionnaire data for analysis.
    
    Extracts data from single_instance_questionnaires (personal, biomechanical,
    psychosocial, environmental) for cross-sectional analysis.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param domains: Specific domains to include (None = all available)
    :param convert_percentages: Convert percentage values (0-100) to proportions (0-1)
    :returns: AnalysisDataset dict with baseline questionnaire data
    
    Note: This is SINGLE level data - one observation per subject.
    Use for between-subject comparisons only.
    """
    from oh_parser import resolve_path
    
    available_domains = ["personal", "biomechanical", "psychosocial", "environmental"]
    domains = domains or available_domains
    
    rows = []
    
    for subject_id, profile in profiles.items():
        row = {"subject_id": subject_id}
        
        siq = profile.get("single_instance_questionnaires", {})
        
        for domain in domains:
            domain_data = siq.get(domain, {})
            if not domain_data or not isinstance(domain_data, dict):
                continue
            
            # Flatten domain data
            flat = _flatten_questionnaire_domain(domain_data, prefix=domain)
            row.update(flat)
        
        if len(row) > 1:  # Has data beyond subject_id
            rows.append(row)
    
    if not rows:
        warnings.warn("No baseline questionnaire data found")
        return create_analysis_dataset(
            data=pd.DataFrame(),
            outcome_vars=[],
            sensor="questionnaire",
            level="single",
        )
    
    df = pd.DataFrame(rows)
    
    # Convert percentage columns to proportions if requested
    if convert_percentages:
        df = _convert_percentages_to_proportions(df)
    
    # Identify outcome columns
    meta_cols = {"subject_id"}
    outcome_vars = [c for c in df.columns if c not in meta_cols]
    
    return create_analysis_dataset(
        data=df,
        outcome_vars=outcome_vars,
        id_var="subject_id",
        time_var="subject_id",  # No time dimension for single-instance
        grouping_vars=[],
        sensor="questionnaire",
        level="single",
    )


def _flatten_questionnaire_domain(
    data: dict,
    prefix: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten nested questionnaire domain data.
    
    Handles special cases like COPSOQ nested structure.
    """
    flat = {}
    
    for key, value in data.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Check if it's a simple stats dict (like {"mean": 0.4})
            if set(value.keys()) <= {"mean", "mean_FO", "std", "min", "max"}:
                # Extract the mean value
                if "mean" in value:
                    flat[full_key] = value["mean"]
                elif "mean_FO" in value:
                    flat[f"{full_key}_FO"] = value["mean_FO"]
            else:
                # Recurse
                nested = _flatten_questionnaire_domain(value, prefix=full_key, sep=sep)
                flat.update(nested)
        else:
            flat[full_key] = value
    
    return flat


def _convert_percentages_to_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert percentage columns (0-100) to proportions (0-1).
    
    Identifies columns likely to be percentages and converts them.
    """
    df = df.copy()
    
    # Patterns that indicate percentage columns
    pct_patterns = [
        "percentagem", "percent", "_pct", "percentage",
        "ROSA_.*_normalized", "ROSA_final_normalized"
    ]
    
    for col in df.columns:
        # Check if column name suggests percentage
        is_pct_col = any(
            pattern.lower() in col.lower() or 
            col.lower().endswith("_pct")
            for pattern in pct_patterns[:4]
        )
        
        # Also check if values are in 0-100 range (not 0-1)
        if is_pct_col and df[col].dtype in [np.float64, np.int64, float, int]:
            values = df[col].dropna()
            if len(values) > 0 and values.max() > 1:
                df[col] = df[col] / 100.0
    
    return df


def prepare_daily_workload(
    profiles: Dict[str, dict],
    add_day_index: bool = True,
    add_weekday: bool = True,
) -> Optional[AnalysisDataset]:
    """
    Prepare daily workload questionnaire data for analysis.
    
    Extracts workload Likert items from daily_questionnaires.workload.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param add_day_index: Add within-subject day index
    :param add_weekday: Add weekday name column
    :returns: AnalysisDataset dict or None if no data
    """
    rows = []
    
    for subject_id, profile in profiles.items():
        daily_q = profile.get("daily_questionnaires", {})
        workload_data = daily_q.get("workload", {})
        
        if not workload_data:
            continue
        
        for date_str, day_data in workload_data.items():
            # Skip non-data entries
            if date_str == "scoring" or not isinstance(day_data, dict):
                continue
            if day_data == "No data available":
                continue
            
            row = {
                "subject_id": subject_id,
                "date": date_str,
            }
            
            # Extract workload items
            for item, value in day_data.items():
                if item != "open_question" and not pd.isna(value):
                    row[f"workload.{item}"] = value
            
            if len(row) > 2:  # Has data beyond subject_id and date
                rows.append(row)
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    
    # Parse dates
    df["date"] = _parse_date_column(df["date"])
    df = df.dropna(subset=["date"])
    
    if df.empty:
        return None
    
    # Add day index
    if add_day_index:
        df = _add_day_index(df)
    
    # Add weekday
    if add_weekday:
        df["weekday"] = df["date"].dt.day_name()
    
    # Sort
    df = df.sort_values(["subject_id", "date"]).reset_index(drop=True)
    
    # Identify outcome columns
    meta_cols = {"subject_id", "date", "day_index", "weekday"}
    outcome_vars = [c for c in df.columns if c not in meta_cols]
    
    return create_analysis_dataset(
        data=df,
        outcome_vars=outcome_vars,
        id_var="subject_id",
        time_var="date",
        grouping_vars=[],
        sensor="questionnaire",
        level="daily",
    )


def prepare_daily_pain(
    profiles: Dict[str, dict],
    add_day_index: bool = True,
    add_weekday: bool = True,
) -> Optional[AnalysisDataset]:
    """
    Prepare daily pain questionnaire data for analysis.
    
    Extracts pain ratings from daily_questionnaires.pain.
    
    :param profiles: Dictionary mapping subject_id -> OH profile dict
    :param add_day_index: Add within-subject day index
    :param add_weekday: Add weekday name column
    :returns: AnalysisDataset dict or None if no data
    """
    rows = []
    
    for subject_id, profile in profiles.items():
        daily_q = profile.get("daily_questionnaires", {})
        pain_data = daily_q.get("pain", {})
        
        if not pain_data:
            continue
        
        for date_str, day_data in pain_data.items():
            if not isinstance(day_data, dict):
                continue
            
            row = {
                "subject_id": subject_id,
                "date": date_str,
            }
            
            # Extract pain items
            for item, value in day_data.items():
                if not pd.isna(value):
                    row[f"pain.{item}"] = value
            
            if len(row) > 2:
                rows.append(row)
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    
    # Parse dates
    df["date"] = _parse_date_column(df["date"])
    df = df.dropna(subset=["date"])
    
    if df.empty:
        return None
    
    # Add day index
    if add_day_index:
        df = _add_day_index(df)
    
    # Add weekday
    if add_weekday:
        df["weekday"] = df["date"].dt.day_name()
    
    # Sort
    df = df.sort_values(["subject_id", "date"]).reset_index(drop=True)
    
    # Identify outcome columns
    meta_cols = {"subject_id", "date", "day_index", "weekday"}
    outcome_vars = [c for c in df.columns if c not in meta_cols]
    
    return create_analysis_dataset(
        data=df,
        outcome_vars=outcome_vars,
        id_var="subject_id",
        time_var="date",
        grouping_vars=[],
        sensor="questionnaire",
        level="daily",
    )


# =============================================================================
# Composite Score Computation
# =============================================================================

def compute_composite_score(
    df: pd.DataFrame,
    items: List[str],
    score_name: str,
    method: str = "mean",
    min_valid_pct: float = 0.8,
    reverse_items: Optional[List[str]] = None,
    scale_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute composite score from multiple items.
    
    :param df: DataFrame with item columns
    :param items: List of column names to aggregate
    :param score_name: Name for the new composite score column
    :param method: Aggregation method ("mean", "sum")
    :param min_valid_pct: Minimum percentage of non-missing items required
    :param reverse_items: Items to reverse-code before aggregation
    :param scale_max: Maximum scale value for reverse coding (e.g., 5 for Likert 1-5)
    :returns: DataFrame with new composite score column
    
    Example:
        >>> df = compute_composite_score(
        ...     df, 
        ...     items=["item1", "item2", "item3"],
        ...     score_name="domain_score",
        ...     reverse_items=["item2"],
        ...     scale_max=5,
        ... )
    """
    df = df.copy()
    
    # Check which items exist
    existing_items = [i for i in items if i in df.columns]
    if not existing_items:
        warnings.warn(f"None of the items {items} found in DataFrame")
        df[score_name] = np.nan
        return df
    
    # Reverse code items if needed
    if reverse_items and scale_max:
        for item in reverse_items:
            if item in df.columns:
                df[f"{item}_rev"] = scale_max + 1 - df[item]
                existing_items = [
                    f"{i}_rev" if i == item else i 
                    for i in existing_items
                ]
    
    # Count valid items per row
    valid_counts = df[existing_items].notna().sum(axis=1)
    min_valid = int(len(existing_items) * min_valid_pct)
    
    # Compute score
    if method == "mean":
        df[score_name] = df[existing_items].mean(axis=1)
    elif method == "sum":
        df[score_name] = df[existing_items].sum(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Set to NA if insufficient valid items
    df.loc[valid_counts < min_valid, score_name] = np.nan
    
    return df


# =============================================================================
# Data Alignment (Sensor + Questionnaire)
# =============================================================================

def align_sensor_questionnaire(
    sensor_ds: AnalysisDataset,
    questionnaire_ds: AnalysisDataset,
    how: str = "inner",
) -> AnalysisDataset:
    """
    Align sensor data with daily questionnaire data by subject×date.
    
    :param sensor_ds: AnalysisDataset with sensor data (e.g., daily EMG)
    :param questionnaire_ds: AnalysisDataset with daily questionnaire data
    :param how: Join type ("inner", "left", "outer")
    :returns: Combined AnalysisDataset
    
    Note: Reports days with missing sensor or questionnaire data.
    """
    sensor_df = sensor_ds["data"].copy()
    quest_df = questionnaire_ds["data"].copy()
    
    # Ensure date columns are comparable
    if "date" not in sensor_df.columns or "date" not in quest_df.columns:
        raise ValueError("Both datasets must have 'date' column")
    
    # Merge on subject_id and date
    merge_cols = ["subject_id", "date"]
    
    # Add day_index from sensor if present
    if "day_index" in sensor_df.columns and "day_index" not in quest_df.columns:
        merge_cols_quest = ["subject_id", "date"]
    else:
        merge_cols_quest = merge_cols
    
    merged = sensor_df.merge(
        quest_df,
        on=["subject_id", "date"],
        how=how,
        suffixes=("", "_quest"),
    )
    
    # Report alignment
    n_sensor = len(sensor_df)
    n_quest = len(quest_df)
    n_merged = len(merged)
    
    if how == "inner" and n_merged < min(n_sensor, n_quest):
        warnings.warn(
            f"Alignment: {n_sensor} sensor rows, {n_quest} questionnaire rows, "
            f"{n_merged} matched. {n_sensor - n_merged} sensor-only, "
            f"{n_quest - n_merged} questionnaire-only days excluded."
        )
    
    # Combine outcome vars
    combined_outcomes = list(set(sensor_ds["outcome_vars"]) | set(questionnaire_ds["outcome_vars"]))
    combined_outcomes = [c for c in combined_outcomes if c in merged.columns]
    
    # Combine grouping vars
    combined_grouping = list(set(sensor_ds["grouping_vars"]) | set(questionnaire_ds["grouping_vars"]))
    
    return create_analysis_dataset(
        data=merged,
        outcome_vars=combined_outcomes,
        id_var="subject_id",
        time_var="date",
        grouping_vars=combined_grouping,
        sensor="combined",
        level="daily",
    )
