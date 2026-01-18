"""
OH Parser Testing Script
========================
Testing various extraction methods with real OH profile data.
"""

from oh_parser import (
    load_profiles, 
    list_subjects, 
    get_profile, 
    inspect_profile, 
    get_available_paths,
    extract,
    extract_nested,
    extract_flat,
    create_filters,
    resolve_path,
    list_keys_at_path,
    path_exists,
    summarize_profiles,
)

# =============================================================================
# SETUP: Load all profiles
# =============================================================================
print("=" * 60)
print("LOADING PROFILES")
print("=" * 60)

OH_PROFILES_PATH = r"/Volumes/NO NAME/Backup PrevOccupAI_PLUS Data/OH_profiles"
profiles = load_profiles(OH_PROFILES_PATH)

subjects = list_subjects(profiles)
print(f"Subjects: {subjects}")
print(f"Total: {len(subjects)} subjects\n")


# =============================================================================
# EXAMPLE 1: Inspect profile structure
# =============================================================================
print("=" * 60)
print("EXAMPLE 1: Profile Structure Inspection")
print("=" * 60)

profile = get_profile(profiles, subjects[0])
print(f"\nStructure of subject {subjects[0]}:")

if profile is not None:
    inspect_profile(profile, max_depth=3)
    
    # Show first 10 available paths
    paths = get_available_paths(profile, max_depth=6)
    print(f"\nFirst 10 extractable paths:")
    for p in paths[:10]:
        print(f"  {p}")
else:
    print("Profile not found!")


# =============================================================================
# EXAMPLE 2: Extract specific paths (wide format)
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 2: Extract Weekly EMG Summary (Wide Format)")
print("=" * 60)

df_weekly = extract(profiles, paths={
    # Left side EMG
    "p10_L": "sensor_metrics.emg.EMG_weekly_metrics.left.EMG_apdf.active.p10",
    "p50_L": "sensor_metrics.emg.EMG_weekly_metrics.left.EMG_apdf.active.p50",
    "p90_L": "sensor_metrics.emg.EMG_weekly_metrics.left.EMG_apdf.active.p90",
    "rest_pct_L": "sensor_metrics.emg.EMG_weekly_metrics.left.EMG_rest_recovery.rest_percent",
    
    # Right side EMG
    "p10_R": "sensor_metrics.emg.EMG_weekly_metrics.right.EMG_apdf.active.p10",
    "p50_R": "sensor_metrics.emg.EMG_weekly_metrics.right.EMG_apdf.active.p50",
    "p90_R": "sensor_metrics.emg.EMG_weekly_metrics.right.EMG_apdf.active.p90",
    "rest_pct_R": "sensor_metrics.emg.EMG_weekly_metrics.right.EMG_rest_recovery.rest_percent",
})

print(f"\nWeekly EMG DataFrame shape: {df_weekly.shape}")
print(f"Columns: {df_weekly.columns.tolist()}")
print("\nFirst 5 rows:")
print(df_weekly.head())


# =============================================================================
# EXAMPLE 3: Extract all EMG sessions (long format)
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 3: Extract All EMG Sessions (Long Format)")
print("=" * 60)

df_sessions = extract_nested(
    profiles,
    base_path="sensor_metrics.emg",
    level_names=["date", "session", "side"],
    value_paths=[
        "EMG_intensity.mean_percent_mvc",
        "EMG_intensity.max_percent_mvc",
        "EMG_apdf.active.p50",
        "EMG_rest_recovery.rest_percent",
    ],
    exclude_patterns=["EMG_daily_metrics", "EMG_weekly_metrics"],
)

print(f"\nSession-level DataFrame shape: {df_sessions.shape}")
print(f"Columns: {df_sessions.columns.tolist()}")
print("\nFirst 10 rows:")
print(df_sessions.head(10))


# =============================================================================
# EXAMPLE 4: Extract with wildcards (all keys under a path)
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 4: Extract with Wildcards")
print("=" * 60)

df_wildcard = extract_nested(
    profiles,
    base_path="sensor_metrics.emg",
    level_names=["date", "session", "side"],
    value_paths=[
        "EMG_intensity.*",        # All intensity metrics
        "EMG_rest_recovery.*",    # All rest/recovery metrics
    ],
    exclude_patterns=["EMG_daily_metrics", "EMG_weekly_metrics"],
)

print(f"\nWildcard extraction shape: {df_wildcard.shape}")
print(f"Columns extracted: {df_wildcard.columns.tolist()}")
print("\nFirst 5 rows:")
print(df_wildcard.head())


# =============================================================================
# EXAMPLE 5: Extract with filters
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 5: Extract with Filters (First 3 Subjects Only)")
print("=" * 60)

filters = create_filters(
    subject_ids=subjects[:3],  # Only first 3 subjects
)

df_filtered = extract_nested(
    profiles,
    base_path="sensor_metrics.emg",
    level_names=["date", "session", "side"],
    value_paths=["EMG_intensity.mean_percent_mvc"],
    exclude_patterns=["EMG_daily_metrics", "EMG_weekly_metrics"],
    filters=filters,
)

print(f"\nFiltered DataFrame shape: {df_filtered.shape}")
print(f"Subjects included: {df_filtered['subject_id'].unique().tolist()}")
print("\nFirst 10 rows:")
print(df_filtered.head(10))


# =============================================================================
# EXAMPLE 6: Flatten everything under a path
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 6: Flatten Weekly Metrics")
print("=" * 60)

df_flat = extract_flat(profiles, base_path="sensor_metrics.emg.EMG_weekly_metrics")

print(f"\nFlattened DataFrame shape: {df_flat.shape}")
print(f"Number of columns: {len(df_flat.columns)}")
print(f"\nFirst 10 columns:")
for col in df_flat.columns[:10]:
    print(f"  {col}")


# =============================================================================
# EXAMPLE 7: Manual path navigation
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 7: Manual Path Navigation")
print("=" * 60)

profile = get_profile(profiles, subjects[0])

if profile is not None:
    # Check if path exists
    has_weekly = path_exists(profile, "sensor_metrics.emg.EMG_weekly_metrics")
    print(f"\nSubject {subjects[0]} has weekly EMG metrics: {has_weekly}")

    # Get a specific value
    p50_left = resolve_path(profile, "sensor_metrics.emg.EMG_weekly_metrics.left.EMG_apdf.active.p50")
    print(f"Left p50 value: {p50_left}")

    # List keys at a path
    emg_keys = list_keys_at_path(profile, "sensor_metrics.emg")
    print(f"\nKeys under sensor_metrics.emg:")
    for key in emg_keys[:10]:
        print(f"  {key}")
else:
    print(f"\nProfile for subject {subjects[0]} not found!")


# =============================================================================
# EXAMPLE 8: Data availability summary
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 8: Data Availability Summary")
print("=" * 60)

summary = summarize_profiles(profiles)
print(f"\nSummary DataFrame shape: {summary.shape}")
print(f"Columns: {summary.columns.tolist()}")
print("\nSummary:")
print(summary.head(10))


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 60)