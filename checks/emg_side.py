"""
EMG Side Comparison
===================
Loads EMG active APDF p90 across all subjects and both sides, and
summarizes total session duration per side to compare data availability.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from oh_parser import extract_nested, load_profiles, extract


def extract_subject_metadata(profiles, participants_csv: Path = None) -> pd.DataFrame:
	"""Extract subject_id and work_type (FO/BO) from participants CSV."""
	# Get subject IDs from profiles
	subject_ids = [str(sid) for sid in profiles.keys()]
	
	if participants_csv is not None and participants_csv.exists():
		# Load from CSV (semicolon-separated)
		csv_df = pd.read_csv(participants_csv, sep=";", dtype={"subject_id": str})
		csv_df["subject_id"] = csv_df["subject_id"].astype(str)
		# Keep only subjects in our profiles
		metadata = csv_df[csv_df["subject_id"].isin(subject_ids)][["subject_id", "work_type"]].copy()
		return metadata
	else:
		# No CSV available - return empty work_type
		return pd.DataFrame({"subject_id": subject_ids, "work_type": [None] * len(subject_ids)})


def load_emg_apdf_p90_by_side(profiles):
	return extract_nested(
		profiles,
		base_path="sensor_metrics.emg",
		level_names=["date", "session", "side"],
		value_paths=[
			"EMG_apdf.active.p90",
			"EMG_session.duration_s",
		],
		exclude_patterns=["EMG_daily_metrics", "EMG_weekly_metrics"],
	)


def load_emg_daily_p90_by_side(profiles) -> pd.DataFrame:
	"""Extract daily EMG p90 per subject × date × side from EMG_daily_metrics."""
	rows = []
	for subject_id, profile in profiles.items():
		emg = profile.get("sensor_metrics", {}).get("emg", {})
		if not isinstance(emg, dict):
			continue
		for date, date_data in emg.items():
			if not isinstance(date_data, dict):
				continue
			daily_metrics = date_data.get("EMG_daily_metrics")
			if not isinstance(daily_metrics, dict):
				continue
			for side in ("left", "right"):
				side_metrics = daily_metrics.get(side)
				if not isinstance(side_metrics, dict):
					continue
				apdf = side_metrics.get("EMG_apdf", {})
				p90 = None
				if isinstance(apdf, dict):
					p90 = apdf.get("active", {}).get("p90")
					if p90 is None:
						p90 = apdf.get("full", {}).get("p90")
				if p90 is None:
					continue
				rows.append(
					{
						"subject_id": str(subject_id),
						"date": date,
						"side": side,
						"emg_daily_p90": p90,
					}
				)
	return pd.DataFrame(rows)


def summarize_side_availability(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df

	return (
		df.groupby(["subject_id", "side"], as_index=False)
		.agg(
			total_duration_s=("EMG_session.duration_s", "sum"),
			session_count=("EMG_session.duration_s", "count"),
			mean_apdf_p90=("EMG_apdf.active.p90", "mean"),
			median_apdf_p90=("EMG_apdf.active.p90", "median"),
		)
		.sort_values(["subject_id", "side"])
	)


def build_side_difference(summary: pd.DataFrame) -> pd.DataFrame:
	if summary.empty:
		return summary

	wide = summary.pivot(index="subject_id", columns="side", values="total_duration_s")
	wide = wide.rename(columns={"left": "left_duration_s", "right": "right_duration_s"})
	wide = wide.reset_index()

	wide["duration_diff_s"] = wide["left_duration_s"] - wide["right_duration_s"]
	wide["duration_diff_abs_s"] = wide["duration_diff_s"].abs()
	wide["duration_diff_pct"] = (
		wide["duration_diff_s"] / wide[["left_duration_s", "right_duration_s"]].sum(axis=1)
	)

	return wide


def build_session_coverage(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df

	session = (
		df.pivot_table(
			index=["subject_id", "date", "session"],
			columns="side",
			values="EMG_session.duration_s",
			aggfunc="max",
		)
		.reset_index()
	)

	session["has_left"] = session.get("left").notna()
	session["has_right"] = session.get("right").notna()
	session["duration_union_s"] = session[["left", "right"]].max(axis=1, skipna=True)

	session["left_only_s"] = session["duration_union_s"].where(
		session["has_left"] & ~session["has_right"], 0
	)
	session["right_only_s"] = session["duration_union_s"].where(
		session["has_right"] & ~session["has_left"], 0
	)
	session["both_s"] = session["duration_union_s"].where(
		session["has_left"] & session["has_right"], 0
	)

	per_subject = (
		session.groupby("subject_id", as_index=False)
		.agg(
			union_total_s=("duration_union_s", "sum"),
			left_only_s=("left_only_s", "sum"),
			right_only_s=("right_only_s", "sum"),
			both_s=("both_s", "sum"),
		)
		.sort_values("subject_id")
	)

	per_subject["right_coverage"] = (
		(per_subject["both_s"] + per_subject["right_only_s"])
		/ per_subject["union_total_s"]
	)
	per_subject["right_only_loss_fraction"] = (
		per_subject["left_only_s"] / per_subject["union_total_s"]
	)
	per_subject["bilateral_fraction"] = per_subject["both_s"] / per_subject["union_total_s"]

	return per_subject


def build_session_level_missingness(df: pd.DataFrame, metadata: pd.DataFrame = None) -> pd.DataFrame:
	"""
	Build session-level data with has_right indicator for missingness diagnostics.
	
	Returns a DataFrame with one row per subject × date × session, including:
	- has_right (0/1)
	- day_of_week (0=Mon, 6=Sun)
	- session_hour (extracted from session time)
	- group (if metadata provided)
	"""
	session = (
		df.pivot_table(
			index=["subject_id", "date", "session"],
			columns="side",
			values="EMG_session.duration_s",
			aggfunc="max",
		)
		.reset_index()
	)
	
	session["has_right"] = session.get("right").notna().astype(int)
	session["has_left"] = session.get("left").notna().astype(int)
	
	# Parse date to get day of week
	session["date_parsed"] = pd.to_datetime(session["date"], format="%d-%m-%Y", errors="coerce")
	session["day_of_week"] = session["date_parsed"].dt.dayofweek  # 0=Mon, 6=Sun
	session["day_name"] = session["date_parsed"].dt.day_name()
	
	# Parse session time to get hour
	def parse_session_hour(s):
		try:
			parts = str(s).split("-")
			return int(parts[0]) if parts else np.nan
		except:
			return np.nan
	
	session["session_hour"] = session["session"].apply(parse_session_hour)
	
	# Merge metadata if provided
	if metadata is not None and "work_type" in metadata.columns:
		session = session.merge(
			metadata[["subject_id", "work_type"]],
			on="subject_id",
			how="left"
		)
	
	return session


def run_missingness_diagnostic(session_df: pd.DataFrame) -> dict:
	"""
	Run missingness diagnostic: test if has_right is predicted by day_of_week or session_hour.
	
	Uses a mixed logistic regression: has_right ~ day_of_week + session_hour + (1|subject)
	
	Returns a dict with model results and interpretation.
	"""
	import warnings
	
	results = {}
	
	# Descriptive: missingness rate by day of week
	day_rates = session_df.groupby("day_name")["has_right"].agg(["mean", "count"]).round(3)
	day_rates.columns = ["right_present_rate", "n_sessions"]
	results["by_day"] = day_rates
	
	# Descriptive: missingness rate by session hour
	hour_rates = session_df.groupby("session_hour")["has_right"].agg(["mean", "count"]).round(3)
	hour_rates.columns = ["right_present_rate", "n_sessions"]
	results["by_hour"] = hour_rates
	
	# Descriptive: missingness rate by subject (to see variation)
	subj_rates = session_df.groupby("subject_id")["has_right"].mean()
	results["subject_variation"] = {
		"min": subj_rates.min(),
		"max": subj_rates.max(),
		"mean": subj_rates.mean(),
		"std": subj_rates.std(),
		"subjects_with_100pct": (subj_rates == 1.0).sum(),
		"subjects_with_0pct": (subj_rates == 0.0).sum(),
	}
	
	# Try mixed logistic regression
	try:
		import statsmodels.formula.api as smf
		
		model_df = session_df[["has_right", "day_of_week", "session_hour", "subject_id"]].dropna()
		model_df["subject_id"] = model_df["subject_id"].astype(str)
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			model = smf.mixedlm(
				"has_right ~ C(day_of_week) + session_hour",
				data=model_df,
				groups=model_df["subject_id"],
			).fit(method="powell", maxiter=100)
		
		results["mixed_model"] = {
			"converged": model.converged,
			"summary": str(model.summary()),
			"pvalues": model.pvalues.to_dict(),
			"significant_predictors": [k for k, v in model.pvalues.items() if v < 0.05],
		}
	except Exception as e:
		results["mixed_model"] = {"error": str(e)}
	
	# Simple chi-square test: day_of_week vs has_right
	try:
		from scipy.stats import chi2_contingency
		
		contingency = pd.crosstab(session_df["day_of_week"], session_df["has_right"])
		chi2, p, dof, expected = chi2_contingency(contingency)
		results["chi2_day"] = {
			"chi2": chi2,
			"p_value": p,
			"dof": dof,
			"significant": p < 0.05,
		}
	except Exception as e:
		results["chi2_day"] = {"error": str(e)}
	
	# Work type analysis (Front office vs Back office)
	if "work_type" in session_df.columns and session_df["work_type"].notna().any():
		try:
			group_rates = session_df.groupby("work_type")["has_right"].agg(["mean", "count"]).round(3)
			group_rates.columns = ["right_present_rate", "n_sessions"]
			results["by_work_type"] = group_rates
			
			# Chi-square test: work_type vs has_right
			from scipy.stats import chi2_contingency
			contingency_group = pd.crosstab(session_df["work_type"], session_df["has_right"])
			chi2_g, p_g, dof_g, _ = chi2_contingency(contingency_group)
			results["chi2_work_type"] = {
				"chi2": chi2_g,
				"p_value": p_g,
				"dof": dof_g,
				"significant": p_g < 0.05,
			}
			
			# Also compute at subject level: does work_type predict subject's overall right_coverage?
			subj_by_group = session_df.groupby(["subject_id", "work_type"])["has_right"].mean().reset_index()
			subj_by_group.columns = ["subject_id", "work_type", "right_present_rate"]
			group_summary = subj_by_group.groupby("work_type")["right_present_rate"].agg(["mean", "std", "count"]).round(3)
			group_summary.columns = ["mean_right_rate", "std_right_rate", "n_subjects"]
			results["subject_level_by_work_type"] = group_summary
			
			# T-test for work_type difference in subject-level right_present_rate
			from scipy.stats import ttest_ind
			groups = subj_by_group["work_type"].unique()
			if len(groups) == 2:
				g1 = subj_by_group[subj_by_group["work_type"] == groups[0]]["right_present_rate"]
				g2 = subj_by_group[subj_by_group["work_type"] == groups[1]]["right_present_rate"]
				t_stat, t_pval = ttest_ind(g1, g2, equal_var=False)
				results["ttest_group"] = {
					"group1": groups[0],
					"group2": groups[1],
					"t_statistic": t_stat,
					"p_value": t_pval,
					"significant": t_pval < 0.05,
				}
		except Exception as e:
			results["work_type_analysis_error"] = str(e)
	else:
		results["by_work_type"] = None
	
	return results


def main() -> None:
	right_coverage_threshold = 0.8
	min_right_total_s = 30 * 60

	oh_profiles_path = Path(r"E:\Backup PrevOccupAI_PLUS Data\OH_profiles_2")

	profiles = load_profiles(str(oh_profiles_path))
	df = load_emg_apdf_p90_by_side(profiles)

	print("Loaded EMG APDF p90 rows:", df.shape[0])
	print("Columns:", df.columns.tolist())

	# Per-subject × side summary (2 rows per subject)
	summary = summarize_side_availability(df)
	print("\nSide availability summary:")
	print(summary)

	# Per-subject metrics
	side_diff = build_side_difference(summary)
	coverage = build_session_coverage(df)

	diff_variance = side_diff["duration_diff_s"].var(ddof=1)
	diff_std = side_diff["duration_diff_s"].std(ddof=1)

	print("\nPer-subject side difference (duration):")
	print(side_diff)
	print("\nVariance of duration difference (s^2):", diff_variance)
	print("Standard deviation of duration difference (s):", diff_std)
	print(
		"\nRule-of-thumb for acceptability: if |duration_diff_s| is within 1 SD for most subjects, "
		"side balance is likely acceptable. For stricter balance, target |duration_diff_s| <= 0.1 * total_duration_s."
	)

	# Build per-subject coverage DataFrame (1 row per subject)
	subject_coverage = side_diff.merge(coverage, on="subject_id", how="left")
	subject_coverage["flag_10pct_imbalance"] = subject_coverage["duration_diff_abs_s"] >= (
		0.10 * (subject_coverage["left_duration_s"] + subject_coverage["right_duration_s"])
	)
	subject_coverage["flag_20pct_imbalance"] = subject_coverage["duration_diff_abs_s"] >= (
		0.20 * (subject_coverage["left_duration_s"] + subject_coverage["right_duration_s"])
	)
	subject_coverage["flag_1sd_imbalance"] = subject_coverage["duration_diff_abs_s"] >= diff_std
	subject_coverage["flag_2sd_imbalance"] = subject_coverage["duration_diff_abs_s"] >= (2 * diff_std)
	subject_coverage["flag_low_right_coverage"] = subject_coverage["right_coverage"] < right_coverage_threshold
	subject_coverage["flag_low_right_minutes"] = subject_coverage["right_duration_s"] < min_right_total_s
	subject_coverage["flag_problematic_right_side"] = (
		subject_coverage["flag_low_right_coverage"] | subject_coverage["flag_low_right_minutes"]
	)

	# Export two CSVs
	output_dir = Path(__file__).resolve().parent

	# 1. Per-subject × side summary (for side-specific stats like mean_apdf_p90)
	side_summary_path = output_dir / "emg_side_summary.csv"
	summary.to_csv(side_summary_path, index=False)
	print(f"\nPer-subject×side summary exported to: {side_summary_path}")
	print(f"  Shape: {summary.shape[0]} rows × {summary.shape[1]} cols (2 rows per subject)")

	# 2. Per-subject coverage (1 row per subject, all flags and coverage metrics)
	coverage_path = output_dir / "emg_subject_coverage.csv"
	subject_coverage.to_csv(coverage_path, index=False)
	print(f"\nPer-subject coverage exported to: {coverage_path}")
	print(f"  Shape: {subject_coverage.shape[0]} rows × {subject_coverage.shape[1]} cols (1 row per subject)")

	# Print problematic subjects
	problematic = subject_coverage[subject_coverage["flag_problematic_right_side"]]
	print(f"\nProblematic subjects (right_coverage < {right_coverage_threshold} or right < {min_right_total_s/60:.0f} min):")
	print(f"  Count: {len(problematic)}")
	print(f"  IDs: {problematic['subject_id'].tolist()}")

	# ==========================================================================
	# MISSINGNESS DIAGNOSTIC
	# ==========================================================================
	print("\n" + "=" * 70)
	print("MISSINGNESS DIAGNOSTIC: Is right-side missingness random or systematic?")
	print("=" * 70)

	# Extract subject metadata (work_type) from participants CSV
	participants_csv = PROJECT_ROOT / "participants_info.csv"
	metadata = extract_subject_metadata(profiles, participants_csv)
	print(f"\nWork type distribution: {metadata['work_type'].value_counts().to_dict()}")
	
	# Build session-level missingness data
	session_miss = build_session_level_missingness(df, metadata=metadata)
	
	# Export session-level data for further analysis
	session_miss_path = output_dir / "emg_session_missingness.csv"
	session_miss.to_csv(session_miss_path, index=False)
	print(f"\nSession-level missingness data exported to: {session_miss_path}")
	print(f"  Shape: {session_miss.shape[0]} rows (1 per subject × date × session)")

	# Run diagnostic
	diag = run_missingness_diagnostic(session_miss)

	# Report results
	print("\n--- Right-side presence rate by DAY OF WEEK ---")
	print(diag["by_day"])

	print("\n--- Right-side presence rate by SESSION HOUR ---")
	print(diag["by_hour"])

	print("\n--- Subject-level variation in right-side presence ---")
	for k, v in diag["subject_variation"].items():
		print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

	print("\n--- Chi-square test: day_of_week vs has_right ---")
	if "error" in diag["chi2_day"]:
		print(f"  Error: {diag['chi2_day']['error']}")
	else:
		print(f"  Chi2 = {diag['chi2_day']['chi2']:.2f}, p = {diag['chi2_day']['p_value']:.4f}")
		if diag["chi2_day"]["significant"]:
			print("  WARNING: Right-side missingness is NOT random across days!")
		else:
			print("  OK: No evidence that missingness depends on day of week.")

	print("\n--- Mixed logistic model: has_right ~ day_of_week + session_hour + (1|subject) ---")
	if "error" in diag["mixed_model"]:
		print(f"  Error: {diag['mixed_model']['error']}")
	else:
		if diag["mixed_model"]["converged"]:
			print("  Model converged.")
			if diag["mixed_model"]["significant_predictors"]:
				print(f"  WARNING: Significant predictors (p<0.05): {diag['mixed_model']['significant_predictors']}")
			else:
				print("  OK: No significant predictors of right-side missingness.")
		else:
			print("  WARNING: Model did not converge. Results may be unreliable.")

	# Work type analysis (FO vs BO)
	if diag.get("by_work_type") is not None:
		print("\n--- Right-side presence rate by WORK TYPE (FO=Front office, BO=Back office) ---")
		print(diag["by_work_type"])
		
		if "subject_level_by_work_type" in diag:
			print("\n--- Subject-level right_present_rate by WORK TYPE ---")
			print(diag["subject_level_by_work_type"])
		
		if "chi2_work_type" in diag:
			print("\n--- Chi-square test: work_type vs has_right (session-level) ---")
			print(f"  Chi2 = {diag['chi2_work_type']['chi2']:.2f}, p = {diag['chi2_work_type']['p_value']:.4f}")
			if diag["chi2_work_type"]["significant"]:
				print("  WARNING: Right-side missingness differs between FO and BO!")
			else:
				print("  OK: No evidence that missingness differs between work types.")
		
		if "ttest_group" in diag:
			print("\n--- T-test: work_type difference in subject-level right_present_rate ---")
			tt = diag["ttest_group"]
			print(f"  {tt['group1']} vs {tt['group2']}: t = {tt['t_statistic']:.2f}, p = {tt['p_value']:.4f}")
			if tt["significant"]:
				print("  WARNING: One work type has systematically more right-side missingness!")
				print("      This could bias FO vs BO comparisons if you exclude low-coverage subjects.")
			else:
				print("  OK: FO and BO have similar right-side coverage.")
	else:
		print("\n--- WORK TYPE ANALYSIS ---")
		print("  No work type information available.")

	print("\n--- INTERPRETATION ---")
	print("If missingness is NOT predicted by day/hour/work_type, it's likely MAR (missing at random)")
	print("conditional on subject, and you can relax the 0.8 threshold without bias concerns.")
	print("If missingness IS predicted by day/hour/work_type, investigate why before deciding.")

	# ==========================================================================
	# DAILY-LEVEL COVERAGE ANALYSIS
	# ==========================================================================
	print("\n" + "=" * 70)
	print("DAILY-LEVEL COVERAGE: What matters for daily aggregation")
	print("=" * 70)
	
	# Compute daily-level coverage: does subject have ANY right-side data on each day?
	daily = df.groupby(["subject_id", "date", "side"])["EMG_session.duration_s"].sum().reset_index()
	daily_pivot = daily.pivot_table(
		index=["subject_id", "date"],
		columns="side",
		values="EMG_session.duration_s",
		aggfunc="sum"
	).reset_index()
	
	daily_pivot["has_right"] = daily_pivot.get("right", 0).notna() & (daily_pivot.get("right", 0) > 0)
	daily_pivot["has_left"] = daily_pivot.get("left", 0).notna() & (daily_pivot.get("left", 0) > 0)
	daily_pivot["has_any"] = daily_pivot["has_right"] | daily_pivot["has_left"]
	
	# Per-subject daily coverage
	daily_coverage = daily_pivot.groupby("subject_id").agg(
		total_days=("has_any", "sum"),
		days_with_right=("has_right", "sum"),
		days_with_left=("has_left", "sum"),
	).reset_index()
	daily_coverage["daily_right_coverage"] = daily_coverage["days_with_right"] / daily_coverage["total_days"]
	daily_coverage["days_missing_right"] = daily_coverage["total_days"] - daily_coverage["days_with_right"]
	
	# Merge with session-level coverage for comparison
	daily_coverage = daily_coverage.merge(
		subject_coverage[["subject_id", "right_coverage", "flag_problematic_right_side"]],
		on="subject_id"
	)
	daily_coverage = daily_coverage.rename(columns={"right_coverage": "session_right_coverage"})
	daily_coverage = daily_coverage.sort_values("daily_right_coverage")
	
	print("\n--- All subjects: daily vs session coverage ---")
	print(daily_coverage[["subject_id", "total_days", "days_with_right", "days_missing_right", 
						   "daily_right_coverage", "session_right_coverage", "flag_problematic_right_side"]]
		  .to_string(index=False))
	
	# Focus on the "problematic" subjects
	problematic_daily = daily_coverage[daily_coverage["flag_problematic_right_side"]]
	print(f"\n--- The 11 'problematic' subjects at DAILY level ---")
	print(problematic_daily[["subject_id", "total_days", "days_with_right", "days_missing_right",
							  "daily_right_coverage", "session_right_coverage"]]
		  .to_string(index=False))
	
	# Suggest thresholds
	print("\n--- DECISION GUIDE ---")
	print("For daily-level analysis, consider these thresholds:")
	print("  - Strict:  daily_right_coverage >= 0.8 (at least 4/5 days have right data)")
	print("  - Moderate: daily_right_coverage >= 0.6 (at least 3/5 days)")
	print("  - Loose:   daily_right_coverage >= 0.4 (at least 2/5 days)")
	print()
	
	for thresh in [0.8, 0.6, 0.4]:
		excluded = daily_coverage[daily_coverage["daily_right_coverage"] < thresh]
		print(f"  Threshold {thresh:.0%}: exclude {len(excluded)} subjects -> {excluded['subject_id'].tolist()}")
	
	# Export
	daily_coverage_path = output_dir / "emg_daily_coverage.csv"
	daily_coverage.to_csv(daily_coverage_path, index=False)
	print(f"\nDaily coverage exported to: {daily_coverage_path}")

	# ==========================================================================
	# MVC QC: daily p90 above expected range
	# ==========================================================================
	print("\n" + "=" * 70)
	print("MVC QC: Daily EMG p90 above expected range")
	print("=" * 70)

	daily_p90 = load_emg_daily_p90_by_side(profiles)
	if daily_p90.empty:
		print("  No daily EMG p90 metrics found in profiles.")
	else:
		daily_p90 = daily_p90.merge(metadata, on="subject_id", how="left")
		daily_p90["work_type"] = daily_p90["work_type"].fillna("Unknown")
		daily_p90_path = output_dir / "emg_daily_p90.csv"
		daily_p90.to_csv(daily_p90_path, index=False)
		print(f"  Daily p90 exported to: {daily_p90_path}")

		daily_p90_right = daily_p90[daily_p90["side"] == "right"].copy()
		if daily_p90_right.empty:
			print("  No right-side daily p90 data available.")
		else:
			median_p90 = daily_p90_right["emg_daily_p90"].median()
			if pd.isna(median_p90):
				print("  Unable to compute median p90 (no numeric values).")
			else:
				if median_p90 > 2:
					base_thr = 100.0
					high_thr = 110.0
					scale_label = "%MVC scale (0-100)"
				else:
					base_thr = 1.0
					high_thr = 1.1
					scale_label = "MVC fraction scale (0-1)"

				daily_p90_right["gt_base"] = daily_p90_right["emg_daily_p90"] > base_thr
				daily_p90_right["gt_high"] = daily_p90_right["emg_daily_p90"] > high_thr

				qc_rates = (
					daily_p90_right.groupby("work_type")[["gt_base", "gt_high"]]
					.mean()
					.round(3)
				)
				qc_counts = daily_p90_right.groupby("work_type")["emg_daily_p90"].count()
				qc = qc_rates.join(qc_counts.rename("n_days"))
				qc = qc.rename(
					columns={
						"gt_base": f"pct_gt_{base_thr:g}",
						"gt_high": f"pct_gt_{high_thr:g}",
					}
				)

				print(f"  Scale detected: {scale_label}")
				print(f"  Thresholds: {base_thr:g} and {high_thr:g}")
				print("\n--- Proportion of days with p90 above threshold (RIGHT side) ---")
				print(qc)

				# Chi-square test for group difference at base threshold
				try:
					from scipy.stats import chi2_contingency
					contingency = pd.crosstab(daily_p90_right["work_type"], daily_p90_right["gt_base"])
					if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
						chi2, p, dof, _ = chi2_contingency(contingency)
						print("\n--- Chi-square test: work_type vs p90 > threshold ---")
						print(f"  Chi2 = {chi2:.2f}, p = {p:.4f}")
						if p < 0.05:
							print("  WARNING: Exceedances differ by work_type (possible MVC bias).")
						else:
							print("  OK: No evidence that exceedances differ by work_type.")
					else:
						print("\n--- Chi-square test skipped: not enough groups/levels ---")
				except Exception as e:
					print(f"\n--- Chi-square test error: {e}")


if __name__ == "__main__":
	main()
