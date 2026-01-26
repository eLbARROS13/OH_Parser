"""
OH Parser CLI.

Command-line interface for quick testing and exploration of OH profiles.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loader import load_profiles, list_subjects, get_profile
from .extract import inspect_profile, get_available_paths, summarize_profiles


DEFAULT_OH_DIRECTORY = r"E:\Backup PrevOccupAI_PLUS Data\OH_profiles_2"


def main():
    """Main entrypoint for the OH Parser CLI."""
    parser = argparse.ArgumentParser(
        prog="oh_parser",
        description="Extract data from Occupational Health profile JSON files.",
    )
    
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=DEFAULT_OH_DIRECTORY,
        help=(
            "Path to directory containing OH profile JSON files "
            f"(default: {DEFAULT_OH_DIRECTORY})."
        ),
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_subjects",
        help="List all subject IDs.",
    )
    
    parser.add_argument(
        "--inspect", "-i",
        type=str,
        metavar="SUBJECT_ID",
        help="Inspect the structure of a specific subject's profile.",
    )
    
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=3,
        help="Max depth for inspection (default: 3).",
    )
    
    parser.add_argument(
        "--paths", "-p",
        type=str,
        metavar="SUBJECT_ID",
        help="List all available paths for a subject.",
    )
    
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show data availability summary for all subjects.",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress loading messages.",
    )
    args = parser.parse_args()
    
    # Load profiles
    profiles = load_profiles(args.directory, verbose=not args.quiet)
    
    if not profiles:
        print("No profiles loaded.", file=sys.stderr)
        sys.exit(1)
    
    # Execute requested action
    if args.list_subjects:
        subjects = list_subjects(profiles)
        print(f"\nSubjects ({len(subjects)}):")
        for s in subjects:
            print(f"  {s}")
    
    elif args.inspect:
        profile = get_profile(profiles, args.inspect)
        if profile is None:
            print(f"Error: Subject '{args.inspect}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f"\nInspecting profile for subject: {args.inspect}")
        inspect_profile(profile, max_depth=args.depth)
    
    elif args.paths:
        profile = get_profile(profiles, args.paths)
        if profile is None:
            print(f"Error: Subject '{args.paths}' not found.", file=sys.stderr)
            sys.exit(1)
        paths = get_available_paths(profile, max_depth=6)
        print(f"\nAvailable paths for subject {args.paths} ({len(paths)} total):")
        for p in paths[:50]:  # Show first 50
            print(f"  {p}")
        if len(paths) > 50:
            print(f"  ... and {len(paths) - 50} more")
    
    elif args.summary:
        print("\nData availability summary:")
        df = summarize_profiles(profiles)
        print(df.to_string(index=False))
    
    else:
        # Default: show basic info
        subjects = list_subjects(profiles)
        print(f"\n{'='*60}")
        print(f"OH Parser - Loaded {len(subjects)} profiles")
        print(f"{'='*60}")
        print(f"\nSubjects: {', '.join(subjects[:10])}", end="")
        if len(subjects) > 10:
            print(f" ... and {len(subjects) - 10} more")
        else:
            print()
        print(f"\nUsage examples:")
        print(f"  oh_parser \"{args.directory}\" --list")
        print(f"  oh_parser \"{args.directory}\" --inspect {subjects[0]}")
        print(f"  oh_parser \"{args.directory}\" --summary")

