#!/usr/bin/env python3
"""
Run Hypothesis Tests
====================

Simple entry point for running all OH Stats hypothesis tests (H1-H6).

Usage:
    python run_hypotheses.py                    # Run all
    python run_hypotheses.py H1 H2             # Run specific hypotheses
    python run_hypotheses.py --describe H1     # Print hypothesis description

Environment:
    OH_PROFILES_PATH: Path to OH profiles directory (optional)
"""
import os
import sys
import argparse

from oh_parser import load_profiles
from hypotheses import (
    run_all,
    run_hypothesis,
    summarize_results,
    apply_multiplicity_correction,
    HYPOTHESES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run OH Stats Hypothesis Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_hypotheses.py              # Run all hypotheses
    python run_hypotheses.py H1 H3 H6     # Run selected hypotheses
    python run_hypotheses.py --describe   # Show all descriptions
    python run_hypotheses.py --describe H1 H5  # Show specific descriptions
        """,
    )
    parser.add_argument(
        "hypotheses",
        nargs="*",
        choices=list(HYPOTHESES.keys()) + [],
        help="Specific hypotheses to run (default: all)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print hypothesis descriptions instead of running",
    )
    parser.add_argument(
        "--profiles-path",
        default=os.getenv("OH_PROFILES_PATH", "/Users/goncalobarros/Documents/projects/OH_profiles"),
        help="Path to OH profiles directory",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Skip multiplicity correction",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    
    args = parser.parse_args()
    hypotheses_to_run = args.hypotheses or list(HYPOTHESES.keys())
    
    # Describe mode
    if args.describe:
        for h_id in hypotheses_to_run:
            module = __import__(f"hypotheses.h{h_id[1]}_{HYPOTHESES[h_id]['name'].lower().replace(' ', '_').split()[0]}", fromlist=["describe"])
            # Actually, let's just print from config
            config = HYPOTHESES[h_id]
            print("=" * 70)
            print(f"{h_id}: {config['name']}")
            print("=" * 70)
            print(config.get("description", "No description available"))
            print()
        return 0
    
    # Run hypotheses
    print("=" * 70)
    print("OH STATS HYPOTHESIS TESTING")
    print("=" * 70)
    print(f"Profiles path: {args.profiles_path}")
    
    profiles = load_profiles(args.profiles_path)
    print(f"Loaded {len(profiles)} subjects\n")
    
    # Run all or selected hypotheses
    results = run_all(profiles, hypotheses=hypotheses_to_run, verbose=not args.quiet)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = summarize_results(results)
    print(summary.to_string(index=False))
    
    # Multiplicity correction
    if not args.no_correction:
        apply_multiplicity_correction(results, verbose=True)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
