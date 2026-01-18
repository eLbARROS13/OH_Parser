"""
OH Parser Loader.

Functions to discover and load OH profile JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


# Standard OH profile filename suffix
_OH_PROFILE_SUFFIX = "_OH_profile.json"


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def load_profiles(
    directory: Union[str, Path],
    subject_ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Load all OH profiles from a directory.
    
    :param directory: Path to directory containing OH profiles.
    :param subject_ids: Optional list of specific subject IDs to load (None = all).
    :param verbose: If True, print loading progress.
    :returns: Dictionary mapping subject_id -> profile dict.
    :raises FileNotFoundError: If directory doesn't exist.
    """
    dir_path = Path(directory)
    profile_paths = _discover_oh_profiles(dir_path)
    
    if not profile_paths:
        if verbose:
            print(f"[oh_parser] No OH profiles found in {dir_path}")
        return {}
    
    profiles: Dict[str, dict] = {}
    errors: List[str] = []
    
    for path in profile_paths:
        subject_id = _extract_subject_id(path)
        
        # Filter by subject_ids if specified
        if subject_ids is not None and subject_id not in subject_ids:
            continue
        
        try:
            profiles[subject_id] = load_profile(path)
        except json.JSONDecodeError as e: # Catch JSON decode errors
            errors.append(f"{subject_id}: JSON decode error - {e}")
        except Exception as e: # Catch all other exceptions
            errors.append(f"{subject_id}: {e}")
    
    if verbose:
        print(f"[oh_parser] Loaded {len(profiles)} OH profiles from {dir_path}")
        if errors:
            print(f"[oh_parser] {len(errors)} profiles failed to load:")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
    
    return profiles


def load_profile(filepath: Union[str, Path]) -> dict:
    """
    Load a single OH profile JSON file.
    
    :param filepath: Path to OH profile JSON file.
    :returns: Parsed JSON as dictionary.
    :raises FileNotFoundError: If file doesn't exist.
    :raises json.JSONDecodeError: If file is not valid JSON.
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"OH profile not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_subjects(profiles: Dict[str, dict]) -> List[str]:
    """
    Get sorted list of subject IDs from loaded profiles.
    
    :param profiles: Dictionary mapping subject_id -> profile dict.
    :returns: Sorted list of subject IDs.
    """
    return sorted(profiles.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

def get_profile(profiles: Dict[str, dict], subject_id: str) -> Optional[dict]:
    """
    Get a single profile by subject ID.
    
    :param profiles: Dictionary mapping subject_id -> profile dict.
    :param subject_id: Subject ID to retrieve.
    :returns: Profile dict or None if not found.
    """
    return profiles.get(subject_id)


# =============================================================================
# PRIVATE FUNCTIONS
# =============================================================================

def _discover_oh_profiles(directory: Union[str, Path]) -> List[Path]:
    """
    Discover all OH profile JSON files in a directory.
    
    :param directory: Path to directory containing OH profiles.
    :returns: List of paths to OH profile files.
    :raises FileNotFoundError: If directory doesn't exist.
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"OH profiles directory not found: {dir_path}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    
    # Find all files matching the OH profile pattern
    # Exclude macOS hidden files (._*) which are metadata, not real profiles
    profiles = [
        p for p in dir_path.glob(f"*{_OH_PROFILE_SUFFIX}")
        if not p.name.startswith("._")
    ]
    
    return sorted(profiles)


def _extract_subject_id(filepath: Path) -> str:
    """
    Extract subject ID from OH profile filename.
    
    Expected format: "{subject_id}_OH_profile.json"
    
    :param filepath: Path to OH profile file.
    :returns: Subject ID string.
    """
    filename = filepath.name
    if filename.endswith(_OH_PROFILE_SUFFIX):
        return filename[:-len(_OH_PROFILE_SUFFIX)]  # Remove the suffix to get subject_id
    return filepath.stem
