"""Utility functions shared across modules."""
from __future__ import annotations

from difflib import SequenceMatcher


def string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using SequenceMatcher.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
