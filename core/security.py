"""core/security.py — Input sanitisation helpers."""

import re
from typing import List

_SAFE_RE = re.compile(r"^[a-zA-Z0-9 _\-]{1,120}$")


def sanitize_interests(interests: List[str]) -> List[str]:
    """Strip, lowercase, deduplicate, and validate interest strings."""
    cleaned: List[str] = []
    seen: set = set()
    for raw in interests:
        value = raw.strip()
        if not value:
            continue
        if not _SAFE_RE.match(value):
            raise ValueError(
                f"Interest '{raw}' contains invalid characters. "
                "Only letters, numbers, spaces, hyphens and underscores are allowed."
            )
        lower = value.lower()
        if lower not in seen:
            seen.add(lower)
            cleaned.append(value)          # preserve original casing for display
    return cleaned
