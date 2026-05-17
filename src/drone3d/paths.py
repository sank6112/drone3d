"""Inject foundation-model repos into sys.path.

The upstream repos (mast3r, cut3r, Point3R, StreamVGGT, monst3r) are not
pip-installable; they ship as plain folders. This helper makes their packages
importable by adding their roots to sys.path.

VGGT is pip-installed (editable) and does not need injection.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = PROJECT_ROOT / "third_party"

# repo_dir -> list of subpaths to add to sys.path (in order)
_REPOS: dict[str, list[str]] = {
    "mast3r": ["", "dust3r"],            # exposes `mast3r.*` and `dust3r.*`
    "cut3r": ["", "src"],                # exposes croco/, dust3r/ under src/
    "Point3R": ["", "src"],
    "StreamVGGT": ["", "src"],
    "monst3r": ["", "third_party/dust3r"],  # MonST3R has its own DUSt3R copy
}


def inject(*repos: str) -> None:
    """Add given repos' relevant subpaths to sys.path.

    Call once at the top of any script using a sys.path-style backbone.
    """
    targets = repos or tuple(_REPOS.keys())
    for repo in targets:
        if repo not in _REPOS:
            raise KeyError(f"unknown repo {repo!r}; known: {sorted(_REPOS)}")
        base = THIRD_PARTY / repo
        if not base.is_dir():
            raise FileNotFoundError(f"missing repo dir: {base}")
        for sub in _REPOS[repo]:
            p = str(base / sub) if sub else str(base)
            if p not in sys.path:
                sys.path.insert(0, p)


def checkpoint(name: str) -> Path:
    """Return absolute path to a checkpoint under ./checkpoints, asserting it exists."""
    p = PROJECT_ROOT / "checkpoints" / name
    if not p.exists():
        raise FileNotFoundError(f"checkpoint missing: {p}")
    return p
