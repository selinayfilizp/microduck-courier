"""Provenance stamps for published eval results and rollout sidecars.

The published JSON used to carry only a seed; the clip-to-checkpoint-to-eval
linkage rested entirely on directory convention. Every sidecar now records
when it was made, from which commit, and the SHA-256 of the exact policy file,
so a number in the README can be traced to bytes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repo_relative(path: str | Path) -> str:
    """Repo-relative form of a path, for sidecars published in the repo.

    Absolute local paths leak the author's machine layout into public
    artifacts; the SHA-256 already identifies the exact bytes. Falls back to
    the basename when the path lies outside the repo.
    """
    p = Path(path).resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        root = Path(out.decode().strip())
        return p.relative_to(root).as_posix()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return p.name


def provenance(policy_path: str | Path, task_id: str) -> dict:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "git_commit": git_commit(),
        "task": task_id,
        "policy_file": repo_relative(policy_path),
        "policy_sha256": file_sha256(policy_path),
    }
