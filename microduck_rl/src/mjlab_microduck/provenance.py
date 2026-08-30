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


def provenance(policy_path: str | Path, task_id: str) -> dict:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "git_commit": git_commit(),
        "task": task_id,
        "policy_file": str(policy_path),
        "policy_sha256": file_sha256(policy_path),
    }
