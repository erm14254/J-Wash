#!/usr/bin/env python3
"""Install the repository-pinned Jacobian Lens checkout in editable mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMIT_FILE = ROOT / "requirements" / "jacobian-lens.commit"
CHECKOUT = ROOT / "vendor" / "jacobian-lens"
REPOSITORY = "https://github.com/anthropics/jacobian-lens.git"


class BootstrapError(RuntimeError):
    """Raised when the pinned checkout cannot be prepared or installed."""


def run(*args: str, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise BootstrapError(f"required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no command output").strip()
        raise BootstrapError(
            f"command failed ({' '.join(args)}): {detail}"
        ) from exc
    return completed.stdout.strip()


def pinned_commit() -> str:
    try:
        commit = COMMIT_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BootstrapError(f"cannot read pin file {COMMIT_FILE}: {exc}") from exc
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise BootstrapError(f"invalid full commit SHA in {COMMIT_FILE}: {commit!r}")
    return commit


def prepare_checkout(commit: str) -> None:
    if CHECKOUT.exists() and not (CHECKOUT / ".git").exists():
        raise BootstrapError(
            f"{CHECKOUT} exists but is not a Git checkout; move or remove it first"
        )
    if not CHECKOUT.exists():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", REPOSITORY, str(CHECKOUT))

    # Fetch the pinned object explicitly so a shallow or stale checkout is safe.
    run("git", "fetch", "--force", "origin", commit, cwd=CHECKOUT)
    run("git", "checkout", "--detach", "--force", commit, cwd=CHECKOUT)
    head = run("git", "rev-parse", "HEAD", cwd=CHECKOUT)
    if head != commit:
        raise BootstrapError(
            f"Jacobian Lens checkout verification failed: expected {commit}, got {head}"
        )


def main() -> int:
    try:
        commit = pinned_commit()
        prepare_checkout(commit)
        run(sys.executable, "-m", "pip", "install", "-e", str(CHECKOUT))
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Jacobian Lens {commit} is installed editable from {CHECKOUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
