"""Shared helpers for the pipeline tests.

Deliberately minimal -- path fixtures, a cached MOSEK-license probe, and a
subprocess runner.  Nothing here is autouse, so the self-sufficient
test_portfolio_* / test_svm_* modules are unaffected.

Markers ``slow`` and ``mosek`` are registered in pyproject.toml; the default
addopts exclude ``slow``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parent

_MOSEK_STATE = {}


def mosek_available():
    """True iff a working MOSEK license is available.

    Probes with one trivial MOSEK solve; the result is cached for the rest of
    the session so the license check runs at most once.
    """
    if "ok" not in _MOSEK_STATE:
        try:
            import mosek

            with mosek.Env() as env:
                with env.Task(0, 0) as task:
                    task.appendvars(1)
                    task.putvarbound(0, mosek.boundkey.ra, 0.0, 1.0)
                    task.putcj(0, 1.0)
                    task.putobjsense(mosek.objsense.minimize)
                    task.optimize()
            _MOSEK_STATE["ok"] = True
        except Exception:
            _MOSEK_STATE["ok"] = False
    return _MOSEK_STATE["ok"]


def skip_without_mosek():
    """Skip the calling test when no MOSEK license is available."""
    if not mosek_available():
        pytest.skip("MOSEK license not available")


@pytest.fixture(scope="session")
def mosek_license():
    """Fixture form of the MOSEK probe: request it to skip without a license."""
    skip_without_mosek()


@pytest.fixture(scope="session")
def paper_dir():
    """Absolute path of the paper_experiments/ project directory."""
    return PAPER_DIR


@pytest.fixture(scope="session")
def repo_root():
    """Absolute path of the repository root (legacy drivers live here)."""
    return REPO_ROOT


def run_cli(args, cwd, env_extra=None, timeout=1800):
    """Run ``[sys.executable, *args]`` in ``cwd``; return CompletedProcess.

    ``env_extra`` entries are layered on top of the current environment
    (e.g. SLURM_ARRAY_TASK_ID for the legacy svm/portfolio MRO drivers).
    """
    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(
        [sys.executable, *[str(a) for a in args]],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_ran(proc, label=""):
    """Assert a driver subprocess exited 0, with helpful output tails."""
    assert proc.returncode == 0, (
        f"{label} driver exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
