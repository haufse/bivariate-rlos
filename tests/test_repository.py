"""pytest entry point for the repository checks.

    pytest -q tests/                  # unit checks of Sec. 8 and regeneration checks (about 1-2 min)
    BRLOS_SLOW=1 pytest -q tests/     # additionally the bootstrap/oracle check (about 2-3 min more)

The Sec. 8 checks in sec8_simulations/tests/check_*.py are plain scripts (they print PASS/FAIL
lines and a final "<passed>/<total> checks passed" line); they are run here as subprocesses.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEC8 = ROOT / "sec8_simulations"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1")


def _run(args, cwd):
    p = subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _all_passed(out):
    m = re.findall(r"(\d+)/(\d+) checks passed", out)
    assert m, out[-2000:]
    a, b = map(int, m[-1])
    assert a == b, out[-4000:]


def test_sec8_units():
    rc, out = _run(["tests/check_units.py"], SEC8)
    assert rc == 0, out[-4000:]
    _all_passed(out)


def test_sec8_bck_and_reproduction():
    rc, out = _run(["tests/check_bck.py"], SEC8)
    assert rc == 0, out[-4000:]
    _all_passed(out)


@pytest.mark.skipif(not os.environ.get("BRLOS_SLOW"), reason="slow; set BRLOS_SLOW=1")
def test_sec8_oracle():
    rc, out = _run(["tests/check_oracle.py"], SEC8)
    assert rc == 0, out[-4000:]
    assert "relative Frobenius error" in out


def test_sec8_headline_reproduces_stored_csv(tmp_path):
    """headline.py on the shipped summaries reproduces data/sec8/main_b5/headline_numbers.csv."""
    import pandas as pd
    ref = pd.read_csv(ROOT / "data/sec8/main_b5/headline_numbers.csv", dtype=str, keep_default_na=False)
    new = tmp_path / "main_b5"
    new.mkdir()
    for f in (ROOT / "data/sec8/main_b5").glob("*.csv"):
        if f.name != "headline_numbers.csv":
            (new / f.name).write_bytes(f.read_bytes())
    rc, out = _run(["headline.py", "--new", str(new)], SEC8)
    assert rc == 0, out[-4000:]
    got = pd.read_csv(new / "headline_numbers.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(ref, got)
