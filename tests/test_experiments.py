import subprocess
import sys
from pathlib import Path

import pytest

from experiments import CHECKS

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module", CHECKS)
def test_check_passes(module):
    r = subprocess.run([sys.executable, "-m", f"experiments.{module}"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
