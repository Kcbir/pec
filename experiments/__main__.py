import subprocess
import sys
import time
from pathlib import Path

from experiments import CHECKS, REPORTS

ROOT = Path(__file__).resolve().parent.parent


def main(groups: list[str]) -> int:
    unknown = set(groups) - {m.split(".")[0] for m in CHECKS + REPORTS}
    if unknown:
        print(f"unknown group: {', '.join(sorted(unknown))}")
        return 2
    selected = [m for m in CHECKS + REPORTS if not groups or m.split(".")[0] in groups]
    failed = []
    for module in selected:
        print(f"\n{'#' * 76}\n# experiments.{module}\n{'#' * 76}", flush=True)
        start = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", f"experiments.{module}"], cwd=ROOT)
        print(f"\n# exit {r.returncode} after {time.perf_counter() - start:.1f} s", flush=True)
        if r.returncode != 0:
            failed.append(module)
    print("\n" + "=" * 76)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    else:
        print(f"{len(selected)} experiments completed")
    print("=" * 76)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
