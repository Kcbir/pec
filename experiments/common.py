import csv
import sys
from fractions import Fraction
from pathlib import Path


class Gate:
    def __init__(self, title: str, subtitle: str = "", width: int = 78):
        self.results: list[tuple[str, bool]] = []
        self.width = width
        print("=" * width)
        print(title)
        print("=" * width)
        if subtitle:
            print(subtitle)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
        return bool(ok)

    def section(self, heading: str) -> None:
        print(f"\n{heading}")

    @property
    def passed(self) -> int:
        return sum(1 for _, ok in self.results if ok)

    @property
    def all_ok(self) -> bool:
        return all(ok for _, ok in self.results)

    def finish(self, verdict_ok: str = "", verdict_bad: str = "") -> None:
        print("\n" + "=" * self.width)
        print(f"{self.passed}/{len(self.results)} checks passed")
        if self.all_ok:
            if verdict_ok:
                print(verdict_ok)
        else:
            print(verdict_bad or "\nSome checks failed - see above.")
        print("=" * self.width)
        sys.exit(0 if self.all_ok else 1)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def _fmt(v):
    if isinstance(v, Fraction):
        return str(v)
    return v


def write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    path = RESULTS_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    return path


def announce(paths: list[Path]) -> None:
    print("\n  CSVs written:")
    for p in paths:
        print(f"    {p.relative_to(ROOT)}")
