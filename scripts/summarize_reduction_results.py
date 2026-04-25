#!/usr/bin/env python3
"""Render reduction benchmark CSV results as a README-ready markdown table."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "rtx_5060_ti" / "reduction_results.csv"


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"missing results CSV: {csv_path}", file=sys.stderr)
        print("run ./scripts/run_reduction.sh first", file=sys.stderr)
        return 1

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"results CSV has no data rows: {csv_path}", file=sys.stderr)
        return 1

    columns = [
        ("variant", "variant"),
        ("n", "n"),
        ("median latency us", "latency_median_us"),
        ("p95 latency us", "latency_p95_us"),
        ("bandwidth GB/s", "bandwidth_gb_s"),
        ("abs error", "abs_error"),
        ("rel error", "rel_error"),
        ("baseline ratio", "baseline_ratio"),
    ]

    print("| " + " | ".join(title for title, _ in columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(key, "") for _, key in columns) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
