"""Aggregate perf.jsonl telemetry into a per-game performance report.

Usage:
    uv run python scripts/perf_report.py                 # latest game
    uv run python scripts/perf_report.py --all           # every game found
    uv run python scripts/perf_report.py path/to/perf_*.jsonl

Reports:
  * APM (actions/min), calls/min, think vs tool vs blocked time
  * time blocked by category (diplomacy, production, ...)
  * yield curves: science/culture/gold/faith per turn
  * empire curves: score, cities, population, districts per turn
  * slowest tools overall
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

LOCAL_DIR = Path.home() / ".civ6-mcp"


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(blocks[int((v - lo) / span * 7)] for v in values)


def report(path: str) -> None:
    rows = [r for r in load_rows(path) if r.get("turn") is not None]
    if not rows:
        print(f"{path}: no turn_perf rows")
        return
    rows.sort(key=lambda r: r["turn"])

    turns = len(rows)
    wall = sum(r["wall_s"] for r in rows)
    calls = sum(r["tool_calls"] for r in rows)
    actions = sum(r["actions"] for r in rows)
    errors = sum(r.get("tool_errors", 0) for r in rows)
    tool_time = sum(r.get("tool_time_s", 0) for r in rows)
    blocked = sum(r.get("blocked_total_s", 0) for r in rows)
    think = max(wall - tool_time, 0)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Turns: {rows[0]['turn']}–{rows[-1]['turn']} ({turns} played) | "
          f"Wall: {wall/60:.1f} min | {wall/turns:.0f}s/turn")
    print(f"APM: {actions/(wall/60):.1f} overall | "
          f"calls/min: {calls/(wall/60):.1f} | "
          f"actions/turn: {actions/turns:.1f} | calls/turn: {calls/turns:.1f} | "
          f"errors: {errors} ({100*errors/max(calls,1):.1f}%)")
    print(f"Time split: think {think/wall*100:.0f}% | "
          f"tools {tool_time/wall*100:.0f}% | "
          f"blocked {blocked/wall*100:.0f}%")

    # Blocked by category
    by_cat: dict[str, float] = defaultdict(float)
    for r in rows:
        for k, v in (r.get("blocked_s") or {}).items():
            by_cat[k] += v
    if by_cat:
        cats = ", ".join(f"{k}: {v:.0f}s" for k, v in
                         sorted(by_cat.items(), key=lambda kv: -kv[1]))
        print(f"Blocked by: {cats}")

    # Slowest tools
    tool_totals: dict[str, float] = defaultdict(float)
    for r in rows:
        for k, v in (r.get("slowest_tools") or {}).items():
            tool_totals[k] += v
    top = sorted(tool_totals.items(), key=lambda kv: -kv[1])[:8]
    if top:
        print("Slowest tools: " + ", ".join(f"{k} {v:.0f}s" for k, v in top))

    # Curves
    print("\nCurves (per turn):")
    for label, key in [
        ("APM      ", "apm"), ("wall_s   ", "wall_s"),
        ("science  ", "snap_science"), ("culture  ", "snap_culture"),
        ("gold     ", "snap_gold"), ("faith    ", "snap_faith"),
        ("score    ", "snap_score"), ("cities   ", "snap_cities"),
        ("pop      ", "snap_population"), ("districts", "snap_districts"),
    ]:
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"  {label} {sparkline(vals)}  "
                  f"first={vals[0]:.0f} last={vals[-1]:.0f} max={max(vals):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="perf_*.jsonl files")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    paths = args.paths
    if not paths:
        found = sorted(glob.glob(str(LOCAL_DIR / "perf_*.jsonl")),
                       key=os.path.getmtime)
        if not found:
            print(f"No perf_*.jsonl files in {LOCAL_DIR}")
            return
        paths = found if args.all else [found[-1]]
    for p in paths:
        report(p)


if __name__ == "__main__":
    main()
