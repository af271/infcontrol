"""Summarize ECAN/WILLIAM proof-search logs into comparable run metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

TIMESTAMP = r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}:\d{3})\]"
ITERATION_RE = re.compile(TIMESTAMP + r" \[INFO\] Iteration (?P<index>\d+): (?P<message>.*)")
UNION_RE = re.compile(
    TIMESTAMP + r" \[INFO\]\s+\[union\] ecan=(?P<ecan>\d+) will=(?P<william>\d+) union=(?P<union>\d+)"
)
RANKING_RE = re.compile(
    TIMESTAMP
    + r" \[INFO\]\s+\[ranking\] index=(?P<index>\d+) theorem=(?P<theorem>\S+)"
    + r" mode=(?P<mode>\S+) budget=(?P<budget>\d+)"
    + r" ecan=(?P<ecan>\([^)]*\)) william=(?P<william>\([^)]*\))"
    + r" selected=(?P<selected>\([^)]*\))"
)
TIER_RE = re.compile(
    TIMESTAMP + r" \[INFO\]\s+\[tier\] focused=(?P<focused>\d+) full=(?P<full>\d+) af-hit=(?P<hit>true|false)"
)
TIMING_RE = re.compile(
    TIMESTAMP
    + r" \[INFO\]\s+\[timing\] focus-seconds=(?P<focus>[\d.eE+-]+)"
    + r" fallback-seconds=(?P<fallback>[\d.eE+-]+)"
    + r" timeout=(?P<timeout>none|focused|fallback)"
)
PROOF_RE = re.compile(TIMESTAMP + r" \[INFO\] Found proof:")
MISS_RE = re.compile(TIMESTAMP + r" \[INFO\] No proof found")
COMPLETE_RE = re.compile(TIMESTAMP + r" \[INFO\] \[bg\] daemon sweeps=")
WALL_RE = re.compile(TIMESTAMP + r" \[INFO\]\s+\[wall\] index=(?P<index>\d+) theorem-seconds=(?P<seconds>[\d.eE+-]+)")


def _time(match: re.Match[str]) -> datetime:
    return datetime.strptime(match["timestamp"], "%Y-%m-%d %H:%M:%S:%f").replace(tzinfo=UTC)


def _seconds(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _labels(value: str) -> list[str]:
    return value[1:-1].split()


def parse_search_log(text: str) -> list[dict[str, Any]]:
    """Parse one or more appended proof-search runs from logger output."""
    runs: list[dict[str, Any]] = []
    run: dict[str, Any] | None = None
    attempt: dict[str, Any] | None = None

    def finish_run() -> None:
        nonlocal run, attempt
        if run is None:
            return
        attempts = run.pop("attempts")
        completed = [item for item in attempts if item.get("result") is not None]
        tiered = [item for item in completed if "tier_hit" in item and item.get("timeout_tier") != "focused"]
        hits = [item for item in tiered if item["tier_hit"]]
        fallbacks = [item for item in tiered if not item["tier_hit"]]
        search_times = [
            (item["focus_seconds"] if "focus_seconds" in item else (item["tier_time"] - item["start"]).total_seconds())
            + item.get("fallback_seconds", 0.0)
            for item in completed
            if "focus_seconds" in item or "tier_time" in item
        ]
        focus_times = [
            item.get(
                "focus_seconds",
                (item["tier_time"] - item["start"]).total_seconds(),
            )
            for item in tiered
        ]
        fallback_times = [item["fallback_seconds"] for item in fallbacks if "fallback_seconds" in item]
        union_sizes = [item["union_size"] for item in attempts if "union_size" in item]
        budgets = [item["focus_budget"] for item in attempts if "focus_budget" in item]
        selected_sizes = [len(item["selected_labels"]) for item in attempts if "selected_labels" in item]
        filled_budgets = [
            len(item["selected_labels"]) == min(item["focus_budget"], item["full_size"])
            for item in attempts
            if {"selected_labels", "focus_budget", "full_size"} <= item.keys()
        ]
        search_seconds_total = float(sum(search_times))
        fallback_seconds_total = float(sum(fallback_times))
        theorems = []
        for item in attempts:
            theorem = {key: value for key, value in item.items() if key not in {"start", "tier_time"}}
            if "tier_time" in item and "focus_seconds" not in item:
                theorem["focus_seconds"] = (item["tier_time"] - item["start"]).total_seconds()
            theorems.append(theorem)
        run.update(
            {
                "theorems_started": len(attempts),
                "theorems_finished": len(completed),
                "proofs_found": sum(item["result"] == "proof" for item in completed),
                "timeouts": sum(item["result"] == "timeout" for item in completed),
                "focus_hits": len(hits),
                "fallbacks": len(fallbacks),
                "focus_hit_rate": len(hits) / len(tiered) if tiered else float("nan"),
                "focus_seconds": _seconds(focus_times),
                "search_seconds": _seconds(search_times),
                "search_seconds_total": search_seconds_total,
                "fallback_seconds_total": fallback_seconds_total,
                "fallback_seconds_share": (
                    fallback_seconds_total / search_seconds_total if search_seconds_total else float("nan")
                ),
                "union_size_mean": (float(np.mean(union_sizes)) if union_sizes else float("nan")),
                "selector_modes": sorted({item["selector_mode"] for item in attempts if "selector_mode" in item}),
                "focus_budget_mean": (float(np.mean(budgets)) if budgets else float("nan")),
                "selected_size_mean": (float(np.mean(selected_sizes)) if selected_sizes else float("nan")),
                "budget_fill_rate": (float(np.mean(filled_budgets)) if filled_budgets else float("nan")),
                "theorems": theorems,
            }
        )
        if run.get("end") is not None:
            run["wall_seconds"] = (run["end"] - run["start"]).total_seconds()
        run["start"] = run["start"].isoformat()
        if run.get("end") is not None:
            run["end"] = run["end"].isoformat()
        runs.append(run)
        run = None
        attempt = None

    for line in text.splitlines():
        iteration = ITERATION_RE.match(line)
        if iteration:
            timestamp = _time(iteration)
            if iteration["index"] == "2":
                finish_run()
                run = {
                    "run": len(runs) + 1,
                    "start": timestamp,
                    "end": None,
                    "complete": False,
                    "attempts": [],
                }
            if run is not None and "search proof of " in iteration["message"]:
                label = iteration["message"].split("search proof of ", 1)[1].split(" up to size", 1)[0]
                attempt = {"index": int(iteration["index"]), "label": label, "start": timestamp}
                run["attempts"].append(attempt)
            continue

        if run is None:
            continue
        wall = WALL_RE.match(line)
        if wall and attempt is not None:
            if int(wall["index"]) != attempt["index"]:
                raise ValueError("wall-time record does not match current theorem")
            seconds = float(wall["seconds"])
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError("theorem wall time must be finite and non-negative")
            attempt["theorem_wall_seconds"] = seconds
            continue
        union = UNION_RE.match(line)
        if union and attempt is not None:
            attempt.update(
                ecan_size=int(union["ecan"]),
                william_size=int(union["william"]),
                union_size=int(union["union"]),
            )
            continue
        ranking = RANKING_RE.match(line)
        if ranking and attempt is not None:
            if int(ranking["index"]) != attempt["index"] or ranking["theorem"] != attempt["label"]:
                raise ValueError("ranking record does not match current theorem")
            attempt.update(
                selector_mode=ranking["mode"],
                focus_budget=int(ranking["budget"]),
                ecan_ranking=_labels(ranking["ecan"]),
                william_ranking=_labels(ranking["william"]),
                selected_labels=_labels(ranking["selected"]),
            )
            continue
        tier = TIER_RE.match(line)
        if tier and attempt is not None:
            attempt.update(
                tier_time=_time(tier),
                focused_size=int(tier["focused"]),
                full_size=int(tier["full"]),
                tier_hit=tier["hit"] == "true",
            )
            continue
        timing = TIMING_RE.match(line)
        if timing and attempt is not None:
            attempt.update(
                focus_seconds=float(timing["focus"]),
                fallback_seconds=float(timing["fallback"]),
            )
            if timing["timeout"] != "none":
                attempt.update(
                    result="timeout",
                    timeout_tier=timing["timeout"],
                    total_seconds=(_time(timing) - attempt["start"]).total_seconds(),
                )
            continue
        result = PROOF_RE.match(line) or MISS_RE.match(line)
        if result and attempt is not None and attempt.get("result") != "timeout":
            end = _time(result)
            attempt["result"] = "proof" if PROOF_RE.match(line) else "miss"
            attempt["total_seconds"] = (end - attempt["start"]).total_seconds()
            if not attempt.get("tier_hit", True) and "tier_time" in attempt and "fallback_seconds" not in attempt:
                attempt["fallback_seconds"] = (end - attempt["tier_time"]).total_seconds()
            continue
        complete = COMPLETE_RE.match(line)
        if complete:
            run["complete"] = True
            run["end"] = _time(complete)

    finish_run()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Search log path, or '-' for stdin")
    parser.add_argument("--summary-only", action="store_true", help="Omit per-theorem records")
    args = parser.parse_args()
    text = sys.stdin.read() if args.log_path == "-" else Path(args.log_path).read_text(encoding="utf-8")
    runs = parse_search_log(text)
    if args.summary_only:
        runs = [{key: value for key, value in run.items() if key != "theorems"} for run in runs]
    print(json.dumps(runs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
