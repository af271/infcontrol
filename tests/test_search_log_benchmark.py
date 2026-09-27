import pytest

from infcontrol.search_log_benchmark import parse_search_log


def test_parse_search_log_separates_complete_and_partial_runs() -> None:
    log = """\
[2026-09-01 12:00:00:000] [INFO] Iteration 2: add axiom ax-mp
[2026-09-01 12:00:01:000] [INFO] Iteration 6: ECAN-guided search proof of first up to size 3
[2026-09-01 12:00:01:100] [INFO]   [union] ecan=2 will=2 union=3
[2026-09-01 12:00:01:200] [INFO]   [ranking] index=6 theorem=first mode=union budget=3 ecan=(a b c d) william=(b d a c) selected=(a b c)
[2026-09-01 12:00:02:000] [INFO]   [tier] focused=3 full=5 af-hit=false
[2026-09-01 12:00:05:000] [INFO] Found proof: proof
[2026-09-01 12:00:06:000] [INFO] [bg] daemon sweeps=2 final-funds=1.0
[2026-09-01 13:00:00:000] [INFO] Iteration 2: add axiom ax-mp
[2026-09-01 13:00:01:000] [INFO] Iteration 6: ECAN-guided search proof of stuck up to size 3
[2026-09-01 13:00:01:100] [INFO]   [union] ecan=2 will=2 union=4
"""

    complete, partial = parse_search_log(log)

    assert complete["complete"] is True
    assert complete["proofs_found"] == 1
    assert complete["fallbacks"] == 1
    assert complete["focus_hit_rate"] == pytest.approx(0.0)
    assert complete["fallback_seconds_total"] == pytest.approx(3.0)
    assert complete["fallback_seconds_share"] == pytest.approx(0.75)
    assert complete["selector_modes"] == ["union"]
    assert complete["focus_budget_mean"] == pytest.approx(3.0)
    assert complete["selected_size_mean"] == pytest.approx(3.0)
    assert complete["budget_fill_rate"] == pytest.approx(1.0)
    assert complete["wall_seconds"] == pytest.approx(6.0)
    assert complete["union_size_mean"] == pytest.approx(3.0)
    assert complete["theorems"] == [
        {
            "ecan_size": 2,
            "fallback_seconds": 3.0,
            "focus_budget": 3,
            "focus_seconds": 1.0,
            "focused_size": 3,
            "full_size": 5,
            "ecan_ranking": ["a", "b", "c", "d"],
            "index": 6,
            "label": "first",
            "result": "proof",
            "selected_labels": ["a", "b", "c"],
            "selector_mode": "union",
            "tier_hit": False,
            "total_seconds": 4.0,
            "union_size": 3,
            "william_ranking": ["b", "d", "a", "c"],
            "william_size": 2,
        }
    ]
    assert partial["complete"] is False
    assert partial["theorems_started"] == 1
    assert partial["theorems_finished"] == 0


def test_parse_search_log_records_shared_budget_timeout() -> None:
    log = """\
[2026-09-01 12:00:00:000] [INFO] Iteration 2: add axiom ax-mp
[2026-09-01 12:00:01:000] [INFO] Iteration 182: ecan-guided search proof of hard up to size 13
[2026-09-01 12:00:01:100] [INFO]   [union] ecan=40 will=0 union=40
[2026-09-01 12:00:01:200] [INFO]   [ranking] index=182 theorem=hard mode=ecan budget=40 ecan=(a b) william=() selected=(a b)
[2026-09-01 12:01:01:250] [INFO]   [tier] focused=2 full=135 af-hit=false
[2026-09-01 12:01:01:251] [INFO]   [timing] focus-seconds=0.25 fallback-seconds=59.75 timeout=fallback
[2026-09-01 12:01:01:252] [INFO] Search timed out after 60s in fallback tier
[2026-09-01 12:01:01:253] [INFO] No proof found
[2026-09-01 12:01:01:500] [INFO] [bg] daemon sweeps=200 final-funds=1.0
"""

    [run] = parse_search_log(log)

    assert run["complete"] is True
    assert run["theorems_finished"] == 1
    assert run["proofs_found"] == 0
    assert run["timeouts"] == 1
    assert run["focus_seconds"]["mean"] == pytest.approx(0.25)
    assert run["fallback_seconds_total"] == pytest.approx(59.75)
    assert run["theorems"][0]["result"] == "timeout"
    assert run["theorems"][0]["timeout_tier"] == "fallback"


def test_focused_timeout_is_not_counted_as_fallback() -> None:
    log = """\
[2026-09-01 12:00:00:000] [INFO] Iteration 2: add axiom ax-mp
[2026-09-01 12:00:01:000] [INFO] Iteration 6: union-guided search proof of hard up to size 13
[2026-09-01 12:00:01:100] [INFO]   [union] ecan=1 will=1 union=1
[2026-09-01 12:00:01:200] [INFO]   [ranking] index=6 theorem=hard mode=union budget=25 ecan=(a) william=(a) selected=(a)
[2026-09-01 12:00:01:300] [INFO]   [tier] focused=1 full=1 af-hit=false
[2026-09-01 12:00:01:301] [INFO]   [timing] focus-seconds=0.1 fallback-seconds=0.0 timeout=focused
[2026-09-01 12:00:01:302] [INFO] No proof found
[2026-09-01 12:00:01:500] [INFO] [bg] daemon sweeps=2 final-funds=1.0
"""

    [run] = parse_search_log(log)

    assert run["timeouts"] == 1
    assert run["fallbacks"] == 0