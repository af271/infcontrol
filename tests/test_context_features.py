from pathlib import Path

import pytest

from infcontrol.context_features import (
    CONTEXT_COLUMNS,
    OnlineContextTracker,
    iter_context_rows,
    read_trace,
)

TRACE = """\
(TraceFullSmallest t0 (a b) (-> x y) (a))
(TraceFullSmallest t1 (a b t0) (-> y z) (t0))
(TraceFullSmallest t2 (a b t0 t1) (-> z x) (t0 t1))
"""


def test_context_features_are_online(tmp_path: Path) -> None:
    source = tmp_path / "trace.metta"
    source.write_text(TRACE, encoding="utf-8")

    records = read_trace(source)
    assert [record.theorem_index for record in records] == [0, 1, 2]

    rows = list(iter_context_rows(source))
    t1_t0 = next(
        row
        for row in rows
        if row["theorem_index"] == 1.0 and row["candidate_age"] == 0.0
    )
    assert t1_t0["candidate_used_count"] == 0.0
    assert t1_t0["candidate_exposure_count"] == 0.0
    assert t1_t0["recent_used_count_1"] == 0.0
    assert t1_t0["ema_used_h5"] == 0.0
    assert t1_t0["candidate_label"] == "t0"
    assert t1_t0["candidate_index"] == 2
    assert set(CONTEXT_COLUMNS).issubset(rows[0])


def test_context_features_do_not_use_current_label(tmp_path: Path) -> None:
    source = tmp_path / "trace.metta"
    source.write_text(TRACE, encoding="utf-8")

    rows = list(iter_context_rows(source))
    t2_t0 = next(
        row
        for row in rows
        if row["theorem_index"] == 2.0 and row["candidate_age"] == 1.0
    )
    assert t2_t0["candidate_used_count"] == 1.0
    assert t2_t0["recent_used_count_1"] == 1.0


def test_online_tracker_reproduces_early_training_context() -> None:
    tracker = OnlineContextTracker()

    initial = tracker.features(["ax-mp"])
    assert initial["candidate_prior_logodds_alpha5"][0] == pytest.approx(0.0)

    tracker.observe(["ax-mp", "ax-1", "ax-2", "ax-3"], ["ax-mp"])
    first = tracker.features(["mp2"])
    assert first["candidate_age"] == [0.0]
    assert first["candidate_prior_alpha1"] == pytest.approx([0.25])
    assert first["candidate_prior_logodds_alpha5"] == pytest.approx(
        [-1.0986122886681098]
    )

    tracker.observe(["ax-mp", "ax-1", "ax-2", "ax-3", "mp2"], ["ax-mp"])
    second = tracker.features(["mp2", "mp2b"])
    assert second["candidate_age"] == [1.0, 0.0]
    assert second["candidate_exposure_count"] == [1.0, 0.0]
    assert second["candidate_prior_alpha1"] == pytest.approx(
        [0.1111111111111111, 0.2222222222222222]
    )
