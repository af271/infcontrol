import csv
import json
import os
import subprocess
import threading
from collections import Counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from infcontrol.chaining_service import ChainingService, Conflict, make_server
from infcontrol.model import save_model
from infcontrol.service_training import restore_memory, train_job


@pytest.fixture
def service(tmp_path):
    initial = save_model(tmp_path / "initial", [], 0.0, [])
    instance = ChainingService(tmp_path / "state", initial, "http://127.0.0.1:8000", min_events=6)
    yield instance
    instance.close()


def score_request(index=0, session="run"):
    return {
        "session_id": session,
        "event_id": str(index),
        "goal": ["->", f"g{index}", "end"],
        "candidate_labels": ["a", "b"],
        "candidates": [["->", f"g{index}", "end"], "irrelevant"],
        "theory_labels": ["a", "b", "axiom"],
    }


def add_event(service, index, session="run", outcome="proof"):
    service.open_session({"session_id": session})
    service.score(score_request(index, session))
    return service.observe(
        {
            "session_id": session,
            "event_id": str(index),
            "outcome": outcome,
            "used_labels": ["a", "axiom"] if outcome == "proof" else [],
        }
    )


def ready_job(service, job_id, *, passed=True, base=None):
    directory = service.root / "jobs" / job_id
    save_model(directory / "model", [], 1.0, [])
    with service.db:
        service.db.execute(
            "INSERT INTO jobs(id,status,base_model,event_count,created,result) VALUES (?,?,?,?,?,?)",
            (
                job_id,
                "ready" if passed else "rejected",
                base or service.setting("active_model"),
                0,
                0,
                json.dumps({"passed": passed, "template_records": ["template"]}),
            ),
        )


def test_durable_idempotent_events_and_pending_context(tmp_path):
    initial_model = save_model(tmp_path / "initial", [], 0.0, [])
    service = ChainingService(tmp_path / "state", initial_model, "http://127.0.0.1:8000", min_events=6)
    pinned = service.open_session({"session_id": "run"})["model_id"]
    first = service.score(score_request())
    assert first["model_id"] == pinned
    assert service.score(score_request()) == first
    with pytest.raises(Conflict, match="different scoring"):
        service.score({**score_request(), "goal": "other"})
    with pytest.raises(Conflict, match="pending"):
        service.score(score_request(1))
    state = service.root
    # Reopen the database/service without discarding the pending theorem.
    service.close()
    resumed = ChainingService(state, state / "does-not-need-to-exist", "http://127.0.0.1:8000", min_events=6)
    try:
        assert resumed.score(score_request()) == first
        outcome = {"session_id": "run", "event_id": "0", "outcome": "timeout", "used_labels": []}
        assert resumed.observe(outcome) == resumed.observe(outcome)
        with pytest.raises(Conflict, match="different committed"):
            resumed.observe({**outcome, "outcome": "proof", "used_labels": ["a"]})
        resumed.score(score_request(1))
        context = json.loads(resumed.db.execute("SELECT context FROM events WHERE event_id='1'").fetchone()[0])
        assert context["candidate_exposure_count"] == [1, 1]
        assert context["candidate_used_count"] == [0, 0]
        assert resumed.status()["distinct_training_goals"] == 0
    finally:
        resumed.close()


def test_trace_validation_and_no_timeout_negatives(service):
    with pytest.raises(ValueError, match="Open a session"):
        service.score(score_request())
    service.open_session({"session_id": "run"})
    with pytest.raises(ValueError, match="duplicates"):
        service.score({**score_request(), "candidate_labels": ["a", "a"]})
    service.score(score_request())
    with pytest.raises(ValueError, match="unsuccessful"):
        service.observe({"session_id": "run", "event_id": "0", "outcome": "timeout", "used_labels": ["a"]})
    with pytest.raises(ValueError, match="exposed theory"):
        service.observe({"session_id": "run", "event_id": "0", "outcome": "proof", "used_labels": ["oracle"]})
    service.observe({"session_id": "run", "event_id": "0", "outcome": "not_found", "used_labels": []})
    for index in range(1, 7):
        add_event(service, index)
    add_event(service, 1, session="repeated-benchmark")
    assert len(service.training_events()) == 6
    events = service.training_events()
    assert events[0]["context"]["candidate_used_count"] == [0, 0]
    assert events[1]["context"]["candidate_used_count"] == [1, 0]
    job = service.enqueue()
    spec = json.loads((service.root / "jobs" / job["job_id"] / "input.json").read_text())
    assert len(spec["events"]) == 6
    assert all(event["seq"] != 1 for event in spec["events"])
    with pytest.raises(Conflict, match="already queued"):
        service.enqueue()


def test_manual_promotion_pins_sessions_and_rejects_stale_gate(service, monkeypatch):
    monkeypatch.setattr("infcontrol.service_training.restore_memory", lambda *args: None)
    initial = service.open_session({"session_id": "run"})["model_id"]
    ready_job(service, "new")
    ready_job(service, "stale")
    ready_job(service, "rejected", passed=False)
    with pytest.raises(Conflict, match="passed gate"):
        service.promote({"job_id": "rejected"})
    service.promote({"job_id": "new"})
    assert service.setting("active_model") == "new"
    assert service.open_session({"session_id": "run"})["model_id"] == initial
    assert service.open_session({"session_id": "later"})["model_id"] == "new"
    assert service.score(score_request())["probabilities"] == [0.5, 0.5]
    assert service.score(score_request(session="later"))["probabilities"][0] > 0.5
    assert service.status()["templates"] == 1
    assert service.promote({"job_id": "new"})["active"] is True
    assert service.db.execute("SELECT count FROM templates").fetchone()[0] == 1
    with pytest.raises(Conflict, match="stale"):
        service.promote({"job_id": "stale"})


def test_state_directory_has_single_owner(service):
    with pytest.raises(Conflict, match="owns"):
        ChainingService(service.root, service.root, "http://127.0.0.1:8000")


def test_failed_memory_sync_never_activates_candidate(service, monkeypatch):
    ready_job(service, "new")
    initial = service.setting("active_model")

    def offline(*args):
        raise ConnectionError("MORK unavailable")

    monkeypatch.setattr("infcontrol.service_training.restore_memory", offline)
    with pytest.raises(ConnectionError, match="MORK unavailable"):
        service.promote({"job_id": "new"})
    assert service.setting("active_model") == initial
    assert service.status()["templates"] == 0
    assert service.status()["jobs"][0]["status"] == "ready"


def test_restarting_marks_interrupted_jobs_failed_and_keeps_queued_jobs(tmp_path):
    initial = save_model(tmp_path / "initial", [], 0.0, [])
    root = tmp_path / "state"
    instance = ChainingService(root, initial, "http://127.0.0.1:8000", min_events=6)
    try:
        for index in range(6):
            add_event(instance, index)
        first = instance.enqueue()
        with instance.db:
            instance.db.execute("UPDATE jobs SET status='running' WHERE id=?", (first["job_id"],))
    finally:
        instance.close()
    resumed = ChainingService(root, initial, "http://127.0.0.1:8000", min_events=6)
    try:
        assert resumed.status()["jobs"][0]["status"] == "failed"
        assert "interrupted" in resumed.status()["jobs"][0]["error"]
        second = resumed.enqueue()
    finally:
        resumed.close()
    again = ChainingService(root, initial, "http://127.0.0.1:8000", min_events=6)
    try:
        assert again.status()["jobs"][0]["id"] == second["job_id"]
        assert again.status()["jobs"][0]["status"] == "queued"
    finally:
        again.close()


def test_http_api_returns_explicit_errors(service):
    server = make_server(service, 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health") as response:
            assert json.load(response)["activation"] == "manual"
        request = Request(
            base + "/sessions", data=b'{"session_id":"http"}', headers={"Content-Type": "application/json"}
        )
        with urlopen(request) as response:
            assert json.load(response)["session_id"] == "http"
        for path, body, expected in (
            ("/score", b"[]", 400),
            ("/train", b"{}", 409),
            ("/sessions", b"{bad", 400),
            ("/missing", b"{}", 404),
        ):
            with pytest.raises(HTTPError) as raised:
                urlopen(Request(base + path, data=body, headers={"Content-Type": "application/json"}))
            assert raised.value.code == expected
            assert "error" in json.load(raised.value)
        with pytest.raises(HTTPError) as raised:
            urlopen(Request(base + "/train", data=b"{}"))
        assert raised.value.code == 415
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_worker_failure_is_recorded_and_not_retried_forever(service, monkeypatch):
    for index in range(6):
        add_event(service, index)
    job = service.enqueue()

    class Process:
        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    assert service.run_next_job()
    status = service.status()["jobs"][0]
    assert status["id"] == job["job_id"]
    assert status["status"] == "failed"
    assert "code 1" in status["error"]
    assert service.enqueue(automatic=True) is None


def test_hard_job_timeout_kills_worker_and_records_error(service, monkeypatch):
    for index in range(6):
        add_event(service, index)
    service.enqueue()

    class Process:
        killed = False

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("trainer", timeout)
            return -9

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    assert service.run_next_job()
    assert process.killed
    assert service.status()["jobs"][0]["status"] == "failed"
    assert "hard job timeout" in service.status()["jobs"][0]["error"]


class Memory:
    stores = {}

    def __init__(self, _url, namespace, *, erase=False):
        self.namespace = namespace
        self.generation = "test-generation"
        if erase or namespace not in self.stores:
            self.stores[namespace] = Counter()
        self.store = self.stores[namespace]

    def remember(self, records):
        self.store.update(set(records))

    def load(self):
        return sorted(self.store.items())


def test_memory_rebuild_preserves_weights_and_other_namespaces(monkeypatch):
    monkeypatch.setattr("infcontrol.service_training.MorkTemplateMemory", Memory)
    other = Memory("", "other", erase=True)
    other.remember(["untouched"])
    restored = restore_memory("fake", "owned", {"a": 3, "b": 1})
    assert restored.load() == [("a", 3), ("b", 1)]
    Memory.stores["owned"].clear()
    assert restore_memory("fake", "owned", {"a": 3, "b": 1}).load() == [("a", 3), ("b", 1)]
    assert other.load() == [("untouched", 1)]


def test_real_training_keeps_outer_gate_out_of_fitting_and_memory(service, monkeypatch):
    monkeypatch.setattr("infcontrol.service_training.MorkTemplateMemory", Memory)
    monkeypatch.setattr("infcontrol.inference_control.MorkTemplateMemory", Memory)
    for index in range(10):
        add_event(service, index)
    job = service.enqueue()
    directory = service.root / "jobs" / job["job_id"]
    spec = json.loads((directory / "input.json").read_text())
    spec.update(max_seconds=3, max_popped=100)
    (directory / "input.json").write_text(json.dumps(spec))
    result = train_job(directory)
    assert len(result["training_event_seqs"]) == 8
    assert len(result["gate_event_seqs"]) == 2
    assert not set(result["training_event_seqs"]) & set(result["gate_event_seqs"])
    with (directory / "training.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert {int(row["theorem_index"]) for row in rows} == set(result["training_event_seqs"])
    assert Memory.stores[spec["namespace"]] == Counter()
    assert (directory / "model" / "template_memory.json").exists()
    assert result["candidate"]["log_loss"] >= 0


@pytest.mark.skipif(not os.environ.get("WILLIAM_MORK_TEST_URL"), reason="Requires real weighted MORK server")
def test_subprocess_training_promotion_and_memory_recovery_with_real_mork(tmp_path):
    from william.mork_memory import MorkTemplateMemory

    url = os.environ["WILLIAM_MORK_TEST_URL"]
    initial = save_model(tmp_path / "initial", [], 0.0, [])
    root = tmp_path / "state"
    service = ChainingService(root, initial, url, min_events=6, max_seconds=8, max_popped=1000)

    def collect(instance, start, end):
        instance.open_session({"session_id": "synthetic"})
        for index in range(start, end):
            request = score_request(index, "synthetic")
            request.update(
                candidate_labels=[f"b{index}", f"a{index}"],
                candidates=["irrelevant", request["goal"]],
                theory_labels=[f"b{index}", f"a{index}"],
            )
            instance.score(request)
            instance.observe(
                {
                    "session_id": "synthetic",
                    "event_id": str(index),
                    "outcome": "proof",
                    "used_labels": [f"a{index}"],
                }
            )

    try:
        collect(service, 0, 12)
        first = service.enqueue()
        assert service.run_next_job()
        job = service.status()["jobs"][0]
        assert job["status"] == "ready", job
        assert job["result"]["template_records"]
        old_model = service.open_session({"session_id": "synthetic"})["model_id"]
        service.promote({"job_id": first["job_id"]})
        assert service.open_session({"session_id": "synthetic"})["model_id"] == old_model
        assert service.open_session({"session_id": "new-client"})["model_id"] == first["job_id"]
        snapshot = dict(service.db.execute("SELECT record,count FROM templates").fetchall())
        namespace = service.setting("namespace")
        assert dict(MorkTemplateMemory(url, namespace).load()) == snapshot
    finally:
        service.close()
    # Emulate lost remote state without affecting any other MORK namespace.
    assert MorkTemplateMemory(url, namespace, erase=True).load() == []
    resumed = ChainingService(root, initial, url, min_events=6, max_seconds=8, max_popped=1000)
    try:
        assert resumed.setting("active_model") == first["job_id"]
        collect(resumed, 12, 18)
        resumed.enqueue()
        assert resumed.run_next_job()
        latest = resumed.status()["jobs"][0]
        assert latest["status"] in ("ready", "rejected"), latest
        assert latest["result"]["loaded_templates"] == len(snapshot)
        assert dict(MorkTemplateMemory(url, namespace).load()) == snapshot
    finally:
        resumed.close()
