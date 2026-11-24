"""Local continuous trace memory and manually promoted chaining models."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from collections import Counter
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from uuid import uuid4

import numpy as np

from infcontrol.context_features import OnlineContextTracker
from infcontrol.model import PredictionModel
from infcontrol.operators import Formula

LOG = logging.getLogger(__name__)
MAX_BODY = 2 * 1024 * 1024


class Conflict(ValueError):
    """A valid request conflicts with already committed state."""


def identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or not all(char.isascii() and (char.isalnum() or char in "-_.") for char in value)
    ):
        raise ValueError(f"{name} must be 1-128 ASCII letters, numbers, '.', '_' or '-'.")
    return value


def formula(value: object, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("Formula nesting exceeds 64 levels.")
    if isinstance(value, str) and value and len(value) <= 1024:
        return
    if isinstance(value, list) and value:
        for child in value:
            formula(child, depth + 1)
        return
    raise ValueError("Formulas must be nonempty strings or nonempty nested lists of strings.")


def labels(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 2000:
        raise ValueError(f"{name} must be a list of at most 2000 labels.")
    if any(not isinstance(item, str) or not item or len(item) > 256 for item in value):
        raise ValueError(f"{name} contains an invalid label.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates.")
    return value


def encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


class ChainingService:
    def __init__(
        self,
        state_dir: Path,
        initial_model: Path,
        mork_url: str,
        *,
        min_events: int = 20,
        train_every: int = 10,
        window: int = 200,
        max_seconds: float = 5,
        max_popped: int = 3000,
        job_timeout: float = 60,
    ) -> None:
        if min_events < 6 or train_every < 1 or window < min_events:
            raise ValueError("Require min-events >= 6, train-every >= 1 and window >= min-events.")
        if (
            not all(math.isfinite(value) and value > 0 for value in (max_seconds, job_timeout))
            or job_timeout <= max_seconds
            or max_popped < 0
        ):
            raise ValueError("Require positive finite budgets, job-timeout > max-seconds and max-popped >= 0.")
        from urllib.parse import urlsplit

        parsed = urlsplit(mork_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("mork-url must be an HTTP(S) server URL without query or fragment.")
        self.root = state_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = (self.root / "service.lock").open("a")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lockfile.close()
            raise Conflict("Another service owns this state directory.") from None
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.models: dict[str, PredictionModel] = {}
        self.mork_url = mork_url
        self.min_events, self.train_every, self.window = min_events, train_every, window
        self.max_seconds, self.max_popped, self.job_timeout = max_seconds, max_popped, job_timeout
        self.db = sqlite3.connect(self.root / "state.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS models(
                id TEXT PRIMARY KEY, path TEXT NOT NULL, job_id TEXT);
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id));
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
                event_id TEXT NOT NULL, request TEXT NOT NULL, context TEXT NOT NULL, response TEXT NOT NULL,
                goal_key TEXT NOT NULL, outcome TEXT, used_labels TEXT,
                trainable INTEGER NOT NULL DEFAULT 0, UNIQUE(session_id,event_id));
            CREATE INDEX IF NOT EXISTS events_session ON events(session_id,seq);
            CREATE INDEX IF NOT EXISTS events_training ON events(trainable,goal_key,seq);
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, status TEXT NOT NULL, base_model TEXT NOT NULL REFERENCES models(id),
                event_count INTEGER NOT NULL, created REAL NOT NULL, result TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS templates(record TEXT PRIMARY KEY, count INTEGER NOT NULL CHECK(count > 0));
        """)
        with self.db:
            if self.setting("active_model") is None:
                # Import only known model files; caller-supplied initial model is trusted local input.
                PredictionModel.load(initial_model)
                version = "initial-" + uuid4().hex
                directory = self.root / "models" / version
                directory.mkdir(parents=True)
                metadata = json.loads((initial_model / "model.json").read_text())
                names = ["model.json", *[feature["graph"] for feature in metadata["features"]]]
                if (initial_model / "template_memory.json").exists():
                    names.append("template_memory.json")
                for name in names:
                    if Path(name).name != name:
                        raise ValueError("Initial model files must be local filenames.")
                    shutil.copyfile(initial_model / name, directory / name)
                self.db.execute("INSERT INTO models VALUES (?,?,NULL)", (version, str(directory)))
                self.set_setting("active_model", version)
                self.set_setting("namespace", "chaining-service-" + uuid4().hex)
            self.db.execute(
                "UPDATE jobs SET status='failed', error='Service interrupted during training; request a new job.' "
                "WHERE status='running'"
            )
        self.load_model(self.setting("active_model"))

    def setting(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, value))

    def load_model(self, version: str) -> PredictionModel:
        if version not in self.models:
            row = self.db.execute("SELECT path FROM models WHERE id=?", (version,)).fetchone()
            if row is None:
                raise ValueError("Unknown model version.")
            self.models[version] = PredictionModel.load(row["path"])
        return self.models[version]

    def open_session(self, request: dict) -> dict:
        session = identifier(request["session_id"], "session_id")
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions VALUES (?,?)", (session, self.setting("active_model")))
            row = self.db.execute("SELECT model_id FROM sessions WHERE id=?", (session,)).fetchone()
            return {"session_id": session, "model_id": row["model_id"]}

    def session_model(self, session: str) -> str:
        row = self.db.execute("SELECT model_id FROM sessions WHERE id=?", (session,)).fetchone()
        if row is None:
            raise ValueError("Open a session before scoring.")
        return row["model_id"]

    def tracker(self, session: str) -> OnlineContextTracker:
        tracker = OnlineContextTracker()
        for row in self.db.execute(
            "SELECT request, used_labels FROM events WHERE session_id=? AND outcome IS NOT NULL ORDER BY seq",
            (session,),
        ):
            request = json.loads(row["request"])
            tracker.observe(request["theory_labels"], json.loads(row["used_labels"]))
        return tracker

    def score(self, request: dict) -> dict:
        session = identifier(request["session_id"], "session_id")
        event_id = identifier(request["event_id"], "event_id")
        formula(request["goal"])
        candidate_labels = labels(request["candidate_labels"], "candidate_labels")
        theory = labels(request["theory_labels"], "theory_labels")
        candidates = request["candidates"]
        if not isinstance(candidates, list) or not candidates or len(candidates) != len(candidate_labels):
            raise ValueError("candidates and candidate_labels must have the same nonzero length.")
        if not set(candidate_labels) <= set(theory):
            raise ValueError("Candidate labels must be exposed theory labels.")
        for candidate in candidates:
            formula(candidate)
        payload = {key: request[key] for key in ("goal", "candidates", "candidate_labels", "theory_labels")}
        serialized = encode(payload)
        with self.lock, self.db:
            version = self.session_model(session)
            old = self.db.execute(
                "SELECT request,response FROM events WHERE session_id=? AND event_id=?", (session, event_id)
            ).fetchone()
            if old:
                if old["request"] != serialized:
                    raise Conflict("event_id already has a different scoring request.")
                return json.loads(old["response"])
            if self.db.execute("SELECT 1 FROM events WHERE session_id=? AND outcome IS NULL", (session,)).fetchone():
                raise Conflict("Observe the pending event before scoring another theorem.")
            context = self.tracker(session).features(candidate_labels)
            model = self.load_model(version)
            probabilities = model.predict([payload["goal"]] * len(candidates), candidates, context)
            if probabilities.shape != (len(candidates),) or not np.all(np.isfinite(probabilities)):
                raise RuntimeError("Model returned invalid probabilities.")
            response = {"probabilities": probabilities.tolist(), "model_id": version, "event_id": event_id}
            goal_key = sha256(encode(Formula(payload["goal"]).canonical_tree).encode()).hexdigest()
            self.db.execute(
                "INSERT INTO events(session_id,event_id,request,context,response,goal_key) VALUES (?,?,?,?,?,?)",
                (session, event_id, serialized, encode(context), encode(response), goal_key),
            )
            return response

    def observe(self, request: dict) -> dict:
        session = identifier(request["session_id"], "session_id")
        event_id = identifier(request["event_id"], "event_id")
        outcome = request["outcome"]
        if outcome not in ("proof", "timeout", "not_found"):
            raise ValueError("outcome must be proof, timeout or not_found.")
        used = sorted(labels(request["used_labels"], "used_labels"))
        if outcome != "proof" and used:
            raise ValueError("An unsuccessful search must not supply proof usage.")
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT * FROM events WHERE session_id=? AND event_id=?", (session, event_id)
            ).fetchone()
            if row is None:
                raise ValueError("No matching scored event.")
            payload = json.loads(row["request"])
            if not set(used) <= set(payload["theory_labels"]):
                raise ValueError("Proof labels must belong to the exposed theory.")
            if row["outcome"] is not None:
                if row["outcome"] != outcome or json.loads(row["used_labels"]) != used:
                    raise Conflict("Event already has a different committed outcome.")
            else:
                self.db.execute(
                    "UPDATE events SET outcome=?,used_labels=?,trainable=? WHERE seq=?",
                    (
                        outcome,
                        encode(used),
                        int(
                            outcome == "proof"
                            and 0 < len(set(used) & set(payload["candidate_labels"])) < len(payload["candidates"])
                        ),
                        row["seq"],
                    ),
                )
        self.wake.set()
        return {"ok": True, "event_id": event_id}

    def training_events(self) -> list[dict]:
        # Repeated benchmark sessions must not put the same canonical goal on both sides of the gate.
        rows = self.db.execute(
            "SELECT * FROM events WHERE seq IN "
            "(SELECT MIN(seq) FROM events WHERE trainable=1 GROUP BY goal_key) ORDER BY seq DESC LIMIT ?",
            (self.window,),
        ).fetchall()
        return [
            {
                "seq": row["seq"],
                "request": json.loads(row["request"]),
                "used_labels": json.loads(row["used_labels"]),
                "context": json.loads(row["context"]),
            }
            for row in reversed(rows)
        ]

    def training_count(self) -> int:
        return self.db.execute("SELECT COUNT(DISTINCT goal_key) FROM events WHERE trainable=1").fetchone()[0]

    def enqueue(self, *, automatic: bool = False) -> dict | None:
        with self.lock, self.db:
            pending = self.db.execute("SELECT id FROM jobs WHERE status IN ('queued','running')").fetchone()
            if pending:
                if automatic:
                    return None
                raise Conflict(f"Job {pending['id']} is already queued or running.")
            count = self.training_count()
            if count < self.min_events:
                if automatic:
                    return None
                raise Conflict(
                    f"Need {self.min_events} distinct successful goals with positive and negative candidates."
                )
            previous = self.db.execute("SELECT MAX(event_count) FROM jobs").fetchone()[0] or 0
            if automatic and count - previous < self.train_every:
                return None
            events = self.training_events()
            job_id = uuid4().hex
            directory = self.root / "jobs" / job_id
            directory.mkdir(parents=True)
            base = self.setting("active_model")
            model_path = self.db.execute("SELECT path FROM models WHERE id=?", (base,)).fetchone()["path"]
            spec = {
                "events": events[-self.window :],
                "base_model_dir": model_path,
                "mork_url": self.mork_url,
                "namespace": self.setting("namespace"),
                "memory": dict(self.db.execute("SELECT record,count FROM templates").fetchall()),
                "max_seconds": self.max_seconds,
                "max_popped": self.max_popped,
            }
            (directory / "input.json").write_text(encode(spec))
            self.db.execute(
                "INSERT INTO jobs(id,status,base_model,event_count,created) VALUES (?,'queued',?,?,?)",
                (job_id, base, count, time()),
            )
        self.wake.set()
        return {"job_id": job_id, "status": "queued"}

    def run_next_job(self) -> bool:
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return False
            job_id = row["id"]
            self.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
        directory = self.root / "jobs" / job_id
        try:
            with (directory / "worker.log").open("w") as log:
                with self.lock:
                    if self.stop.is_set():
                        raise RuntimeError("Service is shutting down.")
                    self.process = subprocess.Popen(
                        [sys.executable, "-m", "infcontrol.service_training", str(directory)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    process = self.process
                try:
                    code = process.wait(timeout=self.job_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(f"Training exceeded hard job timeout ({self.job_timeout}s).") from None
                if code:
                    raise RuntimeError(f"Training exited with code {code}; see {directory / 'worker.log'}.")
            result = json.loads((directory / "result.json").read_text())
            PredictionModel.load(directory / "model")
            with self.lock, self.db:
                self.db.execute(
                    "UPDATE jobs SET status=?,result=? WHERE id=?",
                    ("ready" if result["passed"] else "rejected", encode(result), job_id),
                )
        except Exception as exc:
            LOG.exception("Training job %s failed", job_id)
            with self.lock, self.db:
                self.db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (str(exc), job_id))
        finally:
            with self.lock:
                self.process = None
        return True

    def promote(self, request: dict) -> dict:
        from infcontrol.service_training import restore_memory

        job_id = identifier(request["job_id"], "job_id")
        with self.lock, self.db:
            job = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["status"] not in ("ready", "promoted"):
                raise Conflict("Only a job with a passed gate can be promoted.")
            if job["status"] == "promoted":
                return {"model_id": job_id, "active": self.setting("active_model") == job_id}
            if job["base_model"] != self.setting("active_model"):
                raise Conflict("Gate baseline is stale; train a new candidate against the active model.")
            if self.db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running')").fetchone():
                raise Conflict("Wait for the queued/running training job before promotion.")
            directory = self.root / "jobs" / job_id / "model"
            model = PredictionModel.load(directory)
            result = json.loads(job["result"])
            snapshot = Counter(dict(self.db.execute("SELECT record,count FROM templates").fetchall()))
            snapshot.update(result["template_records"])
            # A failed remote write must not switch the active predictor.
            restore_memory(self.mork_url, self.setting("namespace"), dict(snapshot))
            self.db.execute("INSERT INTO models VALUES (?,?,?)", (job_id, str(directory), job_id))
            for record in result["template_records"]:
                self.db.execute(
                    "INSERT INTO templates VALUES (?,1) ON CONFLICT(record) DO UPDATE SET count=count+1", (record,)
                )
            self.set_setting("active_model", job_id)
            self.db.execute("UPDATE jobs SET status='promoted' WHERE id=?", (job_id,))
            self.models[job_id] = model
            return {"model_id": job_id, "active": True}

    def status(self) -> dict:
        with self.lock:
            jobs = []
            for row in self.db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 20"):
                job = dict(row)
                job["result"] = json.loads(job["result"]) if job["result"] else None
                jobs.append(job)
            return {
                "active_model": self.setting("active_model"),
                "activation": "manual",
                "sessions": self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "completed_events": self.db.execute("SELECT COUNT(*) FROM events WHERE outcome IS NOT NULL").fetchone()[
                    0
                ],
                "pending_events": self.db.execute("SELECT COUNT(*) FROM events WHERE outcome IS NULL").fetchone()[0],
                "distinct_training_goals": self.training_count(),
                "templates": self.db.execute("SELECT COUNT(*) FROM templates").fetchone()[0],
                "mork_namespace": self.setting("namespace"),
                "worker_alive": self.worker is not None and self.worker.is_alive(),
                "jobs": jobs,
            }

    def start_worker(self) -> None:
        def loop() -> None:
            while not self.stop.is_set():
                self.wake.wait(1)
                self.wake.clear()
                try:
                    self.enqueue(automatic=True)
                    if not self.stop.is_set():
                        self.run_next_job()
                except Exception:
                    LOG.exception("Continuous training scheduler failed")

        self.worker = threading.Thread(target=loop, name="chaining-training", daemon=True)
        self.worker.start()

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        if self.worker is not None:
            self.worker.join(timeout=self.job_timeout + 5)
            if self.worker.is_alive():
                raise RuntimeError("Training worker did not stop.")
        self.db.close()
        fcntl.flock(self.lockfile, fcntl.LOCK_UN)
        self.lockfile.close()


def make_server(service: ChainingService, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/health", "/status"):
                self.reply(404, {"error": "Unknown endpoint."})
                return
            self.reply(200, service.status())

        def do_POST(self) -> None:
            routes = {
                "/sessions": service.open_session,
                "/score": service.score,
                "/observe": service.observe,
                "/train": lambda request: service.enqueue(),
                "/promote": service.promote,
            }
            if self.path not in routes:
                self.reply(404, {"error": "Unknown endpoint."})
                return
            try:
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    self.reply(415, {"error": "Content-Type must be application/json."})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"Content-Length must be 1-{MAX_BODY} bytes.")
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Chunked requests are not supported.")
                self.connection.settimeout(10)
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object.")
                response = routes[self.path](request)
            except Conflict as exc:
                self.reply(409, {"error": str(exc)})
            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception:
                LOG.exception("Service request %s failed", self.path)
                self.reply(500, {"error": "Internal service error; inspect service logs."})
            else:
                self.reply(200, response)

        def reply(self, status: int, response: object) -> None:
            body = encode(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # Intentionally loopback-only: no unauthenticated public training/promotion API.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--mork-url", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--train-every", type=int, default=10)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=5)
    parser.add_argument("--max-popped", type=int, default=3000)
    parser.add_argument("--job-timeout", type=float, default=60)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    options = vars(args).copy()
    port = options.pop("port")
    service = ChainingService(**options)
    server = None

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        server = make_server(service, port)
        service.start_worker()
        LOG.info("Chaining service listening on http://127.0.0.1:%s", server.server_port)
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Stopping chaining service")
    finally:
        if server is not None:
            server.server_close()
        service.close()


if __name__ == "__main__":
    main()
