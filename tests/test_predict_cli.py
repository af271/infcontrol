import io
import json

import numpy as np

from infcontrol.predict_cli import serve


class ContextProbeModel:
    context_feature_names = ("candidate_age",)

    def predict(self, goals, candidates, context_features=None):
        assert context_features is not None
        return np.asarray(context_features["candidate_age"], dtype=float)


def test_service_tracks_context_between_batched_requests() -> None:
    requests = [
        {
            "goals": [["g"], ["g"]],
            "candidates": [["a"], ["b"]],
            "candidate_labels": ["a", "b"],
        },
        {
            "command": "observe",
            "theory_labels": ["a", "b"],
            "used_labels": ["a"],
        },
        {
            "goals": [["h"], ["h"]],
            "candidates": [["a"], ["c"]],
            "candidate_labels": ["a", "c"],
        },
    ]
    input_stream = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    output_stream = io.StringIO()

    serve(ContextProbeModel(), input_stream, output_stream)

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses == [
        {"probabilities": [0.0, 0.0]},
        {"ok": True},
        {"probabilities": [1.0, 0.0]},
    ]