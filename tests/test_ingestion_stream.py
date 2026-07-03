"""SSE stream vs polling fallback — identical IngestionLog payloads."""

from __future__ import annotations

import json
import time


def _wait_done(client, model_id: str, run_id: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/api/models/{model_id}/ingestion/{run_id}").json()
        if s["status"] != "streaming":
            return
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def test_sse_matches_polling(client):
    body = client.post("/api/models/import", json={"fileName": "stream-check.onnx"}).json()
    model_id, run_id = body["modelId"], body["runId"]
    _wait_done(client, model_id, run_id)

    polled = client.get(f"/api/models/{model_id}/ingestion/{run_id}/logs").json()
    assert polled["done"] is True
    poll_msgs = [(entry["level"], entry["message"]) for entry in polled["logs"]]
    assert poll_msgs, "polling returned no logs"

    sse_msgs: list[tuple[str, str]] = []
    progress_events: list[int] = []
    got_done = False
    with client.stream("GET", f"/api/models/{model_id}/ingestion/{run_id}/stream") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "log":
                    sse_msgs.append((data["level"], data["message"]))
                elif event == "progress":
                    progress_events.append(data["progress"])
                elif event == "done":
                    got_done = True
                    break

    assert got_done
    assert sse_msgs == poll_msgs  # full replay, same order, same content
    # A finished run's stream reports its percent on connect, before `done`,
    # so a revisiting client can render the bar without polling.
    assert progress_events == [polled["progress"]] == [100]


def test_polling_after_cursor(client):
    body = client.post("/api/models/import", json={"fileName": "cursor-check.pkl"}).json()
    model_id, run_id = body["modelId"], body["runId"]
    _wait_done(client, model_id, run_id)

    first = client.get(f"/api/models/{model_id}/ingestion/{run_id}/logs").json()
    n = len(first["logs"])
    assert n > 5
    cursor = first["logs"][2]["seq"]
    rest = client.get(
        f"/api/models/{model_id}/ingestion/{run_id}/logs", params={"after": cursor},
    ).json()
    assert len(rest["logs"]) == n - 3
    assert rest["logs"][0]["seq"] == cursor + 1


def test_logs_404_for_wrong_model(client):
    assert client.get("/api/models/m_x/ingestion/ing_zzz/logs").status_code == 404
