"""Deployment lifecycle — create / list / pause / resume / rotate / delete."""

from __future__ import annotations


def test_deploy_mints_endpoint_and_key(make_live_model, deploy_model, client):
    model = make_live_model("deploy-a.onnx")
    mid = model["modelId"]

    r = client.post(f"/api/models/{mid}/deployments", json={"region": "ap-northeast-2"})
    assert r.status_code == 200, r.text
    data = r.json()
    dep, key = data["deployment"], data["apiKey"]
    assert dep["id"].startswith("dep_")
    assert dep["endpoint"].endswith(f"/api/v1/artifacts/{dep['id']}")
    assert dep["status"] == "live"
    # Plaintext key shown once; only a masked prefix is persisted/listed.
    assert key.startswith("astra_sk_live_")
    assert "…" in dep["keyPrefix"] and key not in dep["keyPrefix"]

    # The model flips to deployed — both the boolean flag AND the lifecycle
    # status (the AI Models list renders its badge from `status`, so the green
    # "Deployed" badge only appears when status itself flips).
    m = client.get(f"/api/models/{mid}").json()
    assert m["isDeployed"] is True
    assert m["status"] == "deployed"

    # It shows up in the management list and the telemetry deployments widget.
    listed = client.get(f"/api/models/{mid}/deployments").json()
    assert any(d["id"] == dep["id"] for d in listed)
    tele = client.get(f"/api/models/{mid}/telemetry/deployments").json()
    assert any(d["endpoint"] == dep["endpoint"] for d in tele)


def test_pause_resume_and_rotate(make_live_model, deploy_model, client):
    mid = make_live_model("deploy-b.onnx")["modelId"]
    dep_id, key = deploy_model(mid)

    paused = client.post(f"/api/deployments/{dep_id}/pause").json()
    assert paused["status"] == "paused"
    resumed = client.post(f"/api/deployments/{dep_id}/resume").json()
    assert resumed["status"] == "live"

    rot = client.post(f"/api/deployments/{dep_id}/rotate-key").json()
    assert rot["apiKey"].startswith("astra_sk_live_")
    assert rot["apiKey"] != key  # a genuinely new secret


def test_delete_clears_deployed_flag(make_live_model, deploy_model, client):
    mid = make_live_model("deploy-c.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)

    assert client.delete(f"/api/deployments/{dep_id}").status_code == 200
    assert client.get(f"/api/models/{mid}/deployments").json() == []
    m = client.get(f"/api/models/{mid}").json()
    assert m["isDeployed"] is False
    # Undeploying the last deployment returns the model to its ready-to-deploy
    # "draft" status (the compressed artifact still exists) — not stuck "deployed".
    assert m["status"] == "draft"


def test_pause_keeps_model_deployed(make_live_model, deploy_model, client):
    """Pausing a deployment doesn't undeploy the model — it still HAS a
    deployment, so the AI Models badge stays "deployed". Only deleting the last
    deployment reverts it."""
    mid = make_live_model("deploy-d.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    assert client.get(f"/api/models/{mid}").json()["status"] == "deployed"

    assert client.post(f"/api/deployments/{dep_id}/pause").json()["status"] == "paused"
    m = client.get(f"/api/models/{mid}").json()
    assert m["status"] == "deployed"
    assert m["isDeployed"] is True


def test_update_deployment_metadata(make_live_model, deploy_model, client):
    """PATCH sets a custom name + description and they persist on the list."""
    mid = make_live_model("deploy-e.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)

    # Defaults: name comes from the model, description is empty.
    before = client.get(f"/api/models/{mid}/deployments").json()[0]
    assert before["description"] == ""

    r = client.patch(
        f"/api/deployments/{dep_id}",
        json={"name": "EU GPU box", "description": "Seoul region prod mirror"},
    )
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["name"] == "EU GPU box"
    assert updated["description"] == "Seoul region prod mirror"

    listed = client.get(f"/api/models/{mid}/deployments").json()[0]
    assert listed["name"] == "EU GPU box"
    assert listed["description"] == "Seoul region prod mirror"


def test_update_partial_keeps_other_fields(make_live_model, deploy_model, client):
    """An omitted field is left untouched; description can be cleared to ''."""
    mid = make_live_model("deploy-f.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    client.patch(f"/api/deployments/{dep_id}", json={"name": "alpha", "description": "note"})

    # Update only the description — name stays.
    r = client.patch(f"/api/deployments/{dep_id}", json={"description": "new note"})
    assert r.json()["name"] == "alpha"
    assert r.json()["description"] == "new note"

    # Clearing the description is allowed.
    r = client.patch(f"/api/deployments/{dep_id}", json={"description": ""})
    assert r.json()["name"] == "alpha"
    assert r.json()["description"] == ""


def test_update_empty_name_ignored(make_live_model, deploy_model, client):
    """A blank name is ignored so the deployment never loses its label."""
    mid = make_live_model("deploy-g.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    original = client.get(f"/api/models/{mid}/deployments").json()[0]["name"]

    r = client.patch(f"/api/deployments/{dep_id}", json={"name": "   "})
    assert r.status_code == 200
    assert r.json()["name"] == original


def test_deploy_unknown_model_404(client):
    assert client.post("/api/models/m_missing/deployments", json={}).status_code == 404


def test_manage_unknown_deployment_404(client):
    assert client.post("/api/deployments/dep_missing/pause").status_code == 404
    assert client.patch("/api/deployments/dep_missing", json={"name": "x"}).status_code == 404
    assert client.delete("/api/deployments/dep_missing").status_code == 404
