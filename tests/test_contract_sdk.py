"""SDK hub endpoints — docs content is seeded; operational data is never faked."""

from __future__ import annotations


def test_snippets_is_object_keyed_by_language(client):
    snippets = client.get("/api/sdk/snippets").json()
    assert isinstance(snippets, dict)  # zod Record — NOT an array
    # Self-host guide: only the tabs that actually stand up a server.
    assert set(snippets) == {"python", "node", "cli"}
    for lang, snip in snippets.items():
        assert snip["language"] == lang
        assert set(snip) == {"language", "filename", "code"}
        assert len(snip["code"]) > 50


def test_snippets_reference_only_real_apis(client):
    """Docs honesty: every snippet targets APIs/commands that actually exist,
    and presents the packages as embeddable libraries (not hand-wired servers)."""
    snippets = client.get("/api/sdk/snippets").json()
    all_code = "\n".join(s["code"] for s in snippets.values())
    # The real packages / CLI / SDK class appear...
    assert "astra-ai-sdk" in all_code
    assert "astra serve" in all_code            # real CLI command (cli.py)
    assert "AstraRunner" in all_code            # real SDK serving class (runner.py)
    # ...the Node tab uses the real npm package as a library, not raw onnxruntime...
    assert 'from "astra-ai-sdk"' in all_code       # real npm import (clients/node)
    assert "fromDeployment" in all_code         # real Node SDK entrypoint
    # ...and the old fictional / hand-wired APIs never do.
    assert "pip install astra\n" not in all_code
    assert "brew install astra" not in all_code
    assert "client.pareto.run" not in all_code
    assert "npm install astra" not in all_code
    assert "ort.InferenceSession" not in all_code   # no hand-wired onnxruntime-node
    assert "new ort.Tensor" not in all_code
    assert "import express" not in all_code         # no server boilerplate
    # ...and copy-paste code never carries a base URL — it's baked into the SDK
    # (the hosted origin), so users only fill in the deployment id + API key.
    assert "BASE_URL" not in all_code
    assert "__ASTRA_ORIGIN__" not in all_code
    assert "base_url" not in all_code
    assert "baseUrl" not in all_code


def test_keys_and_webhooks_endpoints_removed(client):
    # A local tool has no auth/delivery infra — the husk endpoints are gone.
    assert client.get("/api/sdk/keys").status_code == 404
    assert client.get("/api/sdk/webhooks").status_code == 404


def test_recipes(client):
    recipes = client.get("/api/sdk/recipes").json()
    assert len(recipes) == 3
    for r in recipes:
        assert set(r) == {"id", "title", "description", "language", "steps"}
        assert isinstance(r["steps"], list)
        assert r["language"] in {"python", "node", "cli", "curl"}
