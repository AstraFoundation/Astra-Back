"""First-boot seed — documentation content ONLY.

No demo models, runs, deployments, alerts, activity, API keys or webhooks are
ever fabricated: every operational row in the DB comes from real uploads and
real pipeline events. The only seeded rows are SDK documentation (quick-start
snippets and recipes), which are static docs content, not operational state.

Every snippet below targets APIs that actually exist: the astra-ai-sdk packages
(Python `pip install astra-ai-sdk`, Node `npm i astra-ai-sdk`) — AstraRunner /
`astra` CLI for ON-DEVICE serving — and the real REST endpoints the SDK uses
(/api/v1/artifacts to pull the compressed model, /api/v1/telemetry for the
closed loop). Astra never runs the model server-side. A version marker row lets
upgrades replace stale docs rows in already-seeded databases.
"""

from __future__ import annotations

import json

from sqlmodel import Session, select

from app.dbmodels import RecipeRow, SdkSnippetRow

# Bump when snippets/recipes change — seeded DBs are upgraded in place.
# 12: install commands simplified to `pip install astra-ai-sdk` / `npm i astra-ai-sdk`
#     (onnxruntime is a core dep now — no `[serve]` extra, no explicit onnxruntime-node).
_DOCS_VERSION = "12"

# "Run the compressed model in YOUR code" guide. Each snippet uses the astra-ai-sdk
# package as an embeddable library — `runner.run(inputs)` inside the user's own
# app, NOT a server they must stand up. The base URL is baked into the SDK (the
# hosted origin), so copy-paste code needs only the deployment id + API key —
# never a BASE_URL.
_SNIPPETS: dict[str, dict] = {
    "python": {
        "language": "python", "filename": "infer.py",
        "code": '''# Run the compressed model inside your own code — no server, no base URL.
# pip install astra-ai-sdk
from astra_sdk import AstraRunner

# Deployment id + key from the Deployments tab (key shown once when minted).
# Pulls + caches the compressed artifact, runs onnxruntime locally, and ships
# latency / system / drift telemetry back to this dashboard — buffered on disk
# when offline and flushed the moment you reconnect (closed loop).
runner = AstraRunner.from_deployment("dep_...", "astra_sk_live_...")

out = runner.run({"input": my_array})     # local inference, right here in your code
print(out["latencyMs"], out["outputs"])
runner.close()                            # flush telemetry on shutdown

# Prefer a zero-code HTTP wrapper?  astra serve --port 8765   (see the CLI tab)''',
    },
    "node": {
        "language": "node", "filename": "infer.ts",
        "code": '''// Run the compressed model inside your own code — no server, no base URL.
// npm i astra-ai-sdk
import { AstraRunner } from "astra-ai-sdk";

// Deployment id + key from the Deployments tab (key shown once when minted).
const runner = await AstraRunner.fromDeployment({
  deploymentId: "dep_...",
  apiKey: "astra_sk_live_...",
});

// inputs: name -> { data, dims, type? } (type defaults to "float32").
const out = await runner.run({ input: { data: myFloats, dims: [1, 3, 224, 224] } });
console.log(out.latencyMs, out.outputs);  // local inference; telemetry auto-ships
await runner.close();''',
    },
    "cli": {
        "language": "cli", "filename": "serve.sh",
        "code": '''# Optional zero-code path: wrap the compressed model in a local POST /infer.
pip install astra-ai-sdk      # or: npm i -g astra-ai-sdk

export ASTRA_DEPLOYMENT_ID=dep_...         # Deployments tab -> deployment id
export ASTRA_API_KEY=astra_sk_live_...     # shown once when the key is minted

astra pull                          # cache the compressed artifact
astra serve --port 8765             # POST /infer now live at http://127.0.0.1:8765
# The library (Python/Node tabs) is usually enough — use this only for an HTTP wrapper.''',
    },
}

_RECIPES: list[dict] = [
    {
        "id": "r1", "language": "python",
        "title": "Deploy + mint an API key",
        "description": "Import a model, wait for optimization, create a "
                       "deployment on the Deployments tab and copy the key — "
                       "it authenticates /api/v1/artifacts (pull the model) and "
                       "/api/v1/telemetry (closed loop) for that deployment.",
    },
    {
        "id": "r2", "language": "python",
        "title": "Local serving with live drift alerts",
        "description": "AstraRunner.from_deployment() serves the compressed "
                       "ONNX on your hardware while shipping latency, system "
                       "and input/output stats — prediction/input drift "
                       "alerts appear on the Telemetry tab automatically.",
    },
    {
        "id": "r3", "language": "cli",
        "title": "Download any Pareto trial as ONNX",
        "description": "Pick a point in Pareto 3D Search and Export ONNX — or "
                       "POST /api/models/{id}/pareto/trials/{n}/export and "
                       "fetch the artifact for the exact accuracy/size/latency "
                       "trade-off you need.",
    },
]


def seed_if_empty(session: Session) -> bool:
    """Insert documentation content on first boot, and upgrade stale docs rows
    in place when _DOCS_VERSION changes. Returns True when anything ran."""
    marker = session.exec(
        select(SdkSnippetRow).where(SdkSnippetRow.language == "_meta")
    ).first()
    if marker is not None and marker.filename == _DOCS_VERSION:
        return False

    # Replace docs rows wholesale (docs only — never operational state).
    for row in session.exec(select(SdkSnippetRow)).all():
        session.delete(row)
    for row in session.exec(select(RecipeRow)).all():
        session.delete(row)

    for recipe in _RECIPES:
        session.add(RecipeRow(
            id=recipe["id"], title=recipe["title"],
            description=recipe["description"],
            language=recipe["language"], steps_json=json.dumps([]),
        ))
    for lang, snip in _SNIPPETS.items():
        session.add(SdkSnippetRow(
            language=lang, filename=snip["filename"], code=snip["code"]))
    session.add(SdkSnippetRow(language="_meta", filename=_DOCS_VERSION, code=""))

    session.commit()
    return True
