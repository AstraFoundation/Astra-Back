"""Astra SDK — run Astra-compressed models ON-DEVICE, with a closed telemetry loop.

Astra never runs your model server-side. The SDK pulls the compressed artifact
once and serves it on YOUR hardware with onnxruntime:

    pip install 'astra-ai-sdk[serve]'

    from astra_sdk import AstraRunner

    runner = AstraRunner.from_deployment(deployment_id, api_key)
    out = runner.run({"input": my_array})     # local inference, in your process
    runner.close()

Closed-loop telemetry: every local request records latency breakdown, system
snapshots and windowed input/output stats. When the device is offline these are
buffered durably on disk (~/.cache/astra/<deployment>/telemetry/) and flushed to
Astra the moment connectivity returns — powering the live Telemetry tab and
prediction/input drift alerts, then deleted after the server acks them. Opt out:
report_telemetry=False or ASTRA_SDK_TELEMETRY=0; disable disk buffering with
ASTRA_SDK_SPOOL=0.
"""

from __future__ import annotations

from ._http import AstraApiError
from .runner import AstraRunner, AstraRunnerError, pull_artifact
from .telemetry import AstraTelemetryReporter

__all__ = [
    "AstraApiError",
    "AstraRunner",
    "AstraRunnerError",
    "AstraTelemetryReporter",
    "pull_artifact",
]
__version__ = "0.4.0"
