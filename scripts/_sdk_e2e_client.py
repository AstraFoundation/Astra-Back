#!/usr/bin/env python3
"""SDK e2e step 4 — runs with the WHEEL-INSTALLED astra_sdk from /tmp.

Pulls the deployed artifact, serves it locally with AstraRunner, fires
normal traffic then distribution-shifted traffic, and flushes telemetry.
No repo imports — only the installed package + numpy.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True)
    args = ap.parse_args()
    with open(args.handoff) as f:
        h = json.load(f)

    import numpy as np

    import astra_sdk
    from astra_sdk import AstraRunner

    assert not astra_sdk.__file__.startswith("/Users/kwonminjae/Desktop/Astra"), (
        f"must import the INSTALLED wheel, got {astra_sdk.__file__}")
    print(f"   using {astra_sdk.__file__}")

    runner = AstraRunner.from_deployment(
        h["deploymentId"], h["apiKey"], base_url=h["baseUrl"],
        cache_dir="/tmp/astra-ai-sdk-cache",
    )
    print(f"   artifact cached at {runner.model_path}")

    meta = runner._input_meta  # noqa: SLF001 — e2e introspection
    name = meta[0].name
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta[0].shape]
    rng = np.random.default_rng(7)

    t0 = time.time()
    for i in range(100):
        runner.run({name: rng.standard_normal(shape).astype(np.float32)})
    print(f"   100 baseline inferences in {time.time() - t0:.1f}s")

    # Distribution shift: +5 sigma on every input feature.
    for i in range(60):
        x = rng.standard_normal(shape).astype(np.float32) + 5.0
        runner.run({name: x})
    print("   60 shifted inferences (input mean +5.0)")

    runner.close()  # final telemetry flush (atexit budget)
    print("   telemetry flushed")


if __name__ == "__main__":
    main()
