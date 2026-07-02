"""astra CLI — pull / serve / bench a Astra deployment locally.

    astra pull  --base-url URL --deployment dep_x --api-key KEY
    astra serve --base-url URL --deployment dep_x --api-key KEY --port 8765
    astra bench --base-url URL --deployment dep_x --api-key KEY -n 200

stdlib-only (argparse + http.server); the heavy lifting lives in AstraRunner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=os.environ.get("ASTRA_BASE_URL"),
                   help="Astra origin (optional; defaults to the hosted origin, "
                        "env ASTRA_BASE_URL)")
    p.add_argument("--deployment", default=os.environ.get("ASTRA_DEPLOYMENT_ID"),
                   help="deployment id, e.g. dep_ab12cd34ef (env ASTRA_DEPLOYMENT_ID)")
    p.add_argument("--api-key", default=os.environ.get("ASTRA_API_KEY"),
                   help="deployment API key (env ASTRA_API_KEY)")
    p.add_argument("--cache-dir", default="~/.cache/astra")


def _require(args) -> None:
    missing = [n for n in ("deployment", "api_key")
               if not getattr(args, n)]
    if missing:
        sys.exit(f"missing required option(s): {', '.join('--' + m.replace('_', '-') for m in missing)}")


def cmd_pull(args) -> int:
    from .runner import pull_artifact

    _require(args)
    path = pull_artifact(args.deployment, args.api_key, base_url=args.base_url,
                         cache_dir=args.cache_dir)
    print(path)
    return 0


def cmd_bench(args) -> int:
    from .runner import AstraRunner

    _require(args)
    runner = AstraRunner.from_deployment(
        args.deployment, args.api_key, base_url=args.base_url, cache_dir=args.cache_dir)
    lats: list[float] = []
    try:
        for i in range(args.n):
            out = runner.run(None)
            lats.append(out["latencyMs"])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.n}", file=sys.stderr)
    finally:
        runner.close()
    lats.sort()
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]  # noqa: E731
    print(json.dumps({
        "n": len(lats),
        "p50Ms": round(p(0.50), 3),
        "p95Ms": round(p(0.95), 3),
        "p99Ms": round(p(0.99), 3),
        "meanMs": round(sum(lats) / len(lats), 3),
    }, indent=2))
    return 0


def cmd_serve(args) -> int:
    import http.server

    from .runner import AstraRunner

    _require(args)
    runner = AstraRunner.from_deployment(
        args.deployment, args.api_key, base_url=args.base_url, cache_dir=args.cache_dir)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — http.server API
            if self.path.rstrip("/") != "/infer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                out = runner.run(body.get("inputs"))
                payload = json.dumps({
                    "latencyMs": out["latencyMs"],
                    "outputs": [
                        {**o, "data": None} for o in out["outputs"]
                    ],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:  # noqa: BLE001 — boundary
                msg = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def log_message(self, fmt, *fmt_args):  # quiet
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving {args.deployment} on http://127.0.0.1:{args.port}/infer "
          f"(Ctrl-C to stop; telemetry → the Astra dashboard)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astra",
        description="Serve Astra-compressed models locally with built-in telemetry.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="download the deployed artifact")
    _add_common(p_pull)
    p_pull.set_defaults(fn=cmd_pull)

    p_serve = sub.add_parser("serve", help="local HTTP endpoint (POST /infer)")
    _add_common(p_serve)
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(fn=cmd_serve)

    p_bench = sub.add_parser("bench", help="local latency benchmark")
    _add_common(p_bench)
    p_bench.add_argument("-n", type=int, default=200)
    p_bench.set_defaults(fn=cmd_bench)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
