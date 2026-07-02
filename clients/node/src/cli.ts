#!/usr/bin/env node
// astra CLI — pull / serve / bench a Astra deployment locally.
// Mirrors clients/python/astra_sdk/cli.py. stdlib-only (node:util + node:http);
// the heavy lifting lives in AstraRunner.
//
//     astra pull  --base-url URL --deployment dep_x --api-key KEY
//     astra serve --base-url URL --deployment dep_x --api-key KEY --port 8765
//     astra bench --base-url URL --deployment dep_x --api-key KEY -n 200

import { createServer } from "node:http";
import { parseArgs } from "node:util";

import { AstraRunner, pullArtifact } from "./runner.js";

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "base-url": { type: "string" },
    deployment: { type: "string" },
    "api-key": { type: "string" },
    "cache-dir": { type: "string", default: "~/.cache/astra" },
    port: { type: "string", default: "8765" },
    n: { type: "string", short: "n", default: "200" },
  },
});

async function main(): Promise<number> {
  const cmd = positionals[0];
  const baseUrl = values["base-url"] ?? process.env.ASTRA_BASE_URL;
  const deployment = values.deployment ?? process.env.ASTRA_DEPLOYMENT_ID;
  const apiKey = values["api-key"] ?? process.env.ASTRA_API_KEY;
  const cacheDir = values["cache-dir"]!;

  const require = (): void => {
    const missing: string[] = [];
    if (!deployment) missing.push("--deployment");
    if (!apiKey) missing.push("--api-key");
    if (missing.length) {
      console.error(`missing required option(s): ${missing.join(", ")}`);
      process.exit(1);
    }
  };

  switch (cmd) {
    case "pull": {
      require();
      const path = await pullArtifact({
        baseUrl,
        deploymentId: deployment!,
        apiKey: apiKey!,
        cacheDir,
      });
      console.log(path);
      return 0;
    }

    case "bench": {
      require();
      const runner = await AstraRunner.fromDeployment({
        baseUrl,
        deploymentId: deployment!,
        apiKey: apiKey!,
        cacheDir,
      });
      const total = Number(values.n) || 200;
      const lats: number[] = [];
      try {
        for (let i = 0; i < total; i++) {
          const out = await runner.run(null);
          lats.push(out.latencyMs);
          if ((i + 1) % 50 === 0) console.error(`  ${i + 1}/${total}`);
        }
      } finally {
        await runner.close();
      }
      lats.sort((a, b) => a - b);
      const pct = (q: number): number =>
        lats[Math.min(lats.length - 1, Math.floor(q * lats.length))] ?? 0;
      console.log(
        JSON.stringify(
          {
            n: lats.length,
            p50Ms: round3(pct(0.5)),
            p95Ms: round3(pct(0.95)),
            p99Ms: round3(pct(0.99)),
            meanMs: round3(lats.reduce((a, b) => a + b, 0) / Math.max(1, lats.length)),
          },
          null,
          2,
        ),
      );
      return 0;
    }

    case "serve": {
      require();
      const runner = await AstraRunner.fromDeployment({
        baseUrl,
        deploymentId: deployment!,
        apiKey: apiKey!,
        cacheDir,
      });
      const port = Number(values.port) || 8765;
      const server = createServer((req, res) => {
        if (req.method !== "POST" || (req.url ?? "").replace(/\/+$/, "") !== "/infer") {
          res.writeHead(404).end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          void (async () => {
            try {
              const text = Buffer.concat(chunks).toString() || "{}";
              const body = JSON.parse(text) as { inputs?: Record<string, never> | null };
              const out = await runner.run(body.inputs ?? null);
              const payload = JSON.stringify({ latencyMs: out.latencyMs, outputs: out.outputs });
              res.writeHead(200, { "Content-Type": "application/json" }).end(payload);
            } catch (err) {
              const msg = JSON.stringify({ error: err instanceof Error ? err.message : String(err) });
              res.writeHead(400, { "Content-Type": "application/json" }).end(msg);
            }
          })();
        });
      });
      server.listen(port, "127.0.0.1", () => {
        console.log(
          `serving ${deployment} on http://127.0.0.1:${port}/infer ` +
            `(Ctrl-C to stop; telemetry → the Astra dashboard)`,
        );
      });
      const shutdown = async (): Promise<void> => {
        server.close();
        await runner.close();
        process.exit(0);
      };
      process.on("SIGINT", () => void shutdown());
      process.on("SIGTERM", () => void shutdown());
      // Run until a signal arrives.
      return await new Promise<number>(() => {});
    }

    default:
      console.error(
        "usage: astra <pull|serve|bench> --deployment dep_x --api-key KEY " +
          "[--base-url URL] [--port 8765] [-n 200]\n" +
          "(options also read from ASTRA_BASE_URL / ASTRA_DEPLOYMENT_ID / ASTRA_API_KEY)",
      );
      return 1;
  }
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
