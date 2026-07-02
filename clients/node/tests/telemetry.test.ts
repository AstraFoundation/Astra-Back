import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { AstraTelemetryReporter } from "../src/telemetry.js";
import { installFetch } from "./_mock.js";

interface Batch {
  clientId: string;
  batchId?: string;
  events?: Record<string, unknown>[];
  snapshots?: Record<string, unknown>[];
  windows?: Record<string, unknown>[];
}

function makeBackend() {
  const batches: Batch[] = [];
  const state = { failNext: 0, offline: false };
  installFetch((req) => {
    expect(req.path).toBe("/api/v1/telemetry/dep_x/batch");
    // Offline: non-retryable status so HttpSession gives up immediately (no
    // backoff), the way a durable spool would keep a segment for later.
    if (state.offline) return { status: 500, json: { detail: { code: "offline" } } };
    if (state.failNext > 0) {
      state.failNext -= 1;
      return { status: 503, json: { detail: { code: "unavailable" } } };
    }
    batches.push(req.body as Batch);
    return { json: { accepted: {}, dropped: 0 } };
  });
  return {
    batches,
    setFail: (n: number) => {
      state.failNext = n;
    },
    setOffline: (v: boolean) => {
      state.offline = v;
    },
    events: () => batches.flatMap((b) => b.events ?? []),
    windows: () => batches.flatMap((b) => b.windows ?? []),
  };
}

// spool dir is set per-test (below) so nothing touches the real ~/.cache/astra.
let spoolDir: string;

function reporter(opts: Partial<{ enabled: boolean }> = {}) {
  return new AstraTelemetryReporter("http://test", "dep_x", "astra_sk_test", {
    sdkVersion: "0.2.0",
    ...opts,
  });
}

/** Reach the private flush to drive the spool loop deterministically (no timers). */
type Drivable = { flush(): Promise<boolean> };
const drive = (r: AstraTelemetryReporter): Drivable => r as unknown as Drivable;

const jsonSegs = (dir: string): string[] =>
  readdirSync(dir).filter((f) => f.startsWith("seg-") && f.endsWith(".json"));

beforeEach(() => {
  spoolDir = mkdtempSync(join(tmpdir(), "astra-spool-"));
  process.env.ASTRA_SDK_SPOOL_DIR = spoolDir;
});

afterEach(() => {
  delete process.env.ASTRA_SDK_TELEMETRY;
  delete process.env.ASTRA_SDK_SPOOL;
  delete process.env.ASTRA_SDK_SPOOL_DIR;
  try {
    rmSync(spoolDir, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
});

describe("AstraTelemetryReporter", () => {
  it("flushes buffered events on close", async () => {
    const backend = makeBackend();
    const rep = reporter();
    for (let i = 0; i < 25; i++) {
      rep.recordEvent({ latencyMs: i, preMs: 0.1, postMs: 0.1 });
    }
    await rep.close();
    expect(backend.events()).toHaveLength(25);
    const ev = backend.events()[0]!;
    for (const k of ["id", "ts", "latencyMs", "success", "batchSize", "region", "preMs", "postMs"]) {
      expect(ev).toHaveProperty(k);
    }
    // Every batch carries a client-generated batchId the server dedups on.
    expect(backend.batches.every((b) => typeof b.batchId === "string")).toBe(true);
    // Delete-after-ack: acked segments are gone from disk.
    expect(jsonSegs(spoolDir)).toHaveLength(0);
  });

  it("recording is cheap and never blocks the hot path", async () => {
    const backend = makeBackend();
    const rep = reporter();
    const t0 = performance.now();
    for (let i = 0; i < 5000; i++) rep.recordEvent({ latencyMs: 1.0 });
    expect(performance.now() - t0).toBeLessThan(1000);
    await rep.close();
    expect(backend.events().length).toBe(5000);
  });

  it("survives one failed flush and recovers the events", async () => {
    const backend = makeBackend();
    backend.setFail(1);
    const rep = reporter();
    for (let i = 0; i < 10; i++) rep.recordEvent({ latencyMs: i });
    await rep.close(); // close retries within its budget
    expect(backend.events()).toHaveLength(10);
  });

  it("spools to disk while offline, then sends and deletes on reconnect", async () => {
    const backend = makeBackend();
    backend.setOffline(true);
    const rep = reporter();
    for (let i = 0; i < 5; i++) rep.recordEvent({ latencyMs: i });

    // Drain in-memory → durable spool BEFORE the network; the send fails (offline)
    // so the segment must remain on disk.
    await drive(rep).flush();
    expect(jsonSegs(spoolDir)).toHaveLength(1);
    expect(backend.events()).toHaveLength(0);

    // Reconnect: the next flush ships the pending segment oldest-first and deletes
    // it only after the 2xx ack.
    backend.setOffline(false);
    await drive(rep).flush();
    expect(jsonSegs(spoolDir)).toHaveLength(0);
    expect(backend.events()).toHaveLength(5);

    await rep.close();
  });

  it("ASTRA_SDK_SPOOL=0 keeps a bounded in-memory queue and writes no segments", async () => {
    process.env.ASTRA_SDK_SPOOL = "0";
    const backend = makeBackend();
    const rep = reporter();
    for (let i = 0; i < 5; i++) rep.recordEvent({ latencyMs: i });
    await rep.close();
    expect(backend.events()).toHaveLength(5);
    expect(jsonSegs(spoolDir)).toHaveLength(0);
  });

  it("is disabled via the enabled flag", async () => {
    const backend = makeBackend();
    const rep = reporter({ enabled: false });
    rep.recordEvent({ latencyMs: 1.0 });
    await rep.close();
    expect(backend.batches).toHaveLength(0);
  });

  it("is disabled via ASTRA_SDK_TELEMETRY=0", async () => {
    process.env.ASTRA_SDK_TELEMETRY = "0";
    const backend = makeBackend();
    const rep = reporter();
    rep.recordEvent({ latencyMs: 1.0 });
    await rep.close();
    expect(backend.batches).toHaveLength(0);
  });

  it("flushes the open window stats on close", async () => {
    const backend = makeBackend();
    const rep = reporter();
    for (let i = 0; i < 20; i++) {
      const x = Array.from({ length: 8 }, () => Math.random() - 0.5);
      const logits = Array.from({ length: 5 }, () => Math.random());
      rep.observe({ input: { data: x, dims: [1, 8] } }, { data: logits, dims: [1, 5] });
      rep.recordEvent({ latencyMs: 1.0 });
    }
    await rep.close();
    const windows = backend.windows();
    expect(windows.length).toBeGreaterThan(0);
    const w = windows[0]! as Record<string, unknown>;
    expect(w.n).toBe(20);
    const inputs = w.inputs as Record<string, Record<string, number>>;
    expect(inputs).toHaveProperty("input");
    expect(Math.abs(inputs.input!.mean!)).toBeLessThan(1.0);
    const output = w.output as Record<string, unknown>;
    expect(output).toHaveProperty("classDist");
    expect((output.hist as number[]).length).toBe(16);
  });
});
