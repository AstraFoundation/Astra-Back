// Background telemetry reporter — fault-tolerant, offline-durable by construction.
// Mirrors clients/python/astra_sdk/telemetry.py.
//
// Design contract: NOTHING here may ever throw into the caller's serving path.
// Events buffer into a bounded array (drop-oldest under pressure, with the drop
// count itself reported); an unref'd interval timer drains them into a durable
// on-disk *spool* (one JSON segment per batch) BEFORE touching the network, then
// POSTs each pending segment to POST /api/v1/telemetry/{deployment_id}/batch and
// deletes a segment only after the server acks it (2xx). So the closed loop
// survives being offline: while the device has no connectivity, segments
// accumulate on disk and survive process restarts; when connectivity returns they
// flush oldest-first and are deleted. Each segment carries a client-generated
// `batchId` the server dedups on, so a re-send after a crash never double-counts.
//
// Disable telemetry entirely with enabled:false / ASTRA_SDK_TELEMETRY=0.
// Disable disk buffering (in-memory only, best-effort, lost on exit) with
// ASTRA_SDK_SPOOL=0.

import { randomBytes } from "node:crypto";
import {
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { HttpSession } from "./http.js";
import { type Sampleable, WindowAggregator } from "./stats.js";
import { type OrtRuntimeInfo, runtimeFingerprint, systemSample } from "./system.js";

function envFloat(name: string, def: number): number {
  const v = process.env[name];
  if (v === undefined) return def;
  const n = Number(v);
  return Number.isFinite(n) ? n : def;
}

const QUEUE_MAX = 10_000;
const BATCH_MAX = 450; // below the server's 500-item cap
const MEM_PENDING_MAX = 64; // in-memory pending batches when the spool is off/unwritable
const FLUSH_INTERVAL_S = envFloat("ASTRA_SDK_FLUSH_INTERVAL_S", 5);
const SNAPSHOT_INTERVAL_S = envFloat("ASTRA_SDK_SNAPSHOT_INTERVAL_S", 30);
const WINDOW_INTERVAL_S = envFloat("ASTRA_SDK_WINDOW_INTERVAL_S", 60);
const WINDOW_MAX_REQUESTS = Math.trunc(envFloat("ASTRA_SDK_WINDOW_MAX_REQUESTS", 200));
const SPOOL_MAX_BYTES = Math.trunc(envFloat("ASTRA_SDK_SPOOL_MAX_MB", 32) * 1024 * 1024);
const DEFAULT_CACHE = "~/.cache/astra";
const ATEXIT_BUDGET_MS = 3000;

const round = (x: number, p: number): number => {
  const f = 10 ** p;
  return Math.round(x * f) / f;
};

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** uuid4-hex analogue (32 lowercase hex chars) — matches Python's uuid4().hex. */
const hexUuid = (): string => randomBytes(16).toString("hex");

// Same `~` expansion the runner uses (src/runner.ts). Kept local so telemetry has
// no import cycle with the runner, mirroring Python's self-contained _DEFAULT_CACHE.
function expandHome(dir: string): string {
  if (dir === "~") return homedir();
  if (dir.startsWith("~/") || dir.startsWith("~\\")) return join(homedir(), dir.slice(2));
  return dir;
}

export function telemetryEnabled(flag?: boolean): boolean {
  if (flag === false) return false;
  const v = process.env.ASTRA_SDK_TELEMETRY;
  return !(v === "0" || v === "false" || v === "no");
}

function spoolEnabled(): boolean {
  const v = process.env.ASTRA_SDK_SPOOL;
  return !(v === "0" || v === "false" || v === "no");
}

export interface TelemetryReporterOptions {
  sdkVersion: string;
  enabled?: boolean;
  activeProvider?: string;
  ortVersion?: string;
  availableProviders?: string[];
  /** Root for the durable spool (defaults to ~/.cache/astra, alongside the
   *  artifact cache). ASTRA_SDK_SPOOL_DIR overrides the full path. */
  cacheDir?: string;
}

export interface RecordEventOptions {
  latencyMs: number;
  preMs?: number;
  postMs?: number;
  success?: boolean;
  errorCode?: string;
  batchSize?: number;
  region?: string;
  inputSig?: string;
}

/** One durable batch — a JSON spool segment / one POST body. */
export interface TelemetryBatch {
  clientId: string;
  batchId: string;
  events: Record<string, unknown>[];
  snapshots: Record<string, unknown>[];
  windows: Record<string, unknown>[];
}

export class AstraTelemetryReporter {
  readonly enabled: boolean;
  private readonly clientId: string;
  private readonly deploymentId: string;
  private readonly fingerprint: Record<string, unknown>;
  private readonly aggregator = new WindowAggregator();

  private events: Record<string, unknown>[] = [];
  private snapshots: Record<string, unknown>[] = [];
  private windows: Record<string, unknown>[] = [];
  private memPending: TelemetryBatch[] = []; // pending batches when the spool is off/unwritable
  private dropped = 0;
  private sentEvents = 0;
  private windowRequests = 0;
  private segSeq = 0;

  // Durable spool dir: ~/.cache/astra/<deployment>/telemetry/ (alongside the
  // artifact cache). ASTRA_SDK_SPOOL_DIR overrides; ASTRA_SDK_SPOOL=0 disables.
  private spool: boolean;
  private readonly spoolDir: string;

  private http: HttpSession | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private beforeExitHandler: (() => void) | null = null;
  private closed = false;
  private flushing = false;
  private lastSnapshot = Date.now();
  private lastWindow = Date.now();
  private throughputMarkerTime = Date.now();
  private throughputMarkerN = 0;

  constructor(
    baseUrl: string,
    deploymentId: string,
    apiKey: string,
    opts: TelemetryReporterOptions,
  ) {
    this.enabled = telemetryEnabled(opts.enabled);
    this.clientId = `sdk_${randomBytes(5).toString("hex")}`;
    this.deploymentId = deploymentId;
    const ort: OrtRuntimeInfo = {
      ortVersion: opts.ortVersion,
      availableProviders: opts.availableProviders,
      activeProvider: opts.activeProvider,
    };
    this.fingerprint = runtimeFingerprint(opts.sdkVersion, ort);

    this.spool = this.enabled && spoolEnabled();
    const override = process.env.ASTRA_SDK_SPOOL_DIR;
    const base = override ?? join(opts.cacheDir ?? DEFAULT_CACHE, deploymentId, "telemetry");
    this.spoolDir = expandHome(base);
    if (this.spool) {
      try {
        mkdirSync(this.spoolDir, { recursive: true });
      } catch {
        this.spool = false; // unwritable → fall back to in-memory pending
      }
    }

    if (this.enabled) {
      this.http = new HttpSession(baseUrl, apiKey, {
        timeout: 10,
        maxAttempts: 2,
        maxBackoff: 30,
      });
      this.timer = setInterval(() => {
        void this.tick();
      }, FLUSH_INTERVAL_S * 1000);
      this.timer.unref();
      this.beforeExitHandler = () => {
        void this.close();
      };
      process.once("beforeExit", this.beforeExitHandler);
      // Startup recovery: ship any segments left by a prior (offline) session.
      void this.startupRecover();
    }
  }

  // ── recording (hot path — must be cheap and never raise) ──────────────────

  recordEvent(opts: RecordEventOptions): void {
    if (!this.enabled) return;
    try {
      const event: Record<string, unknown> = {
        id: hexUuid(),
        ts: new Date().toISOString(),
        latencyMs: round(opts.latencyMs, 3),
        success: opts.success ?? true,
        batchSize: Math.trunc(opts.batchSize ?? 1),
        region: opts.region ?? "local",
      };
      if (opts.preMs !== undefined) event.preMs = round(opts.preMs, 3);
      if (opts.postMs !== undefined) event.postMs = round(opts.postMs, 3);
      if (opts.errorCode) event.errorCode = opts.errorCode;
      if (opts.inputSig) event.inputSig = opts.inputSig;

      if (this.events.length >= QUEUE_MAX) {
        this.events.shift();
        this.dropped += 1;
      }
      this.events.push(event);
      this.sentEvents += 1;
      this.windowRequests += 1;
      if (this.windowRequests >= WINDOW_MAX_REQUESTS) this.takeWindow();
    } catch {
      /* telemetry must never break serving */
    }
  }

  observe(inputs: Record<string, Sampleable> | null | undefined, output: Sampleable | null): void {
    if (!this.enabled) return;
    try {
      this.aggregator.observe(inputs, output);
    } catch {
      /* ignore */
    }
  }

  // ── background loop ───────────────────────────────────────────────────────

  private async startupRecover(): Promise<void> {
    try {
      await this.sendPending();
    } catch {
      /* offline / unreadable — retried on the next tick or next process */
    }
  }

  private async tick(): Promise<void> {
    if (this.closed || this.flushing) return;
    this.flushing = true;
    try {
      const now = Date.now();
      if (now - this.lastSnapshot >= SNAPSHOT_INTERVAL_S * 1000) {
        this.takeSnapshot();
        this.lastSnapshot = now;
      }
      if (
        now - this.lastWindow >= WINDOW_INTERVAL_S * 1000 ||
        this.windowRequests >= WINDOW_MAX_REQUESTS
      ) {
        this.takeWindow();
        this.lastWindow = now;
      }
      await this.flush();
    } catch {
      /* the loop must survive anything */
    } finally {
      this.flushing = false;
    }
  }

  private takeSnapshot(): void {
    const now = Date.now();
    const elapsedMin = Math.max(1e-6, (now - this.throughputMarkerTime) / 60000);
    const rpm = (this.sentEvents - this.throughputMarkerN) / elapsedMin;
    this.throughputMarkerTime = now;
    this.throughputMarkerN = this.sentEvents;
    this.snapshots.push({
      id: hexUuid(),
      ts: new Date().toISOString(),
      ...systemSample(),
      throughputRpm: round(rpm, 2),
      droppedEvents: this.dropped,
      ...this.fingerprint,
    });
    if (this.snapshots.length > 64) this.snapshots.shift();
  }

  private takeWindow(): void {
    this.windowRequests = 0;
    const w = this.aggregator.flush();
    if (w) {
      if (w.id === undefined) w.id = hexUuid();
      this.windows.push(w);
      if (this.windows.length > 64) this.windows.shift();
    }
  }

  // ── flush: drain in-memory → durable spool → network (delete-after-ack) ────

  private drainToBatch(): TelemetryBatch | null {
    const events = this.events.splice(0, Math.min(BATCH_MAX, this.events.length));
    const snapshots = this.snapshots.splice(0, this.snapshots.length);
    const windows = this.windows.splice(0, this.windows.length);
    if (!events.length && !snapshots.length && !windows.length) return null;
    return { clientId: this.clientId, batchId: hexUuid(), events, snapshots, windows };
  }

  /** Persist a drained batch durably (or, if the spool is off/unwritable, hold it
   *  in a bounded in-memory queue that drops oldest under pressure). */
  private persist(batch: TelemetryBatch): void {
    if (this.spool && this.spoolWrite(batch)) return;
    if (this.memPending.length >= MEM_PENDING_MAX) {
      this.memPending.shift();
      this.dropped += 1;
    }
    this.memPending.push(batch);
  }

  /** One flush tick: drain a single batch to durable storage, then ship pending. */
  private async flush(): Promise<boolean> {
    if (!this.http) return true;
    const batch = this.drainToBatch();
    if (batch) this.persist(batch);
    return this.sendPending();
  }

  /** Drain ALL buffered in-memory data into durable storage (no network). Used on
   *  close so a short session never loses events past a single batch. */
  private drainAllToStorage(): void {
    if (!this.http) return;
    for (;;) {
      const batch = this.drainToBatch();
      if (!batch) break;
      this.persist(batch);
    }
  }

  /** Send pending batches oldest-first; delete each only after a 2xx ack. Stops
   *  at the first failure (still offline) and returns false. */
  private async sendPending(): Promise<boolean> {
    if (!this.http) return true;
    if (this.spool) {
      for (const seg of this.spoolSegments()) {
        let batch: unknown;
        try {
          batch = JSON.parse(readFileSync(seg, "utf-8"));
        } catch {
          this.unlink(seg); // corrupt segment — drop it
          continue;
        }
        if (await this.sendBatch(batch)) {
          this.unlink(seg);
        } else {
          return false;
        }
      }
    }
    while (this.memPending.length > 0) {
      if (await this.sendBatch(this.memPending[0]!)) {
        this.memPending.shift();
      } else {
        return false;
      }
    }
    return true;
  }

  private async sendBatch(batch: unknown): Promise<boolean> {
    try {
      await this.http!.request("POST", `/api/v1/telemetry/${this.deploymentId}/batch`, {
        json: batch,
      });
      return true;
    } catch {
      return false;
    }
  }

  // ── durable spool helpers ──────────────────────────────────────────────────

  private spoolSegments(): string[] {
    try {
      return readdirSync(this.spoolDir)
        .filter((f) => f.startsWith("seg-") && f.endsWith(".json"))
        .sort()
        .map((f) => join(this.spoolDir, f));
    } catch {
      return [];
    }
  }

  private spoolWrite(batch: TelemetryBatch): boolean {
    try {
      this.enforceSpoolCap();
      this.segSeq += 1;
      const stem = `seg-${String(Date.now()).padStart(13, "0")}-${String(this.segSeq).padStart(4, "0")}`;
      const tmp = join(this.spoolDir, `${stem}.part`);
      writeFileSync(tmp, JSON.stringify(batch), "utf-8");
      renameSync(tmp, join(this.spoolDir, `${stem}.json`));
      return true;
    } catch {
      return false;
    }
  }

  // Bounded disk usage: drop oldest segments when over the byte cap.
  private enforceSpoolCap(): void {
    try {
      const segs = this.spoolSegments();
      let total = segs.reduce((s, p) => s + this.size(p), 0);
      let i = 0;
      while (total > SPOOL_MAX_BYTES && i < segs.length) {
        total -= this.size(segs[i]!);
        this.unlink(segs[i]!);
        this.dropped += 1;
        i += 1;
      }
    } catch {
      /* ignore */
    }
  }

  private size(p: string): number {
    try {
      return statSync(p).size;
    } catch {
      return 0;
    }
  }

  private unlink(p: string): void {
    try {
      unlinkSync(p);
    } catch {
      /* already gone */
    }
  }

  // ── shutdown ──────────────────────────────────────────────────────────────

  /** Durably persist everything still in memory, best-effort flush within a small
   *  budget, and leave any unsent segments on disk for next time. Idempotent. */
  async close(): Promise<void> {
    if (!this.enabled || this.closed) return;
    this.closed = true;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.beforeExitHandler) {
      process.removeListener("beforeExit", this.beforeExitHandler);
      this.beforeExitHandler = null;
    }
    try {
      // Always ship at least one snapshot per session — it carries the runtime
      // fingerprint the dashboard's client-hosts table shows (short sessions
      // would otherwise never hit the 30s cadence).
      this.takeSnapshot();
      this.takeWindow();
      this.drainAllToStorage(); // persist all in-memory to the durable spool
      const deadline = Date.now() + ATEXIT_BUDGET_MS;
      while (Date.now() < deadline) {
        if (await this.sendPending()) break;
        await sleep(200);
      }
    } catch {
      /* best effort — unsent segments stay on disk for the next process */
    } finally {
      this.http?.close();
    }
  }
}
