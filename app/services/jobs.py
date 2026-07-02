"""The compression pipeline worker body — a plain function the dispatcher runs.

`execute_pipeline` carries no transport assumptions: it is invoked either inline
on a daemon thread (single-box / tests) or inside an arq worker process (scaled
deployment). Cancellation/timeout arrive through the injected `should_cancel`
predicate (in-process event, or a Redis flag + deadline). All progress/log state
is written to the DB, so SSE/poll readers in any API process observe it without
needing to share memory with the worker.
"""

from __future__ import annotations

import shutil
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from app.config import get_settings, iso
from app.db import open_session
from app.dbmodels import (
    ActivityRow,
    AlertRow,
    IngestionLogRow,
    IngestionRunRow,
    ModelRow,
    RunRow,
)
from app.repositories import put_cached_result
from app.services.storage import artifact_key as make_artifact_key
from app.services.storage import get_storage, ingested_key


def _now_iso() -> str:
    return iso(datetime.now(timezone.utc))


@dataclass
class JobCtx:
    run_id: str
    model_id: str
    user_id: str


def execute_pipeline(
    *,
    run_id: str,
    model_id: str,
    user_id: str,
    model_name: str,
    file_name: str,
    source_key: str | None,
    input_shape: list[int] | None,
    declared_format: str,
    should_cancel: Callable[[], bool],
) -> None:
    job = JobCtx(run_id=run_id, model_id=model_id, user_id=user_id)
    settings = get_settings()
    seq = 0
    seq_lock = threading.Lock()

    def emit(level: str, message: str) -> None:
        nonlocal seq
        with seq_lock:
            seq += 1
            current = seq
        with open_session() as s:
            s.add(IngestionLogRow(
                run_id=job.run_id, seq=current, ts=_now_iso(),
                level=level, message=message,
            ))
            s.commit()

    def progress(pct: int) -> None:
        with open_session() as s:
            run = s.get(IngestionRunRow, job.run_id)
            if run:
                run.progress = pct
                s.add(run)
            dash = s.get(RunRow, f"run_{job.run_id}")
            if dash:
                dash.progress_pct = pct
                # Same denominator the run was created with (4 in fast mode) —
                # never claim the full real-mode trial budget mid-run.
                total = 4 if settings.fast_pipeline else settings.pareto_trials
                dash.iter = f"{round(total * pct / 100)} / {total}"
                s.add(dash)
            s.commit()

    # Worker has picked the job up — flip the dashboard run queued → running so
    # the UI reflects real execution (the row was created "queued" at enqueue
    # time). Only advance from "queued" so a cancel/fail race isn't overwritten.
    with open_session() as s:
        dash = s.get(RunRow, f"run_{job.run_id}")
        if dash and dash.status == "queued":
            dash.status = "running"
            s.add(dash)
            s.commit()

    # Per-job local scratch: the engine reads/writes files here, decoupled
    # from where source/artifact actually live (local dir or object store).
    work = Path(settings.work_dir) / job.run_id
    work.mkdir(parents=True, exist_ok=True)
    storage = get_storage()
    source_path: str | None = None
    try:
        if source_key is None:
            from app.services.model_factory import synthesize

            emit("INFO", f"No artifact bytes supplied for {file_name} — synthesizing a "
                         f"real reference model for the declared format")
            synth = synthesize(
                file_name, out_dir=str(work),
                fast=settings.fast_pipeline, seed=settings.seed,
            )
            emit("INFO", f"Synthesized: {synth.note}")
            source_path = synth.path
            input_shape = synth.input_shape
            declared_format = synth.declared_format
        else:
            # Pull the uploaded source down from object storage to scratch.
            source_path = str(work / Path(source_key).name)
            storage.download_to(source_key, source_path)

        # ── format dispatch ───────────────────────────────────────────────
        # Weights-only containers (no executable graph) take the honest
        # weight-only pipeline; convertible containers (Keras/TFLite/
        # frozen-pb) are converted to REAL ONNX and take the full pipeline;
        # everything else goes straight to the engine.
        ext = Path(source_path).suffix.lower()
        bundle = None
        if ext in (".pt", ".pth", ".bin", ".ckpt"):
            from app.engine.weight_loaders import load_torch_state_dict

            bundle = load_torch_state_dict(source_path)  # None → full module
        elif ext == ".safetensors":
            from app.engine.weight_loaders import load_safetensors

            bundle = load_safetensors(source_path)
        elif ext == ".mlmodel":
            from app.engine.weight_loaders import load_coreml

            bundle = load_coreml(source_path)
        elif ext == ".gguf":
            from app.engine.weight_loaders import load_gguf

            bundle = load_gguf(source_path)
        elif ext in (".h5", ".keras"):
            from app.engine.converters import ConversionError, convert_keras_to_onnx

            try:
                source_path, input_shape = convert_keras_to_onnx(
                    source_path, str(work), emit,
                )
            except ConversionError as exc:
                emit("WARN", f"Full Keras→ONNX conversion unavailable: {exc}")
                emit("INFO", "Falling back to honest weight-only analysis (h5py)")
                from app.engine.weight_loaders import load_keras_h5

                bundle = load_keras_h5(source_path)
        elif ext == ".tflite":
            from app.engine.converters import convert_tflite_to_onnx

            source_path, input_shape = convert_tflite_to_onnx(
                source_path, str(work), emit,
            )
        elif ext == ".pb":
            from app.engine.converters import convert_frozen_pb_to_onnx

            source_path, input_shape = convert_frozen_pb_to_onnx(
                source_path, str(work), emit,
            )

        if bundle is not None:
            from app.engine.statedict_pipeline import run_weight_only_pipeline

            artifacts = run_weight_only_pipeline(
                model_id=job.model_id,
                model_name=model_name,
                file_name=file_name,
                source_path=source_path,
                bundle=bundle,
                run_id=job.run_id,
                emit=emit,
                progress=progress,
                storage_dir=str(work),
                should_cancel=should_cancel,
            )
            _on_success(job, model_name, artifacts, weights_only=True)
            return

        from app.engine.adapter import run_pipeline

        n_trials = 4 if settings.fast_pipeline else settings.pareto_trials
        n_probes = 8 if settings.fast_pipeline else settings.n_probes
        max_ops = 8 if settings.fast_pipeline else settings.max_compressible_ops
        # Fast mode forces a fixed 4-trial budget (below any adaptive floor).
        adaptive = (not settings.fast_pipeline) and settings.pareto_adaptive

        artifacts = run_pipeline(
            model_id=job.model_id,
            model_name=model_name,
            file_name=file_name,
            source_path=source_path,
            input_shape=input_shape,
            declared_format=declared_format,
            run_id=job.run_id,
            emit=emit,
            progress=progress,
            n_trials=n_trials,
            n_probes=n_probes,
            seed=settings.seed,
            max_compressible_ops=max_ops,
            storage_dir=str(work),
            should_cancel=should_cancel,
            benchmark_samples=50 if settings.fast_pipeline else 200,
            guarantee_mode=settings.guarantee_mode,
            tau=settings.tau,
            adaptive=adaptive,
            trials_per_dim=settings.pareto_trials_per_dim,
            startup_trials=settings.pareto_startup_trials,
            min_trials=settings.pareto_min_trials,
            early_stop=settings.pareto_early_stop,
            hv_patience=settings.pareto_hv_patience,
            hv_epsilon=settings.pareto_hv_epsilon,
        )
        _on_success(job, model_name, artifacts)
    except Exception as exc:  # noqa: BLE001 — job boundary
        cancelled = exc.__class__.__name__ == "PipelineCancelled"
        if cancelled:
            emit("ERROR", f"Pipeline cancelled (timeout {settings.job_timeout_sec}s "
                          f"or explicit cancel)")
        else:
            emit("ERROR", f"Pipeline failed: {exc}")
            traceback.print_exc()
        _on_failure(job, model_name, str(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _on_success(
    job: JobCtx, model_name: str, artifacts, *, weights_only: bool = False,
) -> None:
    # Push the engine's local artifact up to object storage and record its
    # key on the model so the download endpoints can stream it back.
    artifact_path = getattr(artifacts, "artifact_path", None)
    new_artifact_key: str | None = None
    new_artifact_sha: str | None = None
    new_artifact_size: int | None = None
    if artifact_path:
        suffix = Path(artifact_path).suffix
        new_artifact_key = make_artifact_key(job.model_id, suffix)
        # Hash + size the LOCAL file once (before upload) so the SDK /artifacts
        # endpoints can serve the ETag/size from metadata instead of re-reading
        # and re-hashing the whole object on every pull/304.
        import hashlib

        _h = hashlib.sha256()
        with open(artifact_path, "rb") as _fh:
            for _chunk in iter(lambda: _fh.read(1024 * 1024), b""):
                _h.update(_chunk)
        new_artifact_sha = _h.hexdigest()
        new_artifact_size = Path(artifact_path).stat().st_size
        get_storage().upload_file(artifact_path, new_artifact_key)

    # Persist the post-ingestion ONNX — the source graph that per-trial Pareto
    # exports re-apply compression configs onto.
    ingested_path = getattr(artifacts, "ingested_path", None)
    if ingested_path and Path(ingested_path).exists():
        get_storage().upload_file(ingested_path, ingested_key(job.model_id))

    with open_session() as s:
        put_cached_result(s, job.model_id, "architecture", artifacts.architecture,
                          user_id=job.user_id)
        # Pareto/benchmark exist only for executable models — a weights-only
        # checkpoint has nothing measurable here, so nothing is cached.
        if not weights_only:
            put_cached_result(s, job.model_id, "pareto", artifacts.pareto,
                              user_id=job.user_id)
            if artifacts.benchmark:
                put_cached_result(s, job.model_id, "benchmark", artifacts.benchmark,
                                  user_id=job.user_id)
            trial_configs = getattr(artifacts, "trial_configs", None)
            if trial_configs:
                put_cached_result(
                    s, job.model_id, "pareto_configs",
                    {"schema": 1, "trials": trial_configs},
                    user_id=job.user_id,
                )
            # Provenance of the served artifact — the SDK Hub labels which
            # candidate `artifact_key` actually holds (Pareto pick / ladder rung).
            artifact_meta = getattr(artifacts, "artifact_meta", None)
            if artifact_meta:
                put_cached_result(
                    s, job.model_id, "artifact_meta", artifact_meta,
                    user_id=job.user_id,
                )

        run = s.get(IngestionRunRow, job.run_id)
        if run:
            run.status = "completed"
            run.progress = 100
            run.finished_at = _now_iso()
            s.add(run)

        dash = s.get(RunRow, f"run_{job.run_id}")
        if dash:
            dash.status = "done"
            dash.progress_pct = 100
            total = dash.iter.split("/")[-1].strip()
            dash.iter = f"{total} / {total}"
            dash.best_acc = artifacts.best_accuracy or 0
            # Real accuracy delta: best trial vs the experiment baseline (pp).
            # Weights-only runs have no measurable accuracy → honest 0.
            base_acc = (artifacts.pareto or {}).get("baseAccuracy") if not weights_only else None
            dash.delta_acc = (
                round((artifacts.best_accuracy or 0) - base_acc, 2)
                if base_acc is not None else 0.0
            )
            s.add(dash)

        model = s.get(ModelRow, job.model_id)
        if model and new_artifact_key:
            # Record the artifact key regardless of the status guard so
            # downloads work even if /complete finalized the model first.
            model.artifact_key = new_artifact_key
            model.artifact_sha256 = new_artifact_sha
            model.artifact_size_bytes = new_artifact_size
            s.add(model)
        if model and model.status in ("analyzing", "optimizing"):
            # The worker is the single source of truth for finishing the
            # transition — the model must reach a terminal status even if
            # the SPA never calls /ingestion/complete (tab closed, crash).
            # Run-completed + model-draft commit atomically together so
            # the /complete endpoint can't observe a half-applied state.
            model.status = "draft"
            model.best_accuracy = artifacts.best_accuracy
            model.weights_only = weights_only
            model.last_optimized_at = _now_iso()
            model.analysis_run_id = None
            s.add(model)

        # Real warning alert from real measurements — DFCV risk rating or
        # a highly quantization-sensitive layer.
        risk = getattr(artifacts, "risk_level", "safe")
        max_sens = getattr(artifacts, "max_sensitivity", 0.0)
        if risk != "safe" or max_sens > 0.7:
            s.add(AlertRow(
                id=f"al_risk_{job.run_id}", user_id=job.user_id, model_id=job.model_id,
                level="warning",
                title=f"Compression risk — {model_name}",
                body=(f"DFCV risk={risk} · max layer sensitivity "
                      f"{max_sens:.2f} — review before deploying"),
                at=_now_iso(),
            ))

        s.add(ActivityRow(
            id=f"act_done_{job.run_id}", user_id=job.user_id, kind="run_completed",
            text=f"Run completed — run_{job.run_id} · {model_name}",
            timestamp=_now_iso(),
        ))
        s.commit()


def _on_failure(job: JobCtx, model_name: str, error: str) -> None:
    with open_session() as s:
        run = s.get(IngestionRunRow, job.run_id)
        if run:
            run.status = "failed"
            run.error = error[:500]
            run.finished_at = _now_iso()
            s.add(run)
        dash = s.get(RunRow, f"run_{job.run_id}")
        if dash:
            dash.status = "failed"
            s.add(dash)
        model = s.get(ModelRow, job.model_id)
        # Defense-in-depth: a deployed model never enters the pipeline today
        # (only uploads enqueue it, and there is no re-optimize path), so this
        # guard is currently unreachable — but it keeps "deployed" provably
        # immune to a silent → "failed" flip if re-optimization is ever added.
        if model and model.status != "deployed":
            model.status = "failed"
            model.analysis_run_id = None
            s.add(model)
        # Real danger alert — the pipeline genuinely failed.
        s.add(AlertRow(
            id=f"al_fail_{job.run_id}", user_id=job.user_id, model_id=job.model_id,
            level="danger",
            title=f"Pipeline failed — {model_name}",
            body=error[:200],
            at=_now_iso(),
        ))
        s.commit()


def latest_log_seq(run_id: str) -> int:
    with open_session() as s:
        rows = s.exec(
            select(IngestionLogRow.seq)
            .where(IngestionLogRow.run_id == run_id)
            .order_by(IngestionLogRow.seq.desc())  # type: ignore[attr-defined]
            .limit(1)
        ).first()
        return rows or 0


def reset_job_manager() -> None:
    """Back-compat shim for tests — inline dispatch keeps no global state to reset."""
    return None
