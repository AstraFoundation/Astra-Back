"""File-name → frontend ModelFormat mapping (mirrors models/types.ts enum).

The format string is derived from the capability SSOT (`app/services/
capabilities.py`) so format inference and the per-format capability matrix can
never disagree.
"""

from __future__ import annotations

from app.services.capabilities import _DEFAULT_FORMAT, capability_for_filename


def is_executable(artifact_key: str | None) -> bool:
    """True only for ONNX artifacts — weights-only .npz can't be served on-device."""
    return bool(artifact_key) and artifact_key.endswith(".onnx")


def infer_format(file_name: str) -> tuple[str, str, str]:
    """Returns (format, typeFull, typeShort)."""
    cap = capability_for_filename(file_name)
    if cap is None:
        return _DEFAULT_FORMAT, "Imported Model", "Imported Model"
    label = f"{cap.format} Imported Model"
    return cap.format, label, label


def display_name(file_name: str) -> str:
    dot = file_name.rfind(".")
    base = file_name if dot == -1 else file_name[:dot]
    return base or "Uploaded model"
