"""Metric attestation for ML model promotion (audit fix H1).

Promotion gates must never trust mutable metric blobs on their own. At
registration the trainer commits a digest over the metric payload; at
promotion the registry recomputes it and rejects any mismatch, so tampering
with the store, the artifact, or the digest itself is detected.

Digest = HMAC-SHA256(key, canonical_json) when METRIC_SIGNING_KEY is present
(process environment first, then the Settings field so a .env-only value is
honored); otherwise a plain SHA-256 integrity digest (detects accidental
corruption but not a determined attacker — set the key in production).
"""
import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional


def _signing_key() -> bytes:
    """Signing key precedence: process environment first (explicit override),
    then the Settings field — pydantic-settings loads .env into Settings but
    does NOT inject it into os.environ, so a .env-only key would otherwise be
    silently ignored and the digests would silently degrade to plain SHA-256
    (corruption detection, not tamper-evidence)."""
    key = os.environ.get("METRIC_SIGNING_KEY", "").strip()
    if not key:
        try:
            from backend.app.core.config import settings

            key = (getattr(settings, "METRIC_SIGNING_KEY", "") or "").strip()
        except Exception:
            key = ""  # config not importable; keep the degraded-but-working path
    if key:
        return key.encode("utf-8")
    return b""


def _canonical(model_id: str, metrics_map: Dict[str, Any]) -> bytes:
    payload: Dict[str, Any] = {"model_id": model_id}
    for name in sorted(metrics_map):
        payload[name] = metrics_map[name]
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def metric_digest(
    model_id: str,
    validation_metrics: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
    walk_forward_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    message = _canonical(
        model_id,
        {
            "test_metrics": test_metrics or {},
            "validation_metrics": validation_metrics or {},
            "walk_forward_metrics": walk_forward_metrics or {},
        },
    )
    key = _signing_key()
    if key:
        return hmac.new(key, message, hashlib.sha256).hexdigest()
    return hashlib.sha256(message).hexdigest()


def verify_metric_digest(
    model_id: str,
    validation_metrics: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
    walk_forward_metrics: Optional[Dict[str, Any]],
    digest: Optional[str],
) -> bool:
    if not digest:
        return False
    return hmac.compare_digest(
        metric_digest(model_id, validation_metrics, test_metrics, walk_forward_metrics),
        digest,
    )


def ml_walk_forward_digest(model_id: str, payload: Optional[Dict[str, Any]]) -> str:
    """Attestation digest over the ML walk-forward evidence payload (M8).

    The payload is stored next to this digest on the model record; the same
    trust model as metric_digest applies — an HMAC key (METRIC_SIGNING_KEY)
    makes tampering evident, a missing key degrades to corruption detection.
    """
    message = _canonical(
        model_id,
        {"ml_walk_forward_metrics": payload or {}},
    )
    key = _signing_key()
    if key:
        return hmac.new(key, message, hashlib.sha256).hexdigest()
    return hashlib.sha256(message).hexdigest()


def verify_ml_walk_forward_digest(
    model_id: str,
    payload: Optional[Dict[str, Any]],
    digest: Optional[str],
) -> bool:
    if not digest:
        return False
    return hmac.compare_digest(
        ml_walk_forward_digest(model_id, payload),
        digest,
    )
