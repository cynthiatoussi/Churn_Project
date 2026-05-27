"""
services/monitoring/app.py
Service de monitoring — enregistre les requêtes et expose les métriques.

Endpoints :
    POST /log      ← appelé par inference après chaque prédiction
    GET  /metrics  ← consulté pendant le stress test
    GET  /health   ← health check Kubernetes
"""

import threading
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Monitoring Service")

# ─────────────────────────────────────────────────────────────
# Stockage en mémoire (thread-safe)
# ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "total_requests":    0,
    "success_count":     0,
    "error_count":       0,
    "latencies":         [],
    "churn_probs":       [],
    "offers":            {},
}


# ─────────────────────────────────────────────────────────────
# Schémas
# ─────────────────────────────────────────────────────────────
class LogEntry(BaseModel):
    status_code:        int
    latency_s:          float
    churn_probability:  Optional[float] = None
    recommended_offer:  Optional[str]   = None
    error:              Optional[str]   = None


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/log")
def log_prediction(entry: LogEntry):
    """Reçoit les métriques d'une prédiction et les stocke."""
    with _lock:
        _state["total_requests"] += 1
        _state["latencies"].append(entry.latency_s)

        if entry.status_code == 200:
            _state["success_count"] += 1
        else:
            _state["error_count"] += 1

        if entry.churn_probability is not None:
            _state["churn_probs"].append(entry.churn_probability)

        if entry.recommended_offer:
            _state["offers"][entry.recommended_offer] = (
                _state["offers"].get(entry.recommended_offer, 0) + 1
            )

    return {"status": "logged"}


@app.get("/metrics")
def get_metrics():
    """Expose les métriques agrégées — consulté pendant le stress test."""
    with _lock:
        n         = _state["total_requests"]
        latencies = _state["latencies"]
        probs     = _state["churn_probs"]

        sorted_lat = sorted(latencies) if latencies else [0.0]
        p95_idx    = max(0, int(0.95 * len(sorted_lat)) - 1)

        return {
            "total_requests":       n,
            "success_count":        _state["success_count"],
            "error_count":          _state["error_count"],
            "success_rate_pct":     round(_state["success_count"] / n * 100, 1) if n else 0,
            "avg_latency_s":        round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "p95_latency_s":        round(sorted_lat[p95_idx], 3),
            "max_latency_s":        round(sorted_lat[-1], 3) if sorted_lat else 0,
            "avg_churn_probability": round(sum(probs) / len(probs), 3) if probs else 0,
            "offer_distribution":   dict(_state["offers"]),
        }


@app.post("/reset")
def reset_metrics():
    """Remet les compteurs à zéro entre deux tests."""
    with _lock:
        _state["total_requests"] = 0
        _state["success_count"]  = 0
        _state["error_count"]    = 0
        _state["latencies"]      = []
        _state["churn_probs"]    = []
        _state["offers"]         = {}
    return {"status": "reset"}


@app.get("/health")
def health():
    """Health check pour Kubernetes."""
    return {"status": "ok", "total_requests": _state["total_requests"]}