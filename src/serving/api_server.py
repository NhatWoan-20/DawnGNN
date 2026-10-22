"""
api_server.py — FastAPI REST server for DawnGNN Windows malware detection.

Endpoints:
    GET  /health              — Liveness and readiness health probe
    GET  /model/info          — Model architecture metadata and runtime specs
    POST /predict             — Single-sample API sequence classification
    POST /predict/batch       — Batch sequence classification (up to 64 samples)

Run:
    uvicorn dawngnn.serving.api_server:app --host 0.0.0.0 --port 8000 --reload
    # Or:
    uvicorn src.serving.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


try:
    from dawngnn.serving.infer import (
        GATClassifier,
        load_api_embeddings,
        load_config,
        load_model,
        predict,
    )
    from dawngnn.utils.logger import setup_logger
    from dawngnn.utils.paths import get_config_path, get_data_path, get_model_path, get_project_root
except ImportError:
    import sys
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from serving.infer import (
        GATClassifier,
        load_api_embeddings,
        load_config,
        load_model,
        predict,
    )
    from utils.logger import setup_logger
    from utils.paths import get_config_path, get_data_path, get_model_path, get_project_root

logger = setup_logger("api_server")
PROJECT_ROOT = get_project_root()

# ---------------------------------------------------------------------------
# FastAPI (imported here so the file can be imported without the dep installed)
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    try:
        from pydantic import field_validator
    except ImportError:
        from pydantic import validator as field_validator  # type: ignore[assignment]
except ImportError as e:
    raise ImportError(
        "FastAPI is required to run the API server.\n"
        "Install it with: pip install fastapi uvicorn"
    ) from e

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Single-sample prediction request."""

    api_sequence: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of Windows API call names.",
        json_schema_extra={"example": ["NtCreateFile", "NtWriteFile", "NtReadFile", "NtClose"]},
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of top API nodes to include in explanation.")

    @field_validator("api_sequence")
    @classmethod
    def strip_whitespace(cls, v: List[str]) -> List[str]:
        cleaned = [item.strip() for item in v if item.strip()]
        if not cleaned:
            raise ValueError("api_sequence must contain at least one non-empty API name.")
        return cleaned


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    samples: List[PredictRequest] = Field(
        ..., min_length=1, max_length=64, description="List of API sequences (max 64 per call)."
    )


class GraphStats(BaseModel):
    nodes: int
    edges: int


class TopAPI(BaseModel):
    api: str
    degree: int


class PredictResponse(BaseModel):
    label: int
    label_name: str
    malware_probability: float
    benign_probability: float
    unique_apis: int
    total_calls: int
    graph_stats: GraphStats
    top_apis: List[TopAPI]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_latency_ms: float


class ModelInfo(BaseModel):
    model_type: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    input_dim: int
    embedding_model: str
    device: str
    model_path: str
    api_embeddings_loaded: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# App factory & global state
# ---------------------------------------------------------------------------
_state: Dict = {
    "model": None,
    "api_embeddings": None,
    "config": None,
    "device": None,
    "model_path": None,
    "start_time": time.time(),
    "bert_model_name": "bert-base-uncased",
}


def _load_resources():
    """Load model weights and API embeddings into module-level state."""
    config_path = get_config_path("gnn.yaml")
    config = load_config(config_path)

    # BERT config
    try:
        bert_cfg_path = get_config_path("bert.yaml")
        with open(bert_cfg_path, "r") as f:
            bert_cfg = yaml.safe_load(f) or {}
        _state["bert_model_name"] = bert_cfg.get("model", {}).get("name", "bert-base-uncased")
    except FileNotFoundError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Embeddings
    embeddings_dir = PROJECT_ROOT / config["paths"]["embeddings_dir"]
    default_size = config.get("embedding", {}).get("default_size", 768)
    api_embeddings = load_api_embeddings(embeddings_dir, default_size)

    # Model
    model_path_env = os.environ.get("DAWNGNN_MODEL_PATH")
    model_path = Path(model_path_env) if model_path_env else get_model_path("gnn", "best_model.pt")

    if not model_path.exists():
        logger.error(
            f"Model checkpoint not found at {model_path}. "
            "Run training first or set DAWNGNN_MODEL_PATH env variable."
        )
        model = None
    else:
        model = load_model(model_path, config, device)

    _state.update(
        {
            "model": model,
            "api_embeddings": api_embeddings,
            "config": config,
            "device": device,
            "model_path": str(model_path),
        }
    )
    logger.info("Resources loaded successfully.")


# ---------------------------------------------------------------------------
# FastAPI app & lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DawnGNN API server …")
    _load_resources()
    logger.info("API server ready.")
    yield


app = FastAPI(
    title="DawnGNN — Windows Malware Detection API",
    description=(
        "REST API for detecting Windows malware via Graph Neural Networks augmented with "
        "BERT semantic embeddings of Windows API documentation.\n\n"
        "Based on the paper: *DawnGNN: Documentation Augmented Windows Malware Detection "
        "Using Graph Neural Network*."
    ),
    version="1.0.0",
    contact={"name": "DawnGNN Authors", "url": "https://github.com/NhatWoan-20/DawnGNN"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok" if _state["model"] is not None else "degraded",
        model_loaded=_state["model"] is not None,
        device=str(_state["device"]),
        uptime_seconds=round(time.time() - _state["start_time"], 2),
    )


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
async def model_info():
    """Return model architecture metadata and runtime information."""
    if _state["config"] is None:
        raise HTTPException(status_code=503, detail="Server not fully initialized.")
    mc = _state["config"]["model"]
    return ModelInfo(
        model_type="GATClassifier (Graph Attention Network)",
        hidden_dim=mc["hidden_dim"],
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        dropout=mc["dropout"],
        input_dim=mc["input_dim"],
        embedding_model=_state["bert_model_name"],
        device=str(_state["device"]),
        model_path=_state["model_path"] or "not loaded",
        api_embeddings_loaded=len(_state["api_embeddings"] or {}),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_single(request: PredictRequest):
    """
    Classify a single Windows API call sequence as **Malware** or **Benign**.

    The sequence is converted into a directed weighted graph, semantically
    augmented with BERT embeddings of Windows API documentation, and then
    classified by the trained GAT model.
    """
    _require_model()
    t0 = time.perf_counter()

    try:
        result = predict(
            api_sequence=request.api_sequence,
            model=_state["model"],
            api_embeddings=_state["api_embeddings"],
            device=_state["device"],
            config=_state["config"],
            top_k_nodes=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        **result,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    Classify a batch of API call sequences (max 64 per request).

    Each sample is processed independently; results are returned in the same
    order as the input list.
    """
    _require_model()
    t0 = time.perf_counter()
    results = []

    for i, sample in enumerate(request.samples):
        try:
            result = predict(
                api_sequence=sample.api_sequence,
                model=_state["model"],
                api_embeddings=_state["api_embeddings"],
                device=_state["device"],
                config=_state["config"],
                top_k_nodes=sample.top_k,
            )
            results.append(
                PredictResponse(
                    **result,
                    latency_ms=0.0,  # filled below for per-sample timing if needed
                )
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Sample {i}: {e}")
        except Exception as e:
            logger.exception(f"Error processing sample {i}")
            raise HTTPException(status_code=500, detail=f"Sample {i} failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(results=results, total_latency_ms=round(total_ms, 2))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception for {request.method} {request.url}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_model():
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs and ensure training has been completed.",
        )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "dawngnn.serving.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
