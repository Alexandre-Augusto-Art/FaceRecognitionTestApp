"""Servidor local do testbed."""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import detectors
from .config import (
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    MODEL_CATALOG,
    MODELS_DIR,
    PRESETS,
    RESULTS_DIR,
    STATIC_DIR,
    UPLOADS_DIR,
    ensure_dirs,
)
from .downloader import fetch_all, is_present
from .metrics import evaluate_all
from .schemas import (
    DetectRequest,
    DetectResponse,
    ImageInfo,
    MetricsRequest,
)

ensure_dirs()

app = FastAPI(title="HyperFace Detection Testbed", version="1.0.0")

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

_download_state: dict = {"running": False, "report": {}, "started_at": None}
_download_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_path(image_id: str) -> Path:
    if not _ID_PATTERN.match(image_id):
        raise HTTPException(status_code=400, detail="Identificador de imagem invalido.")
    matches = sorted(UPLOADS_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Imagem nao encontrada.")
    return matches[0]


def _load_image(image_id: str) -> np.ndarray:
    path = _image_path(image_id)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Nao foi possivel decodificar a imagem.")
    return image


# ---------------------------------------------------------------------------
# Metadados
# ---------------------------------------------------------------------------


@app.get("/api/detectors")
def list_detectors() -> dict:
    return {
        "detectors": [info.model_dump() for info in detectors.describe_all()],
        "presets": [
            {
                "key": preset.key,
                "label": preset.label,
                "description": preset.description,
                "params": preset.params,
            }
            for preset in PRESETS
        ],
    }


@app.get("/api/models")
def list_models() -> dict:
    items = []
    for spec in MODEL_CATALOG.values():
        present = is_present(spec)
        items.append(
            {
                "key": spec.key,
                "description": spec.description,
                "approx_mb": spec.approx_mb,
                "present": present,
                "size_mb": (
                    round(spec.target.stat().st_size / 1048576, 2) if present else None
                ),
            }
        )
    missing = [i for i in items if not i["present"]]
    return {
        "models_dir": str(MODELS_DIR),
        "models": items,
        "missing_count": len(missing),
        "missing_mb": round(sum(i["approx_mb"] for i in missing), 1),
        "download": _download_state,
    }


def _download_worker(keys: list[str]) -> None:
    try:
        report = fetch_all(keys, progress=False)
    except Exception as exc:  # noqa: BLE001 - o estado precisa refletir a falha
        report = {"erro": str(exc)}
    with _download_lock:
        _download_state["running"] = False
        _download_state["report"] = report


@app.post("/api/models/download")
def download_models() -> dict:
    with _download_lock:
        if _download_state["running"]:
            return {"started": False, "reason": "Ja existe um download em andamento."}
        missing = [k for k, spec in MODEL_CATALOG.items() if not is_present(spec)]
        if not missing:
            return {"started": False, "reason": "Todos os modelos ja estao presentes."}
        _download_state["running"] = True
        _download_state["report"] = {}
        _download_state["started_at"] = datetime.now().isoformat(timespec="seconds")

    thread = threading.Thread(target=_download_worker, args=(missing,), daemon=True)
    thread.start()
    return {"started": True, "keys": missing}


# ---------------------------------------------------------------------------
# Imagens
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> ImageInfo:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato nao suportado ({suffix or 'sem extensao'}).",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo acima do limite de {MAX_UPLOAD_BYTES // 1048576} MB.",
        )

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Nao foi possivel decodificar a imagem.")

    height, width = decoded.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Imagem de {width}x{height} excede o lado maximo de "
                f"{MAX_IMAGE_SIDE}px. Redimensione antes de enviar."
            ),
        )

    image_id = uuid.uuid4().hex
    destination = UPLOADS_DIR / f"{image_id}{suffix}"
    destination.write_bytes(payload)

    return ImageInfo(
        image_id=image_id,
        filename=file.filename or destination.name,
        width=width,
        height=height,
    )


@app.get("/api/image/{image_id}")
def get_image(image_id: str) -> FileResponse:
    return FileResponse(_image_path(image_id))


# ---------------------------------------------------------------------------
# Deteccao e metricas
# ---------------------------------------------------------------------------


@app.post("/api/detect")
def detect(request: DetectRequest) -> DetectResponse:
    if not request.detectors:
        raise HTTPException(status_code=400, detail="Nenhum detector selecionado.")

    image = _load_image(request.image_id)
    height, width = image.shape[:2]

    # Sequencial de proposito: rodar em paralelo distorceria o tempo medido de
    # cada detector, que e um dado que interessa na comparacao.
    results = [
        detectors.run_one(key, image, request.params.get(key, {}))
        for key in request.detectors
    ]

    return DetectResponse(
        image=ImageInfo(
            image_id=request.image_id,
            filename=_image_path(request.image_id).name,
            width=width,
            height=height,
        ),
        results=results,
    )


@app.post("/api/metrics")
def compute_metrics(request: MetricsRequest) -> dict:
    metrics = evaluate_all(
        request.results, request.ground_truth, request.iou_threshold
    )
    return {"metrics": [m.model_dump() for m in metrics]}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@app.post("/api/export")
def export(payload: dict) -> dict:
    image_id = payload.get("image_id")
    if not image_id:
        raise HTTPException(status_code=400, detail="image_id ausente.")
    _image_path(image_id)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(payload.get("label") or "run"))[:40]
    folder = RESULTS_DIR / f"{stamp}_{label}"
    folder.mkdir(parents=True, exist_ok=True)

    record = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "image": payload.get("image"),
        "ground_truth": payload.get("ground_truth", []),
        "iou_threshold": payload.get("iou_threshold"),
        "results": payload.get("results", []),
        "metrics": payload.get("metrics", []),
    }
    json_path = folder / "resultado.json"
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    written = [str(json_path)]
    data_url = payload.get("annotated_png")
    if isinstance(data_url, str) and "," in data_url:
        try:
            raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"PNG anotado invalido: {exc}"
            ) from exc
        png_path = folder / "anotado.png"
        png_path.write_bytes(raw)
        written.append(str(png_path))

    return {"folder": str(folder), "files": written}


@app.get("/api/results")
def list_results() -> dict:
    runs = []
    for folder in sorted(RESULTS_DIR.iterdir(), reverse=True) if RESULTS_DIR.exists() else []:
        if not folder.is_dir():
            continue
        json_path = folder / "resultado.json"
        if not json_path.exists():
            continue
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runs.append(
            {
                "name": folder.name,
                "exported_at": record.get("exported_at"),
                "image": record.get("image"),
                "metrics": record.get("metrics", []),
            }
        )
    return {"results": runs[:100]}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(HTTPException)
def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
