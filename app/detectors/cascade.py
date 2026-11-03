"""Detectores baseados em cascade: Haar (Viola-Jones) e LBP.

O Haar e o algoritmo que o HyperFace ataca. A informacao que torna a avaliacao
possivel e o `levelWeight`: `detectMultiScale3` com `outputRejectLevels=True`
devolve, para cada caixa, a certeza acumulada no estagio final do cascade. Sem
isso todas as deteccoes pareceriam igualmente fortes e nao daria para dizer se
uma face falsa venceu a verdadeira.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import MODEL_CATALOG
from ..schemas import Detection, DetectorParam
from .base import BaseDetector, cascade_params, logistic


class _CascadeDetector(BaseDetector):
    profiles: list[dict] = []
    default_profile: str = ""
    _cache: dict[str, cv2.CascadeClassifier] = {}

    def params_spec(self) -> list[DetectorParam]:
        return cascade_params(self.profiles, self.default_profile)

    def _load(self, model_key: str) -> cv2.CascadeClassifier:
        cached = self._cache.get(model_key)
        if cached is not None:
            return cached
        spec = MODEL_CATALOG[model_key]
        if not spec.target.exists():
            raise FileNotFoundError(f"cascade ausente: {spec.target}")
        classifier = cv2.CascadeClassifier(str(spec.target))
        if classifier.empty():
            raise RuntimeError(f"cascade invalido: {spec.target}")
        self._cache[model_key] = classifier
        return classifier

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        opts = self.resolved_params(params)
        model_key = opts["profile"]
        if model_key not in self.requires_models:
            model_key = self.default_profile
        classifier = self._load(model_key)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if opts["equalize_hist"]:
            gray = cv2.equalizeHist(gray)

        min_size = int(opts["min_size"])
        max_size = int(opts["max_size"])
        kwargs = {
            "scaleFactor": float(opts["scale_factor"]),
            "minNeighbors": int(opts["min_neighbors"]),
            "minSize": (min_size, min_size),
        }
        if max_size > 0:
            kwargs["maxSize"] = (max_size, max_size)

        rects, _reject_levels, level_weights = classifier.detectMultiScale3(
            gray, outputRejectLevels=True, **kwargs
        )

        if len(rects) == 0:
            return []

        weights = np.asarray(level_weights, dtype=np.float32).reshape(-1)

        neighbors: np.ndarray | None = None
        if opts["count_neighbors"]:
            try:
                rects2, num_detections = classifier.detectMultiScale2(gray, **kwargs)
                if len(rects2) == len(rects):
                    neighbors = np.asarray(num_detections, dtype=np.int32).reshape(-1)
            except cv2.error:
                neighbors = None

        detections: list[Detection] = []
        for i, (x, y, w, h) in enumerate(rects):
            raw = float(weights[i]) if i < len(weights) else 0.0
            detections.append(
                Detection(
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                    score=logistic(raw),
                    raw_score=raw,
                    neighbors=(
                        int(neighbors[i])
                        if neighbors is not None and i < len(neighbors)
                        else None
                    ),
                )
            )
        return detections


class HaarDetector(_CascadeDetector):
    key = "haar"
    label = "Haar Cascade (Viola-Jones)"
    family = "cascade"
    tier = 1
    description = (
        "O detector de 2001 que o HyperFace ataca. Foi o padrao universal em "
        "aplicacoes de visao computacional por mais de uma decada, e como "
        "praticamente todo mundo usava os mesmos arquivos XML, quebrar um "
        "perfil significava quebrar o detector em todo lugar."
    )
    requires_models = (
        "haar_frontal_default",
        "haar_frontal_alt",
        "haar_frontal_alt2",
        "haar_frontal_alt_tree",
        "haar_profile",
    )
    default_profile = "haar_frontal_default"
    profiles = [
        {"value": "haar_frontal_default", "label": "frontalface_default (alvo do HyperFace)"},
        {"value": "haar_frontal_alt", "label": "frontalface_alt"},
        {"value": "haar_frontal_alt2", "label": "frontalface_alt2"},
        {"value": "haar_frontal_alt_tree", "label": "frontalface_alt_tree"},
        {"value": "haar_profile", "label": "profileface"},
    ]
    _cache: dict[str, cv2.CascadeClassifier] = {}


class LbpDetector(_CascadeDetector):
    key = "lbp"
    label = "LBP Cascade"
    family = "cascade"
    tier = 1
    description = (
        "Mesma arquitetura de cascade do Haar, mas com features de Local Binary "
        "Patterns. E varias vezes mais rapido e por isso aparecia muito em "
        "hardware embarcado e cameras de baixo custo."
    )
    requires_models = ("lbp_frontal_improved", "lbp_frontal", "lbp_profile")
    default_profile = "lbp_frontal_improved"
    profiles = [
        {"value": "lbp_frontal_improved", "label": "frontalface_improved"},
        {"value": "lbp_frontal", "label": "frontalface (original)"},
        {"value": "lbp_profile", "label": "profileface"},
    ]
    _cache: dict[str, cv2.CascadeClassifier] = {}
