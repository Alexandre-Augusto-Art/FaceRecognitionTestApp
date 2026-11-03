"""UltraFace RFB - detector ultraleve, comum em camera embarcada."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from ..config import MODEL_CATALOG
from ..schemas import Detection, DetectorParam
from .base import BaseDetector, dnn_params, nms

# Cada variante tem a resolucao fixa embutida no grafo ONNX.
_VARIANTS = {
    "ultraface_320": (320, 240),
    "ultraface_640": (640, 480),
}


class UltraFaceDetector(BaseDetector):
    key = "ultraface"
    label = "UltraFace RFB"
    family = "dnn"
    tier = 2
    description = (
        "Rede de deteccao facial de ~1 MB projetada para dispositivos de borda. "
        "Representa a classe de detectores que roda dentro da propria camera, "
        "onde nao ha orcamento computacional para um SCRFD."
    )
    requires_models = ("ultraface_320", "ultraface_640")
    requires_packages = ("onnxruntime",)

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}
        self._lock = threading.Lock()

    def params_spec(self) -> list[DetectorParam]:
        params = dnn_params(
            default_score=0.5,
            default_nms=0.4,
            input_sizes=[
                {"value": "ultraface_320", "label": "RFB-320 (320 x 240)"},
                {"value": "ultraface_640", "label": "RFB-640 (640 x 480)"},
            ],
            default_input="ultraface_640",
        )
        for param in params:
            if param.key == "input_size":
                param.label = "Variante do modelo"
        return params

    def _session(self, variant: str):
        cached = self._sessions.get(variant)
        if cached is not None:
            return cached
        with self._lock:
            if variant in self._sessions:
                return self._sessions[variant]
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            session = onnxruntime.InferenceSession(
                str(MODEL_CATALOG[variant].target),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._sessions[variant] = session
        return session

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        opts = self.resolved_params(params)
        variant = str(opts["input_size"])
        if variant not in _VARIANTS:
            variant = "ultraface_640"
        net_w, net_h = _VARIANTS[variant]
        session = self._session(variant)

        img_h, img_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32)
        blob = ((rgb - 127.0) / 128.0).transpose(2, 0, 1)[None]

        input_name = session.get_inputs()[0].name
        confidences, boxes = session.run(None, {input_name: blob})

        # A rede devolve duas colunas de score (fundo, face) e caixas ja
        # normalizadas em 0..1.
        scores = confidences[0][:, 1]
        boxes = boxes[0]

        threshold = float(opts["score_threshold"])
        keep_idx = np.where(scores >= threshold)[0]
        if keep_idx.size == 0:
            return []

        scores = scores[keep_idx]
        boxes = boxes[keep_idx] * np.array(
            [img_w, img_h, img_w, img_h], dtype=np.float32
        )

        keep = nms(boxes, scores, float(opts["nms_threshold"]))
        keep = keep[: int(opts["top_k"])]

        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            score = float(scores[i])
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=max(0.0, x2 - x1),
                    h=max(0.0, y2 - y1),
                    score=max(0.0, min(1.0, score)),
                    raw_score=score,
                )
            )
        return detections
