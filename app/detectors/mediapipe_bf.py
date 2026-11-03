"""BlazeFace via MediaPipe Tasks.

A partir do MediaPipe 1.0 a API legada `mp.solutions` deixou de existir, entao
usamos a API de Tasks.
"""

from __future__ import annotations

import threading

import numpy as np

from ..config import MODEL_CATALOG
from ..schemas import Detection, DetectorParam
from .base import BaseDetector


class MediaPipeBlazeFaceDetector(BaseDetector):
    key = "mediapipe_blazeface"
    label = "BlazeFace (MediaPipe)"
    family = "dnn"
    tier = 3
    description = (
        "Detector do Google usado em produtos de consumo com camera em tempo "
        "real. Foi desenhado para GPU movel, e uma referencia util do que roda "
        "hoje em um celular ou em uma camera inteligente."
    )
    requires_models = ("mediapipe_blazeface",)
    requires_packages = ("mediapipe",)

    def __init__(self) -> None:
        self._detector = None
        self._signature: tuple | None = None
        self._lock = threading.Lock()

    def params_spec(self) -> list[DetectorParam]:
        return [
            DetectorParam(
                key="score_threshold",
                label="Limiar de confianca",
                kind="float",
                default=0.5,
                minimum=0.01,
                maximum=0.99,
                step=0.01,
            ),
            DetectorParam(
                key="nms_threshold",
                label="Limiar de supressao",
                kind="float",
                default=0.3,
                minimum=0.1,
                maximum=0.9,
                step=0.05,
            ),
        ]

    def _load(self, score_threshold: float, nms_threshold: float):
        signature = (round(score_threshold, 4), round(nms_threshold, 4))
        if self._detector is not None and self._signature == signature:
            return self._detector
        with self._lock:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODEL_CATALOG["mediapipe_blazeface"].target)
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=score_threshold,
                min_suppression_threshold=nms_threshold,
            )
            if self._detector is not None:
                try:
                    self._detector.close()
                except Exception:  # noqa: BLE001 - fechar e best effort
                    pass
            self._detector = vision.FaceDetector.create_from_options(options)
            self._signature = signature
        return self._detector

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        import mediapipe as mp

        opts = self.resolved_params(params)
        detector = self._load(
            float(opts["score_threshold"]), float(opts["nms_threshold"])
        )

        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        img_h, img_w = image_bgr.shape[:2]
        detections: list[Detection] = []
        for item in result.detections:
            box = item.bounding_box
            score = float(item.categories[0].score) if item.categories else 0.0
            landmarks = None
            if getattr(item, "keypoints", None):
                landmarks = [
                    [float(k.x) * img_w, float(k.y) * img_h] for k in item.keypoints
                ]
            detections.append(
                Detection(
                    x=float(box.origin_x),
                    y=float(box.origin_y),
                    w=float(box.width),
                    h=float(box.height),
                    score=max(0.0, min(1.0, score)),
                    raw_score=score,
                    landmarks=landmarks,
                )
            )
        return detections
