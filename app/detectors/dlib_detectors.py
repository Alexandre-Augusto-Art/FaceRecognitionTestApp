"""Detectores do dlib: HOG + SVM e a CNN MMOD.

O HOG/SVM importa aqui porque Adam Harvey cita explicitamente os detectores
"HoG/SVM-based edge" como o segundo alvo possivel de um padrao HyperFace,
ao lado do Haar.
"""

from __future__ import annotations

import threading

import numpy as np

from ..config import MODEL_CATALOG
from ..schemas import Detection, DetectorParam
from .base import BaseDetector, logistic


class DlibHogDetector(BaseDetector):
    key = "dlib_hog"
    label = "dlib HOG + SVM"
    family = "classic"
    tier = 3
    description = (
        "Histogram of Oriented Gradients com classificador linear. Foi o "
        "detector padrao da era pre-deep-learning e ainda aparece em muitas "
        "bibliotecas, incluindo o popular pacote face_recognition."
    )
    requires_packages = ("dlib",)

    def __init__(self) -> None:
        self._detector = None
        self._lock = threading.Lock()

    def params_spec(self) -> list[DetectorParam]:
        return [
            DetectorParam(
                key="upsample",
                label="Upsample",
                kind="int",
                default=1,
                minimum=0,
                maximum=3,
                step=1,
                help=(
                    "Quantas vezes a imagem e ampliada antes da varredura. "
                    "Cada nivel encontra faces menores e custa 4x mais tempo."
                ),
            ),
            DetectorParam(
                key="adjust_threshold",
                label="Ajuste do limiar",
                kind="float",
                default=0.0,
                minimum=-1.0,
                maximum=1.0,
                step=0.05,
                help=(
                    "Valores negativos deixam o detector mais permissivo e "
                    "revelam ativacoes fracas que o padrao provoca."
                ),
            ),
        ]

    def _load(self):
        if self._detector is not None:
            return self._detector
        with self._lock:
            if self._detector is None:
                import dlib

                self._detector = dlib.get_frontal_face_detector()
        return self._detector

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        opts = self.resolved_params(params)
        detector = self._load()
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])

        boxes, scores, _idx = detector.run(
            rgb, int(opts["upsample"]), float(opts["adjust_threshold"])
        )

        detections: list[Detection] = []
        for box, raw in zip(boxes, scores):
            raw = float(raw)
            detections.append(
                Detection(
                    x=float(box.left()),
                    y=float(box.top()),
                    w=float(box.width()),
                    h=float(box.height()),
                    score=logistic(raw),
                    raw_score=raw,
                )
            )
        return detections


class DlibCnnDetector(BaseDetector):
    key = "dlib_cnn"
    label = "dlib CNN (MMOD)"
    family = "dnn"
    tier = 3
    description = (
        "Detector CNN treinado com Max-Margin Object Detection. Bem mais "
        "robusto que o HOG a rotacao e iluminacao, ao custo de ser lento na CPU."
    )
    requires_models = ("dlib_mmod",)
    requires_packages = ("dlib",)

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def params_spec(self) -> list[DetectorParam]:
        return [
            DetectorParam(
                key="upsample",
                label="Upsample",
                kind="int",
                default=0,
                minimum=0,
                maximum=2,
                step=1,
                help="Cuidado: cada nivel multiplica muito o tempo na CPU.",
            ),
            DetectorParam(
                key="score_threshold",
                label="Limiar de confianca",
                kind="float",
                default=0.0,
                minimum=-1.0,
                maximum=2.0,
                step=0.05,
                help="A MMOD devolve confianca sem limite fixo; 0 e o corte natural.",
            ),
            DetectorParam(
                key="max_side",
                label="Lado maximo (px)",
                kind="int",
                default=1024,
                minimum=256,
                maximum=2048,
                step=64,
                help="A imagem e reduzida antes da inferencia para nao travar a CPU.",
            ),
        ]

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                import dlib

                self._model = dlib.cnn_face_detection_model_v1(
                    str(MODEL_CATALOG["dlib_mmod"].target)
                )
        return self._model

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        import cv2

        opts = self.resolved_params(params)
        model = self._load()

        img_h, img_w = image_bgr.shape[:2]
        max_side = int(opts["max_side"])
        scale = 1.0
        work = image_bgr
        if max(img_h, img_w) > max_side:
            scale = max_side / max(img_h, img_w)
            work = cv2.resize(
                image_bgr,
                (int(img_w * scale), int(img_h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        rgb = np.ascontiguousarray(work[:, :, ::-1])
        results = model(rgb, int(opts["upsample"]))

        threshold = float(opts["score_threshold"])
        detections: list[Detection] = []
        for item in results:
            raw = float(item.confidence)
            if raw < threshold:
                continue
            rect = item.rect
            detections.append(
                Detection(
                    x=float(rect.left()) / scale,
                    y=float(rect.top()) / scale,
                    w=float(rect.width()) / scale,
                    h=float(rect.height()) / scale,
                    score=logistic(raw),
                    raw_score=raw,
                )
            )
        return detections
