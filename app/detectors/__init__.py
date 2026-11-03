"""Registry de detectores.

Um detector que nao consegue carregar nunca derruba o app: ele e listado como
indisponivel, com o motivo, e os demais continuam funcionando.
"""

from __future__ import annotations

import time

import numpy as np

from ..schemas import DetectorInfo, DetectorResult
from .base import BaseDetector
from .cascade import HaarDetector, LbpDetector
from .dlib_detectors import DlibCnnDetector, DlibHogDetector
from .mediapipe_bf import MediaPipeBlazeFaceDetector
from .scrfd import Scrfd10GDetector, Scrfd500MDetector
from .ultraface import UltraFaceDetector
from .yunet import YuNetDetector

_DETECTOR_CLASSES = [
    HaarDetector,
    LbpDetector,
    YuNetDetector,
    Scrfd10GDetector,
    Scrfd500MDetector,
    UltraFaceDetector,
    DlibHogDetector,
    DlibCnnDetector,
    MediaPipeBlazeFaceDetector,
]

_REGISTRY: dict[str, BaseDetector] = {}
for cls in _DETECTOR_CLASSES:
    instance = cls()
    _REGISTRY[instance.key] = instance


def all_detectors() -> list[BaseDetector]:
    return list(_REGISTRY.values())


def get(key: str) -> BaseDetector | None:
    return _REGISTRY.get(key)


def describe_all() -> list[DetectorInfo]:
    infos: list[DetectorInfo] = []
    for detector in _REGISTRY.values():
        available, reason = detector.availability()
        infos.append(
            DetectorInfo(
                key=detector.key,
                label=detector.label,
                family=detector.family,
                tier=detector.tier,
                color=detector.color,
                available=available,
                unavailable_reason=reason,
                description=detector.description,
                params=detector.params_spec(),
            )
        )
    return infos


def run_one(
    key: str, image_bgr: np.ndarray, params: dict | None = None
) -> DetectorResult:
    detector = _REGISTRY.get(key)
    if detector is None:
        return DetectorResult(
            detector=key,
            label=key,
            color="#8e8e93",
            ok=False,
            error="Detector desconhecido.",
        )

    available, reason = detector.availability()
    if not available:
        return DetectorResult(
            detector=detector.key,
            label=detector.label,
            color=detector.color,
            ok=False,
            error=reason,
        )

    resolved = detector.resolved_params(params)
    started = time.perf_counter()
    try:
        detections = detector.detect(image_bgr, params or {})
    except Exception as exc:  # noqa: BLE001 - a UI mostra o erro por detector
        return DetectorResult(
            detector=detector.key,
            label=detector.label,
            color=detector.color,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            params_used=resolved,
        )

    detections.sort(key=lambda d: d.score, reverse=True)
    return DetectorResult(
        detector=detector.key,
        label=detector.label,
        color=detector.color,
        ok=True,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        detections=detections,
        params_used=resolved,
    )
