"""YuNet - o detector DNN oficial do OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import MODEL_CATALOG
from ..schemas import Detection, DetectorParam
from .base import BaseDetector, dnn_params


class YuNetDetector(BaseDetector):
    key = "yunet"
    label = "YuNet"
    family = "dnn"
    tier = 1
    description = (
        "Detector CNN que o proprio OpenCV recomenda no lugar dos cascades "
        "desde a versao 4.5.4. Tem menos de 100 mil parametros e roda em "
        "milissegundos na CPU, o que o tornou a escolha natural para sistemas "
        "de video em tempo real sem GPU."
    )
    requires_models = ("yunet",)

    def params_spec(self) -> list[DetectorParam]:
        return dnn_params(
            default_score=0.5,
            default_nms=0.3,
            input_sizes=[
                {"value": "320", "label": "320 x 320"},
                {"value": "640", "label": "640 x 640"},
                {"value": "960", "label": "960 x 960"},
                {"value": "native", "label": "Resolucao original da imagem"},
            ],
            default_input="640",
        )

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        opts = self.resolved_params(params)
        model_path = str(MODEL_CATALOG["yunet"].target)

        img_h, img_w = image_bgr.shape[:2]
        if opts["input_size"] == "native":
            work = image_bgr
            scale_x = scale_y = 1.0
        else:
            side = int(opts["input_size"])
            work = cv2.resize(image_bgr, (side, side), interpolation=cv2.INTER_LINEAR)
            scale_x = img_w / side
            scale_y = img_h / side

        work_h, work_w = work.shape[:2]
        detector = cv2.FaceDetectorYN_create(
            model_path,
            "",
            (work_w, work_h),
            float(opts["score_threshold"]),
            float(opts["nms_threshold"]),
            int(opts["top_k"]),
        )
        detector.setInputSize((work_w, work_h))
        _retval, faces = detector.detect(work)

        detections: list[Detection] = []
        if faces is None:
            return detections

        for face in faces:
            x, y, w, h = (float(v) for v in face[:4])
            landmarks = [
                [float(face[4 + i * 2]) * scale_x, float(face[5 + i * 2]) * scale_y]
                for i in range(5)
            ]
            score = float(face[14])
            detections.append(
                Detection(
                    x=x * scale_x,
                    y=y * scale_y,
                    w=w * scale_x,
                    h=h * scale_y,
                    score=max(0.0, min(1.0, score)),
                    raw_score=score,
                    landmarks=landmarks,
                )
            )
        return detections
