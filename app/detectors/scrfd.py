"""SCRFD - o detector do InsightFace, rodando em onnxruntime puro.

O pre e pos-processamento estao implementados aqui em vez de usar o pacote
`insightface` por dois motivos: o pacote arrasta muitas dependencias pesadas e
ainda nao tem suporte firme a Python 3.14, e, mais importante, fazendo na mao
temos acesso direto ao score bruto de cada ancora, que e exatamente o numero
que as metricas do HyperFace precisam.

SCRFD e RetinaFace sao os detectores livres mais proximos do que rodam os
sistemas comerciais de vigilancia. O pacote buffalo_l do InsightFace usa o
SCRFD-10GF, que e o modelo carregado aqui.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from ..config import MODEL_CATALOG
from ..schemas import Detection, DetectorParam
from .base import BaseDetector, distance2bbox, distance2kps, dnn_params, nms


class _ScrfdBase(BaseDetector):
    family = "dnn"
    tier = 2
    requires_packages = ("onnxruntime",)
    model_key = ""

    def __init__(self) -> None:
        self._session = None
        self._lock = threading.Lock()
        self._input_name = ""
        self._output_names: list[str] = []
        self._fmc = 3
        self._strides = [8, 16, 32]
        self._num_anchors = 2
        self._use_kps = True

    def params_spec(self) -> list[DetectorParam]:
        return dnn_params(
            default_score=0.5,
            default_nms=0.4,
            input_sizes=[
                {"value": "320", "label": "320 x 320"},
                {"value": "640", "label": "640 x 640 (padrao do InsightFace)"},
                {"value": "960", "label": "960 x 960"},
                {"value": "1280", "label": "1280 x 1280"},
            ],
            default_input="640",
        )

    # ------------------------------------------------------------------ ONNX

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            import onnxruntime

            path = str(MODEL_CATALOG[self.model_key].target)
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            session = onnxruntime.InferenceSession(
                path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._input_name = session.get_inputs()[0].name
            self._output_names = [o.name for o in session.get_outputs()]
            self._configure_from_outputs(len(self._output_names))
            self._session = session
        return self._session

    def _configure_from_outputs(self, count: int) -> None:
        """Descobre a topologia do modelo pelo numero de saidas.

        As variantes do SCRFD diferem em quantos niveis de piramide usam, se
        preveem landmarks e quantas ancoras existem por posicao. O numero de
        tensores de saida identifica a combinacao sem ambiguidade.
        """
        if count == 6:
            self._fmc, self._strides, self._num_anchors, self._use_kps = (
                3,
                [8, 16, 32],
                2,
                False,
            )
        elif count == 9:
            self._fmc, self._strides, self._num_anchors, self._use_kps = (
                3,
                [8, 16, 32],
                2,
                True,
            )
        elif count == 10:
            self._fmc, self._strides, self._num_anchors, self._use_kps = (
                5,
                [8, 16, 32, 64, 128],
                1,
                False,
            )
        elif count == 15:
            self._fmc, self._strides, self._num_anchors, self._use_kps = (
                5,
                [8, 16, 32, 64, 128],
                1,
                True,
            )
        else:
            raise RuntimeError(f"topologia SCRFD desconhecida: {count} saidas")

    # ------------------------------------------------------------- inferencia

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        opts = self.resolved_params(params)
        session = self._ensure_session()

        side = int(opts["input_size"])
        img_h, img_w = image_bgr.shape[:2]

        # A imagem entra redimensionada preservando proporcao e com padding a
        # direita e embaixo, que e como o SCRFD foi treinado.
        det_scale = min(side / img_w, side / img_h)
        new_w, new_h = int(round(img_w * det_scale)), int(round(img_h * det_scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        rgb = canvas[:, :, ::-1].astype(np.float32)
        blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]

        outputs = session.run(self._output_names, {self._input_name: blob})

        threshold = float(opts["score_threshold"])
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []

        for idx, stride in enumerate(self._strides):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + self._fmc].reshape(-1, 4) * stride

            grid_h, grid_w = side // stride, side // stride
            anchor_centers = np.stack(
                np.mgrid[:grid_h, :grid_w][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            if self._num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * self._num_anchors, axis=1
                ).reshape(-1, 2)

            keep = np.where(scores >= threshold)[0]
            if keep.size == 0:
                continue

            boxes = distance2bbox(anchor_centers[keep], bbox_preds[keep])
            all_boxes.append(boxes)
            all_scores.append(scores[keep])

            if self._use_kps:
                kps_preds = outputs[idx + self._fmc * 2].reshape(-1, 10) * stride
                kps = distance2kps(anchor_centers[keep], kps_preds[keep])
                all_kps.append(kps.reshape(-1, 5, 2))

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes) / det_scale
        scores = np.concatenate(all_scores)
        kps = np.vstack(all_kps) / det_scale if all_kps else None

        keep = nms(boxes, scores, float(opts["nms_threshold"]))
        keep = keep[: int(opts["top_k"])]

        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            landmarks = (
                [[float(p[0]), float(p[1])] for p in kps[i]] if kps is not None else None
            )
            score = float(scores[i])
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=max(0.0, x2 - x1),
                    h=max(0.0, y2 - y1),
                    score=max(0.0, min(1.0, score)),
                    raw_score=score,
                    landmarks=landmarks,
                )
            )
        return detections


class Scrfd10GDetector(_ScrfdBase):
    key = "scrfd_10g"
    label = "SCRFD-10GF (InsightFace buffalo_l)"
    model_key = "scrfd_10g"
    requires_models = ("scrfd_10g",)
    description = (
        "O detector do pacote buffalo_l do InsightFace, a referencia livre mais "
        "proxima do estado da arte usado em sistemas de seguranca reais. E uma "
        "rede convolucional profunda, imune ao tipo de ataque que funciona "
        "contra o Haar."
    )


class Scrfd500MDetector(_ScrfdBase):
    key = "scrfd_500m"
    label = "SCRFD-500MF (leve)"
    model_key = "scrfd_500m"
    requires_models = ("scrfd_500m",)
    description = (
        "Variante enxuta do SCRFD, treinada para rodar em hardware modesto. "
        "Util para checar se um detector menor e mais facil de enganar."
    )
