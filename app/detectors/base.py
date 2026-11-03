"""Interface comum a todos os detectores e utilitarios compartilhados."""

from __future__ import annotations

import math

import numpy as np

from ..config import DETECTOR_COLORS, MODEL_CATALOG
from ..downloader import is_present
from ..schemas import Detection, DetectorParam


class DetectorUnavailable(RuntimeError):
    """O detector existe no catalogo mas nao pode rodar agora."""


class BaseDetector:
    key: str = ""
    label: str = ""
    family: str = ""
    tier: int = 1
    description: str = ""
    requires_models: tuple[str, ...] = ()
    requires_packages: tuple[str, ...] = ()

    # ---------------------------------------------------------------- infra

    @property
    def color(self) -> str:
        return DETECTOR_COLORS.get(self.key, "#8e8e93")

    def params_spec(self) -> list[DetectorParam]:
        return []

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params_spec()}

    def missing_models(self) -> list[str]:
        return [k for k in self.requires_models if not is_present(MODEL_CATALOG[k])]

    def missing_packages(self) -> list[str]:
        import importlib.util

        return [
            name
            for name in self.requires_packages
            if importlib.util.find_spec(name) is None
        ]

    def availability(self) -> tuple[bool, str | None]:
        missing_pkgs = self.missing_packages()
        if missing_pkgs:
            return False, (
                "Pacote(s) ausente(s): "
                + ", ".join(missing_pkgs)
                + ". Rode setup.ps1 -Extras."
            )
        missing = self.missing_models()
        if missing:
            names = ", ".join(missing)
            return False, f"Modelo(s) nao baixado(s): {names}. Rode setup.ps1 -Models."
        return True, None

    # --------------------------------------------------------------- runtime

    def detect(self, image_bgr: np.ndarray, params: dict) -> list[Detection]:
        raise NotImplementedError

    def resolved_params(self, params: dict | None) -> dict:
        merged = self.defaults()
        for spec in self.params_spec():
            if params and spec.key in params and params[spec.key] is not None:
                value = params[spec.key]
                if spec.kind == "int":
                    value = int(round(float(value)))
                elif spec.kind == "float":
                    value = float(value)
                elif spec.kind == "bool":
                    value = bool(value)
                else:
                    value = str(value)
                if spec.minimum is not None and isinstance(value, (int, float)):
                    value = max(spec.minimum, value)
                if spec.maximum is not None and isinstance(value, (int, float)):
                    value = min(spec.maximum, value)
                merged[spec.key] = value
        return merged


# -------------------------------------------------------------------------
# Especificacoes de parametros reutilizaveis
# -------------------------------------------------------------------------


def cascade_params(profiles: list[dict], default_profile: str) -> list[DetectorParam]:
    return [
        DetectorParam(
            key="profile",
            label="Perfil (arquivo XML)",
            kind="choice",
            default=default_profile,
            choices=profiles,
            help="Cada XML e um classificador treinado separadamente.",
        ),
        DetectorParam(
            key="scale_factor",
            label="scaleFactor",
            kind="float",
            default=1.05,
            minimum=1.01,
            maximum=1.5,
            step=0.01,
            help=(
                "Quanto a janela cresce a cada passagem da piramide. Valores "
                "baixos varrem mais escalas e revelam mais faces falsas; "
                "Harvey usa 1.05."
            ),
        ),
        DetectorParam(
            key="min_neighbors",
            label="minNeighbors",
            kind="int",
            default=1,
            minimum=0,
            maximum=15,
            step=1,
            help=(
                "Quantas deteccoes vizinhas precisam concordar. 0 mostra toda "
                "janela aceita, sem agrupamento - o modo 0-overlap."
            ),
        ),
        DetectorParam(
            key="min_size",
            label="minSize (px)",
            kind="int",
            default=24,
            minimum=12,
            maximum=400,
            step=1,
            help="Menor face procurada. 24 e o minimo absoluto util do cascade.",
        ),
        DetectorParam(
            key="max_size",
            label="maxSize (px, 0 = sem limite)",
            kind="int",
            default=0,
            minimum=0,
            maximum=4096,
            step=1,
        ),
        DetectorParam(
            key="equalize_hist",
            label="Equalizar histograma",
            kind="bool",
            default=True,
            help="Pre-processamento classico do pipeline Viola-Jones.",
        ),
        DetectorParam(
            key="count_neighbors",
            label="Contar vizinhos (overlaps)",
            kind="bool",
            default=True,
            help=(
                "Faz uma segunda passagem para obter o numero de janelas que "
                "concordaram com cada caixa. Custa mais tempo."
            ),
        ),
    ]


def dnn_params(
    default_score: float = 0.5,
    default_nms: float = 0.4,
    input_sizes: list[dict] | None = None,
    default_input: str = "640",
) -> list[DetectorParam]:
    params = [
        DetectorParam(
            key="score_threshold",
            label="Limiar de confianca",
            kind="float",
            default=default_score,
            minimum=0.01,
            maximum=0.99,
            step=0.01,
            help="Abaixe para ver as deteccoes fracas que o padrao provoca.",
        ),
        DetectorParam(
            key="nms_threshold",
            label="Limiar de NMS",
            kind="float",
            default=default_nms,
            minimum=0.1,
            maximum=0.9,
            step=0.05,
            help="IoU acima do qual duas caixas sao consideradas a mesma face.",
        ),
    ]
    if input_sizes:
        params.append(
            DetectorParam(
                key="input_size",
                label="Resolucao de entrada",
                kind="choice",
                default=default_input,
                choices=input_sizes,
                help=(
                    "A rede redimensiona a imagem para este tamanho. Maior "
                    "resolucao encontra faces menores."
                ),
            )
        )
    params.append(
        DetectorParam(
            key="top_k",
            label="Maximo de deteccoes",
            kind="int",
            default=500,
            minimum=10,
            maximum=5000,
            step=10,
        )
    )
    return params


# -------------------------------------------------------------------------
# Utilitarios numericos
# -------------------------------------------------------------------------


def logistic(value: float) -> float:
    """Mapeia um score sem limite superior para 0..1 preservando a ordem.

    O Haar devolve levelWeights que podem ir de valores negativos a mais de 10,
    e o dlib HOG devolve um score de SVM tambem sem escala fixa. Para comparar
    detectores no mesmo grafico precisamos de uma escala comum, e a logistica
    e monotonica, entao nao altera o ranking das deteccoes.
    """
    if value > 30:
        return 1.0
    if value < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """Non-maximum suppression sobre caixas no formato x1,y1,x2,y2."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Converte a saida do SCRFD (distancias ao centro da ancora) em caixas."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    landmarks: np.ndarray | None = None,
    raw_scores: np.ndarray | None = None,
) -> list[Detection]:
    """Converte arrays x1,y1,x2,y2 em objetos Detection."""
    out: list[Detection] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in boxes[i][:4])
        kps = None
        if landmarks is not None and len(landmarks) > i:
            kps = [[float(p[0]), float(p[1])] for p in landmarks[i]]
        out.append(
            Detection(
                x=x1,
                y=y1,
                w=max(0.0, x2 - x1),
                h=max(0.0, y2 - y1),
                score=float(scores[i]),
                raw_score=(
                    float(raw_scores[i]) if raw_scores is not None else float(scores[i])
                ),
                landmarks=kps,
            )
        )
    return out
