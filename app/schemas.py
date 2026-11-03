"""Contratos de dados trocados entre backend e frontend."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Uma caixa retornada por um detector.

    `score` e sempre normalizado em 0..1 para permitir comparacao entre
    algoritmos. `raw_score` preserva o valor nativo, que para o Haar e o
    levelWeight do estagio final do cascade e nao tem limite superior fixo.
    """

    x: float
    y: float
    w: float
    h: float
    score: float
    raw_score: float | None = None
    neighbors: int | None = None
    landmarks: list[list[float]] | None = None


class DetectorParam(BaseModel):
    """Descricao de um parametro ajustavel, usada para montar a UI."""

    key: str
    label: str
    kind: str  # "float" | "int" | "bool" | "choice"
    default: bool | int | float | str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[dict] | None = None
    help: str | None = None


class DetectorInfo(BaseModel):
    key: str
    label: str
    family: str
    tier: int
    color: str
    available: bool
    unavailable_reason: str | None = None
    description: str
    params: list[DetectorParam] = Field(default_factory=list)


class DetectorResult(BaseModel):
    detector: str
    label: str
    color: str
    ok: bool
    error: str | None = None
    elapsed_ms: float = 0.0
    detections: list[Detection] = Field(default_factory=list)
    params_used: dict = Field(default_factory=dict)


class GroundTruthBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectRequest(BaseModel):
    image_id: str
    detectors: list[str]
    params: dict[str, dict] = Field(default_factory=dict)


class DetectorMetrics(BaseModel):
    detector: str
    label: str
    total: int
    true_positives: int
    false_faces: int
    true_face_score: float | None = None
    max_false_score: float | None = None
    decoy_ratio: float | None = None
    largest_box_is_fake: bool | None = None
    top_box_is_fake: bool | None = None
    mean_false_score: float | None = None
    score_histogram: list[dict] = Field(default_factory=list)


class MetricsRequest(BaseModel):
    ground_truth: list[GroundTruthBox]
    results: list[DetectorResult]
    iou_threshold: float = 0.3


class ImageInfo(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int


class DetectResponse(BaseModel):
    image: ImageInfo
    results: list[DetectorResult]
