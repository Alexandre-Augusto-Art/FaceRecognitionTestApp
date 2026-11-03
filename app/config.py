"""Caminhos, catalogo de modelos e presets de parametros."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"


def _app_home() -> Path:
    """Diretorio para venv, modelos e cache.

    Fica fora do projeto por padrao porque o projeto costuma viver em uma pasta
    do Google Drive, onde escrever centenas de MB e milhares de arquivos e
    lento e consome cota de sincronizacao sem necessidade. Os modelos sao
    sempre re-baixaveis, entao nao precisam ser versionados nem sincronizados.
    """
    override = os.environ.get("HFT_HOME")
    if override:
        return Path(override)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "HyperFaceTestbed"
    return PROJECT_ROOT / ".hft"


APP_HOME = _app_home()
MODELS_DIR = APP_HOME / "models"


def ensure_dirs() -> None:
    for path in (DATA_DIR, UPLOADS_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Catalogo de modelos
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    url: str
    filename: str
    approx_mb: float
    description: str
    # "zip" extrai o arquivo; "bz2" descomprime. None grava direto.
    archive: str | None = None
    # Para archives: caminho do arquivo desejado dentro do zip.
    member: str | None = None
    # Nome final apos extracao (quando archive != None).
    extracted_as: str | None = None

    @property
    def target(self) -> Path:
        name = self.extracted_as or self.filename
        return MODELS_DIR / name


_HAAR_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
_LBP_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/lbpcascades/"

MODEL_CATALOG: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        key="yunet",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        filename="face_detection_yunet_2023mar.onnx",
        approx_mb=0.23,
        description="YuNet - detector DNN oficial do OpenCV desde a 4.5.4",
    ),
    # Os cascades Haar deixaram de vir empacotados no opencv-python a partir da
    # versao 5.0 (o diretorio cv2/data ficou vazio), entao baixamos os XMLs
    # originais do repositorio do OpenCV.
    "haar_frontal_default": ModelSpec(
        key="haar_frontal_default",
        url=_HAAR_BASE + "haarcascade_frontalface_default.xml",
        filename="haarcascade_frontalface_default.xml",
        approx_mb=0.89,
        description="Haar frontal default - o perfil exato que o HyperFace ataca",
    ),
    "haar_frontal_alt": ModelSpec(
        key="haar_frontal_alt",
        url=_HAAR_BASE + "haarcascade_frontalface_alt.xml",
        filename="haarcascade_frontalface_alt.xml",
        approx_mb=0.65,
        description="Haar frontal alt",
    ),
    "haar_frontal_alt2": ModelSpec(
        key="haar_frontal_alt2",
        url=_HAAR_BASE + "haarcascade_frontalface_alt2.xml",
        filename="haarcascade_frontalface_alt2.xml",
        approx_mb=0.52,
        description="Haar frontal alt2",
    ),
    "haar_frontal_alt_tree": ModelSpec(
        key="haar_frontal_alt_tree",
        url=_HAAR_BASE + "haarcascade_frontalface_alt_tree.xml",
        filename="haarcascade_frontalface_alt_tree.xml",
        approx_mb=2.56,
        description="Haar frontal alt_tree - cascade em arvore, mais exigente",
    ),
    "haar_profile": ModelSpec(
        key="haar_profile",
        url=_HAAR_BASE + "haarcascade_profileface.xml",
        filename="haarcascade_profileface.xml",
        approx_mb=0.79,
        description="Haar de perfil",
    ),
    "lbp_frontal_improved": ModelSpec(
        key="lbp_frontal_improved",
        url=_LBP_BASE + "lbpcascade_frontalface_improved.xml",
        filename="lbpcascade_frontalface_improved.xml",
        approx_mb=0.05,
        description="LBP Cascade frontal (versao melhorada)",
    ),
    "lbp_frontal": ModelSpec(
        key="lbp_frontal",
        url=_LBP_BASE + "lbpcascade_frontalface.xml",
        filename="lbpcascade_frontalface.xml",
        approx_mb=0.05,
        description="LBP Cascade frontal (original)",
    ),
    "lbp_profile": ModelSpec(
        key="lbp_profile",
        url=_LBP_BASE + "lbpcascade_profileface.xml",
        filename="lbpcascade_profileface.xml",
        approx_mb=0.04,
        description="LBP Cascade de perfil",
    ),
    "scrfd_10g": ModelSpec(
        key="scrfd_10g",
        url="https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx",
        filename="scrfd_10g_bnkps.onnx",
        approx_mb=16.1,
        description=(
            "SCRFD-10GF, o detector do pacote buffalo_l do InsightFace. "
            "E o detector livre mais proximo do estado da arte usado em "
            "sistemas de vigilancia reais."
        ),
    ),
    "scrfd_500m": ModelSpec(
        key="scrfd_500m",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        filename="buffalo_sc.zip",
        approx_mb=14.3,
        description="SCRFD-500MF, variante leve do SCRFD (pacote buffalo_sc)",
        archive="zip",
        member="buffalo_sc/det_500m.onnx",
        extracted_as="scrfd_500m.onnx",
    ),
    "ultraface_320": ModelSpec(
        key="ultraface_320",
        url=(
            "https://github.com/onnx/models/raw/main/validated/vision/"
            "body_analysis/ultraface/models/version-RFB-320.onnx"
        ),
        filename="ultraface_rfb_320.onnx",
        approx_mb=1.2,
        description=(
            "UltraFace RFB-320 - detector ultraleve muito usado em camera "
            "embarcada e dispositivos de borda"
        ),
    ),
    "ultraface_640": ModelSpec(
        key="ultraface_640",
        url=(
            "https://github.com/onnx/models/raw/main/validated/vision/"
            "body_analysis/ultraface/models/version-RFB-640.onnx"
        ),
        filename="ultraface_rfb_640.onnx",
        approx_mb=1.5,
        description="UltraFace RFB-640 - mesma rede em resolucao maior",
    ),
    "dlib_mmod": ModelSpec(
        key="dlib_mmod",
        url="http://dlib.net/files/mmod_human_face_detector.dat.bz2",
        filename="mmod_human_face_detector.dat.bz2",
        approx_mb=0.7,
        description="dlib CNN (MMOD) - detector CNN do dlib",
        archive="bz2",
        extracted_as="mmod_human_face_detector.dat",
    ),
    "mediapipe_blazeface": ModelSpec(
        key="mediapipe_blazeface",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ),
        filename="blaze_face_short_range.tflite",
        approx_mb=0.23,
        description="BlazeFace (MediaPipe) - detector do Google usado em produtos de consumo",
    ),
}


# --------------------------------------------------------------------------
# Presets de parametros
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    params: dict = field(default_factory=dict)


PRESETS: list[Preset] = [
    Preset(
        key="hyperface",
        label="HyperFace (Harvey, 2017)",
        description=(
            "Configuracao descrita por Adam Harvey para revelar as faces falsas: "
            "scaleFactor reduzido a 1.05 e minSize no minimo absoluto de 24. "
            "minNeighbors baixo expoe as candidatas de 0 sobreposicoes."
        ),
        params={
            "scale_factor": 1.05,
            "min_neighbors": 1,
            "min_size": 24,
            "max_size": 0,
            "score_threshold": 0.5,
            "nms_threshold": 0.3,
        },
    ),
    Preset(
        key="opencv_default",
        label="OpenCV padrao",
        description=(
            "Os valores que a maioria dos tutoriais e aplicacoes usa. "
            "Serve de linha de base: e o que um sistema desatento rodaria."
        ),
        params={
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": 30,
            "max_size": 0,
            "score_threshold": 0.5,
            "nms_threshold": 0.3,
        },
    ),
    Preset(
        key="cctv",
        label="CCTV / baixa resolucao",
        description=(
            "Varredura fina com janela minima pequena, aproximando cameras de "
            "vigilancia que precisam pegar rostos distantes."
        ),
        params={
            "scale_factor": 1.03,
            "min_neighbors": 3,
            "min_size": 20,
            "max_size": 0,
            "score_threshold": 0.4,
            "nms_threshold": 0.4,
        },
    ),
    Preset(
        key="permissive",
        label="Permissivo (0-overlap)",
        description=(
            "minNeighbors = 0 mostra toda janela que o cascade aceitou, sem "
            "agrupamento. E a visao mais crua de onde o padrao ativa o detector."
        ),
        params={
            "scale_factor": 1.05,
            "min_neighbors": 0,
            "min_size": 24,
            "max_size": 0,
            "score_threshold": 0.25,
            "nms_threshold": 0.5,
        },
    ),
    Preset(
        key="strict",
        label="Estrito",
        description="Alta exigencia: poucas deteccoes, so as muito confiantes.",
        params={
            "scale_factor": 1.2,
            "min_neighbors": 8,
            "min_size": 48,
            "max_size": 0,
            "score_threshold": 0.8,
            "nms_threshold": 0.3,
        },
    ),
]

# Cor de cada detector no overlay do canvas.
DETECTOR_COLORS: dict[str, str] = {
    "haar": "#ff3b30",
    "lbp": "#ff9500",
    "yunet": "#34c759",
    "scrfd_10g": "#0a84ff",
    "scrfd_500m": "#5e5ce6",
    "ultraface": "#bf5af2",
    "dlib_hog": "#ffd60a",
    "dlib_cnn": "#ff6482",
    "mediapipe_blazeface": "#64d2ff",
}

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
