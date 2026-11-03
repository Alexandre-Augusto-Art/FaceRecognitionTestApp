"""Validacao ponta a ponta pela API HTTP.

Monta uma imagem composta - um rosto real cercado de faces falsas - e a envia
para o servidor exatamente como a interface faz: upload, deteccao e metricas.
E o cenario que o testbed existe para medir, e serve para conferir que a pilha
inteira esta coerente.

Uso (com o servidor rodando):
    python -m tools.validate [--url http://127.0.0.1:8010]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import APP_HOME  # noqa: E402
from tools.make_pattern import generate  # noqa: E402

FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"


def _real_face() -> np.ndarray:
    cached = APP_HOME / "lena.jpg"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(FACE_URL, timeout=60)
        response.raise_for_status()
        cached.write_bytes(response.content)
    return cv2.imdecode(np.fromfile(str(cached), dtype=np.uint8), cv2.IMREAD_COLOR)


def build_composite() -> tuple[np.ndarray, dict]:
    """Rosto real no centro, faces falsas ao redor.

    Devolve tambem a caixa do rosto verdadeiro, localizada com o SCRFD antes da
    composicao para nao depender de coordenadas chutadas na mao.
    """
    from app import detectors

    face = _real_face()
    canvas = generate(1400, 1000, faces=38, seed=11)

    face_h, face_w = face.shape[:2]
    origin_x = (canvas.shape[1] - face_w) // 2
    origin_y = (canvas.shape[0] - face_h) // 2
    canvas[origin_y : origin_y + face_h, origin_x : origin_x + face_w] = face

    located = detectors.run_one("scrfd_10g", face, {"score_threshold": 0.5})
    if not located.ok or not located.detections:
        raise SystemExit("Nao consegui localizar o rosto real na foto de referencia.")
    box = located.detections[0]

    ground_truth = {
        "x": box.x + origin_x,
        "y": box.y + origin_y,
        "w": box.w,
        "h": box.h,
    }
    return canvas, ground_truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print("Montando a imagem composta (rosto real + faces falsas) ...")
    composite, ground_truth = build_composite()

    output = Path("data/exemplos/composto_rosto_e_falsas.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", composite)
    if not ok:
        raise SystemExit("Falha ao codificar a composicao.")
    buffer.tofile(str(output))
    print(f"  salvo em {output}")
    print(
        f"  rosto verdadeiro em x={ground_truth['x']:.0f} y={ground_truth['y']:.0f} "
        f"{ground_truth['w']:.0f}x{ground_truth['h']:.0f}\n"
    )

    try:
        catalog = requests.get(f"{base}/api/detectors", timeout=30).json()
    except requests.RequestException as exc:
        raise SystemExit(f"Servidor inacessivel em {base}. Rode .\\run.ps1 antes. ({exc})")

    keys = [d["key"] for d in catalog["detectors"] if d["available"]]
    hyperface = next(p for p in catalog["presets"] if p["key"] == "hyperface")

    with open(output, "rb") as handle:
        upload = requests.post(
            f"{base}/api/upload",
            files={"file": (output.name, handle, "image/png")},
            timeout=120,
        )
    upload.raise_for_status()
    image = upload.json()
    print(f"Upload ok: {image['width']} x {image['height']}\n")

    params = {}
    for detector in catalog["detectors"]:
        if detector["key"] not in keys:
            continue
        values = {}
        for spec in detector["params"]:
            values[spec["key"]] = hyperface["params"].get(spec["key"], spec["default"])
        params[detector["key"]] = values

    print(f"Rodando {len(keys)} detectores com o preset HyperFace ...")
    detection = requests.post(
        f"{base}/api/detect",
        json={"image_id": image["image_id"], "detectors": keys, "params": params},
        timeout=600,
    )
    detection.raise_for_status()
    results = detection.json()["results"]

    metrics_response = requests.post(
        f"{base}/api/metrics",
        json={
            "ground_truth": [ground_truth],
            "results": results,
            "iou_threshold": 0.3,
        },
        timeout=120,
    )
    metrics_response.raise_for_status()
    metrics = metrics_response.json()["metrics"]

    print()
    header = (
        f"{'detector':<24} {'caixas':>6} {'falsas':>7} {'real':>7} "
        f"{'falso':>7} {'decoy':>7}  veredito"
    )
    print(header)
    print("-" * len(header))

    def show(value, digits=3):
        return "  --  " if value is None else f"{value:.{digits}f}"

    for metric in metrics:
        if metric["total"] == 0:
            verdict = "nada detectado"
        elif metric["top_box_is_fake"] or metric["largest_box_is_fake"]:
            verdict = "PADRAO VENCEU"
        elif metric["false_faces"] > 0:
            verdict = "ruido gerado"
        else:
            verdict = "detector resistiu"
        print(
            f"{metric['detector']:<24} {metric['total']:>6} {metric['false_faces']:>7} "
            f"{show(metric['true_face_score']):>7} {show(metric['max_false_score']):>7} "
            f"{show(metric['decoy_ratio'], 2):>7}  {verdict}"
        )

    print(
        "\nColunas: 'real' e o score da melhor caixa sobre o rosto verdadeiro, "
        "'falso' o da isca mais forte,\n'decoy' a razao entre os dois. "
        "Acima de 1.00 a isca e mais convincente que o rosto de verdade."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
