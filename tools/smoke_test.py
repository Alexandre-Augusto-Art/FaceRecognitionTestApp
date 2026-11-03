"""Teste de fumaca: roda todos os detectores em uma imagem e reporta o resultado.

Uso:
    python -m tools.smoke_test [caminho_da_imagem]

Sem argumento, baixa uma foto de teste do repositorio do OpenCV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import detectors  # noqa: E402
from app.config import MODELS_DIR  # noqa: E402

FALLBACK_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"


def _test_image() -> tuple[np.ndarray, str]:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Nao consegui ler {path}")
        return image, str(path)

    cached = MODELS_DIR.parent / "lena.jpg"
    if not cached.exists():
        import requests

        cached.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(FALLBACK_URL, timeout=60)
        response.raise_for_status()
        cached.write_bytes(response.content)
    image = cv2.imdecode(np.fromfile(str(cached), dtype=np.uint8), cv2.IMREAD_COLOR)
    return image, str(cached)


def main() -> int:
    image, source = _test_image()
    height, width = image.shape[:2]
    print(f"Imagem: {source}  ({width} x {height})\n")

    print(f"{'detector':<26} {'status':<8} {'n':>4} {'ms':>9}  melhor score")
    print("-" * 72)

    failures = 0
    for info in detectors.describe_all():
        if not info.available:
            print(f"{info.key:<26} {'N/D':<8} {'-':>4} {'-':>9}  {info.unavailable_reason}")
            failures += 1
            continue
        result = detectors.run_one(info.key, image, {})
        if not result.ok:
            print(f"{info.key:<26} {'ERRO':<8} {'-':>4} {'-':>9}  {result.error}")
            failures += 1
            continue
        best = result.detections[0].score if result.detections else 0.0
        raw = result.detections[0].raw_score if result.detections else 0.0
        print(
            f"{info.key:<26} {'ok':<8} {len(result.detections):>4} "
            f"{result.elapsed_ms:>9.1f}  {best:.3f} (bruto {raw:.3f})"
        )

    print()
    if failures:
        print(f"{failures} detector(es) com problema.")
        return 1
    print("Todos os detectores responderam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
