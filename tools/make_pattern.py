"""Gera um padrao sintetico de faces falsas, no espirito do HyperFace.

Serve para dois usos: validar que o testbed mede o que promete, e dar um ponto
de partida para desenhar padroes reais.

O principio explorado e o mesmo do HyperFace. O Haar Cascade decide olhando
diferencas de intensidade entre regioes retangulares vizinhas: a regiao dos
olhos e mais escura que a das bochechas, a ponte do nariz e mais clara que os
olhos ao lado. Uma mancha abstrata que reproduza essas relacoes de luminancia
ativa o cascade sem se parecer com um rosto para um olho humano nem para uma
rede convolucional profunda, que aprendeu textura facial de verdade.

Uso:
    python -m tools.make_pattern [saida.png] [--faces 40] [--size 1200x900]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def draw_proto_face(canvas: np.ndarray, cx: int, cy: int, size: int, rng) -> None:
    """Desenha uma unica face falsa: so as relacoes de luminancia que importam."""
    half = size // 2
    base = int(rng.integers(185, 240))

    # Ovalo claro fazendo o papel do rosto.
    cv2.ellipse(
        canvas,
        (cx, cy),
        (int(half * 0.78), half),
        0,
        0,
        360,
        (base, base, base),
        -1,
        lineType=cv2.LINE_AA,
    )

    eye_dark = int(rng.integers(20, 65))
    eye_w = max(2, int(size * 0.20))
    eye_h = max(1, int(size * 0.11))
    eye_y = cy - int(size * 0.13)
    offset = int(size * 0.21)

    for sign in (-1, 1):
        cv2.ellipse(
            canvas,
            (cx + sign * offset, eye_y),
            (eye_w // 2, eye_h // 2),
            0,
            0,
            360,
            (eye_dark, eye_dark, eye_dark),
            -1,
            lineType=cv2.LINE_AA,
        )

    # Ponte do nariz clara entre os olhos: e essa a feature central do
    # classificador frontal do OpenCV.
    bridge = min(255, base + 15)
    cv2.rectangle(
        canvas,
        (cx - max(1, int(size * 0.05)), eye_y - eye_h),
        (cx + max(1, int(size * 0.05)), cy + int(size * 0.10)),
        (bridge, bridge, bridge),
        -1,
    )

    mouth_dark = int(rng.integers(55, 105))
    cv2.ellipse(
        canvas,
        (cx, cy + int(size * 0.27)),
        (max(2, int(size * 0.20)), max(1, int(size * 0.06))),
        0,
        0,
        360,
        (mouth_dark, mouth_dark, mouth_dark),
        -1,
        lineType=cv2.LINE_AA,
    )


def generate(width: int, height: int, faces: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)

    # Fundo com blocos coloridos, para o padrao nao ser um campo liso.
    for _ in range(90):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        w = int(rng.integers(20, 140))
        h = int(rng.integers(20, 140))
        color = tuple(int(c) for c in rng.integers(15, 90, size=3))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)

    for _ in range(faces):
        size = int(rng.integers(34, 96))
        cx = int(rng.integers(size, max(size + 1, width - size)))
        cy = int(rng.integers(size, max(size + 1, height - size)))
        draw_proto_face(canvas, cx, cy, size, rng)

    noise = rng.normal(0, 5, canvas.shape)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="padrao_falsas_faces.png")
    parser.add_argument("--faces", type=int, default=40)
    parser.add_argument("--size", default="1200x900")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    image = generate(width, height, args.faces, args.seed)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(destination.suffix or ".png", image)
    if not ok:
        raise SystemExit("Falha ao codificar a imagem.")
    buffer.tofile(str(destination))

    print(f"Padrao gerado: {destination}  ({width} x {height}, {args.faces} faces falsas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
