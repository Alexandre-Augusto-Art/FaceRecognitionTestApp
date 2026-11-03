"""Metricas de eficacia de um padrao tipo HyperFace.

A pergunta que estas metricas respondem nao e "o detector achou o rosto?", e
sim "o padrao produziu faces falsas fortes o bastante para desviar a atencao
do detector do rosto verdadeiro?".

O HyperFace explora uma preferencia comum nos sistemas que consomem um
detector: quando varias caixas voltam, o sistema costuma escolher a maior, ou a
de maior confianca, e ignorar o resto. Por isso `largest_box_is_fake` e
`top_box_is_fake` sao as duas leituras mais diretas de sucesso do padrao.
"""

from __future__ import annotations

from .schemas import Detection, DetectorMetrics, DetectorResult, GroundTruthBox

HISTOGRAM_BINS = 12


def iou(a: Detection | GroundTruthBox, b: Detection | GroundTruthBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h

    inter_w = min(ax2, bx2) - max(a.x, b.x)
    inter_h = min(ay2, by2) - max(a.y, b.y)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union


def _histogram(values: list[float], flags: list[bool]) -> list[dict]:
    """Distribuicao dos scores brutos, separando faces reais de falsas.

    Para o Haar os valores sao levelWeights, que nao tem limite superior fixo,
    entao os limites do histograma vem dos proprios dados.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        high = low + 1.0
    width = (high - low) / HISTOGRAM_BINS

    bins = [
        {
            "start": low + i * width,
            "end": low + (i + 1) * width,
            "true_count": 0,
            "false_count": 0,
        }
        for i in range(HISTOGRAM_BINS)
    ]
    for value, is_false in zip(values, flags):
        index = int((value - low) / width)
        index = max(0, min(HISTOGRAM_BINS - 1, index))
        bins[index]["false_count" if is_false else "true_count"] += 1
    return bins


def evaluate(
    result: DetectorResult,
    ground_truth: list[GroundTruthBox],
    iou_threshold: float = 0.3,
) -> DetectorMetrics:
    detections = result.detections
    total = len(detections)

    is_false: list[bool] = []
    for detection in detections:
        matched = any(iou(detection, gt) >= iou_threshold for gt in ground_truth)
        is_false.append(not matched)

    true_scores = [d.score for d, f in zip(detections, is_false) if not f]
    false_scores = [d.score for d, f in zip(detections, is_false) if f]

    true_face_score = max(true_scores) if true_scores else None
    max_false_score = max(false_scores) if false_scores else None

    decoy_ratio = None
    if max_false_score is not None and true_face_score is not None:
        decoy_ratio = (
            max_false_score / true_face_score if true_face_score > 1e-9 else None
        )
    elif max_false_score is not None and not ground_truth:
        decoy_ratio = None

    largest_box_is_fake = None
    top_box_is_fake = None
    if detections:
        largest_index = max(
            range(total), key=lambda i: detections[i].w * detections[i].h
        )
        largest_box_is_fake = is_false[largest_index]
        top_index = max(range(total), key=lambda i: detections[i].score)
        top_box_is_fake = is_false[top_index]

    raw_values = [
        float(d.raw_score if d.raw_score is not None else d.score) for d in detections
    ]

    return DetectorMetrics(
        detector=result.detector,
        label=result.label,
        total=total,
        true_positives=len(true_scores),
        false_faces=len(false_scores),
        true_face_score=true_face_score,
        max_false_score=max_false_score,
        decoy_ratio=decoy_ratio,
        largest_box_is_fake=largest_box_is_fake,
        top_box_is_fake=top_box_is_fake,
        mean_false_score=(
            sum(false_scores) / len(false_scores) if false_scores else None
        ),
        score_histogram=_histogram(raw_values, is_false),
    )


def evaluate_all(
    results: list[DetectorResult],
    ground_truth: list[GroundTruthBox],
    iou_threshold: float = 0.3,
) -> list[DetectorMetrics]:
    return [
        evaluate(result, ground_truth, iou_threshold) for result in results if result.ok
    ]
