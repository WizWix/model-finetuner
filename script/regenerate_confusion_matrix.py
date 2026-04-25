#!/usr/bin/env python3
"""평가 JSON으로부터 혼동 행렬 이미지를 재생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.confusion_matrix_plot import (
    extract_labels_from_metrics,
    load_metrics_json,
    render_confusion_matrix_image,
    render_row_normalized_confusion_matrix_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="평가 JSON(confusion_matrix/per_class 포함)으로 혼동 행렬 PNG를 생성합니다."
    )
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="입력 평가 JSON 경로 (예: aaa.json)",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="",
        help="카운트 버전 출력 PNG 경로 (미지정 시 metrics JSON 이름 기반 자동 생성)",
    )
    parser.add_argument(
        "--row-normalized-output",
        required=False,
        default="",
        help="row-normalized 버전 출력 PNG 경로 (미지정 시 카운트 파일명에서 자동 파생)",
    )
    parser.add_argument(
        "--trial-label",
        default="",
        help="타이틀에 추가할 Trial 라벨 (예: Trial predefined-finetune)",
    )
    parser.add_argument(
        "--font",
        action="append",
        dest="fonts",
        default=[],
        help="우선 적용할 폰트 패밀리명(여러 번 지정 가능). 기본값은 NanumGothic 우선.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="출력 DPI",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = load_metrics_json(args.metrics_json)

    cm = metrics.get("confusion_matrix")
    if not isinstance(cm, list) or not cm:
        raise ValueError("metrics['confusion_matrix']가 비어 있거나 리스트가 아닙니다.")

    labels = extract_labels_from_metrics(metrics)
    metrics_path = Path(args.metrics_json).resolve()
    if args.output:
        count_output_path = Path(args.output).resolve()
    else:
        count_output_path = metrics_path.with_name(f"{metrics_path.stem}_counts.png")
    if args.row_normalized_output:
        row_norm_output_path = Path(args.row_normalized_output).resolve()
    else:
        row_norm_output_path = count_output_path.with_name(
            f"{count_output_path.stem}_row_normalized.png"
        )

    result_count = render_confusion_matrix_image(
        confusion_matrix_values=cm,
        labels=labels,
        output_path=count_output_path,
        accuracy=float(metrics["accuracy"]) if "accuracy" in metrics else None,
        f1_macro=float(metrics["f1_macro"]) if "f1_macro" in metrics else None,
        trial_label=args.trial_label or None,
        font_candidates=args.fonts or None,
        dpi=args.dpi,
    )
    result_row = render_row_normalized_confusion_matrix_image(
        confusion_matrix_values=cm,
        labels=labels,
        output_path=row_norm_output_path,
        accuracy=float(metrics["accuracy"]) if "accuracy" in metrics else None,
        f1_macro=float(metrics["f1_macro"]) if "f1_macro" in metrics else None,
        f1_weighted=float(metrics["f1_weighted"]) if "f1_weighted" in metrics else None,
        sample_count=int(metrics["total"]) if "total" in metrics else None,
        font_candidates=args.fonts or None,
        dpi=args.dpi,
    )

    count_path = Path(result_count["output_path"]).resolve()
    row_norm_path = Path(result_row["output_path"]).resolve()
    selected_font = result_count.get("font_family") or result_row.get("font_family")
    if selected_font:
        print(f"[ok] 생성 완료(카운트): {count_path}")
        print(f"[ok] 생성 완료(Row-normalized): {row_norm_path}")
        print(f"[font] 적용 폰트: {selected_font}")
    else:
        print(f"[ok] 생성 완료(카운트): {count_path}")
        print(f"[ok] 생성 완료(Row-normalized): {row_norm_path}")
        print("[font] 한글 폰트를 찾지 못해 Matplotlib 기본 폰트로 렌더링했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
