from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_KOREAN_FONT_CANDIDATES = [
    "NanumGothic",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Malgun Gothic",
    "AppleGothic",
    "UnDotum",
]


def _compute_figure_size(labels: list[str]) -> float:
    n = len(labels)
    max_label_len = max((len(label) for label in labels), default=4)
    return max(8.0, n * 0.6, max_label_len * 0.45)


def _refresh_font_cache() -> None:
    from matplotlib import font_manager

    font_manager.fontManager = font_manager.FontManager()


def select_korean_font_family(
    preferred_families: list[str] | None = None,
) -> str | None:
    from matplotlib import font_manager

    candidates = preferred_families or DEFAULT_KOREAN_FONT_CANDIDATES
    lower_candidates = [name.lower() for name in candidates]

    for _ in range(2):
        names = {f.name for f in font_manager.fontManager.ttflist if f.name}
        lower_to_name = {name.lower(): name for name in names}
        for candidate in lower_candidates:
            if candidate in lower_to_name:
                return lower_to_name[candidate]
        _refresh_font_cache()

    # Last-resort fallback by filename keyword match.
    for font_path in font_manager.findSystemFonts():
        lower_path = font_path.lower()
        if any(keyword in lower_path for keyword in ("nanum", "noto", "malgun", "cjk")):
            try:
                return font_manager.FontProperties(fname=font_path).get_name()
            except Exception:
                continue
    return None


def extract_labels_from_metrics(metrics: dict[str, Any]) -> list[str]:
    if isinstance(metrics.get("labels"), list):
        return [str(x) for x in metrics["labels"]]
    per_class = metrics.get("per_class")
    if isinstance(per_class, dict) and per_class:
        return [str(x) for x in per_class.keys()]
    raise ValueError(
        "클래스 라벨을 찾을 수 없습니다. metrics['labels'] 또는 metrics['per_class']가 필요합니다."
    )


def load_metrics_json(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON 루트는 dict 여야 합니다.")
    return data


def render_confusion_matrix_image(
    *,
    confusion_matrix_values: list[list[int]] | list[list[float]],
    labels: list[str],
    output_path: str | Path,
    accuracy: float | None = None,
    f1_macro: float | None = None,
    trial_label: str | None = None,
    font_candidates: list[str] | None = None,
    dpi: int = 150,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(confusion_matrix_values)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion_matrix는 정사각 2차원 배열이어야 합니다.")
    if cm.shape[0] != len(labels):
        raise ValueError(
            f"라벨 수({len(labels)})와 confusion_matrix 크기({cm.shape[0]})가 다릅니다."
        )

    selected_font = select_korean_font_family(font_candidates)
    if selected_font:
        plt.rcParams["font.family"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(labels)
    fig_size = _compute_figure_size(labels)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("예측 (Predicted)")
    ax.set_ylabel("실제 (Actual)")

    title = "Confusion Matrix"
    if trial_label:
        title += f" {trial_label}"
    if accuracy is not None and f1_macro is not None:
        title += f"\nAcc={accuracy:.3f}  F1(macro)={f1_macro:.3f}"
    ax.set_title(title)

    threshold = float(cm.max()) / 2 if cm.size else 0.0
    for i in range(n):
        for j in range(n):
            value = cm[i, j]
            if value > 0:
                ax.text(
                    j,
                    i,
                    str(int(value) if float(value).is_integer() else value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > threshold else "black",
                )

    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    return {"output_path": str(output), "font_family": selected_font}


def render_row_normalized_confusion_matrix_image(
    *,
    confusion_matrix_values: list[list[int]] | list[list[float]],
    labels: list[str],
    output_path: str | Path,
    accuracy: float | None = None,
    f1_macro: float | None = None,
    f1_weighted: float | None = None,
    sample_count: int | None = None,
    font_candidates: list[str] | None = None,
    dpi: int = 150,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    cm = np.asarray(confusion_matrix_values, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion_matrix는 정사각 2차원 배열이어야 합니다.")
    if cm.shape[0] != len(labels):
        raise ValueError(
            f"라벨 수({len(labels)})와 confusion_matrix 크기({cm.shape[0]})가 다릅니다."
        )

    row_sum = cm.sum(axis=1, keepdims=True)
    row_norm_percent = np.divide(
        cm * 100.0,
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )

    selected_font = select_korean_font_family(font_candidates)
    if selected_font:
        plt.rcParams["font.family"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(labels)
    fig_size = _compute_figure_size(labels)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(row_norm_percent, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    cbar.set_label("Row-normalized %")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("예측 (Predicted)")
    ax.set_ylabel("실제 (Actual)")

    summary = "Row-normalized to recall."
    if accuracy is not None:
        summary += f" Acc={accuracy * 100:.2f}%"
    if f1_macro is not None:
        summary += f" F1_macro={f1_macro * 100:.2f}%"
    if f1_weighted is not None:
        summary += f" F1_weighted={f1_weighted * 100:.2f}%"
    if sample_count is not None:
        summary += f" n={int(sample_count)}"
    ax.set_title(f"Confusion Matrix (%)\n{summary}")

    threshold = 50.0
    for i in range(n):
        for j in range(n):
            value = row_norm_percent[i, j]
            if value > 0:
                text = f"{value:.1f}".rstrip("0").rstrip(".")
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > threshold else "black",
                )

    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    return {"output_path": str(output), "font_family": selected_font}
