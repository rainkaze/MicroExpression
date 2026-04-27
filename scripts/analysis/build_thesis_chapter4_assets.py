from __future__ import annotations

import csv
import html
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis"
OUT_DIR = PROJECT_ROOT / "artifacts" / "thesis" / "chapter4"

MODEL_LABELS = {
    "uv4_baseline_5fold": "UV",
    "depth4_baseline_5fold": "Depth",
    "uvd4_concat_5fold": "UVD-Concat",
    "uvd4_attention_5fold": "UVD-Attention",
    "uvd4_attention_focal_sampler_5fold": "Focal+Sampler",
    "uvd4_masked_attention_5fold": "Masked",
    "uvd4_residual_masked_attention_5fold": "Residual-Masked",
    "uv7_baseline_5fold": "UV",
    "depth7_baseline_5fold": "Depth",
    "uvd7_concat_5fold": "UVD-Concat",
    "uvd7_attention_5fold": "UVD-Attention",
    "uvd7_attention_focal_sampler_5fold": "Focal+Sampler",
    "uvd7_masked_attention_5fold": "Masked",
    "uvd7_residual_masked_attention_5fold": "Residual-Masked",
}

ORDER_4 = [
    "uv4_baseline_5fold",
    "depth4_baseline_5fold",
    "uvd4_concat_5fold",
    "uvd4_attention_5fold",
    "uvd4_attention_focal_sampler_5fold",
    "uvd4_masked_attention_5fold",
    "uvd4_residual_masked_attention_5fold",
]
ORDER_7 = [
    "uv7_baseline_5fold",
    "depth7_baseline_5fold",
    "uvd7_concat_5fold",
    "uvd7_attention_5fold",
    "uvd7_attention_focal_sampler_5fold",
    "uvd7_masked_attention_5fold",
    "uvd7_residual_masked_attention_5fold",
]
LABELS_4 = ["negative", "positive", "surprise", "others"]
LABELS_7 = ["disgust", "surprise", "others", "fear", "anger", "sad", "happy"]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_mean_std(row: dict[str, str], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return f"{mean:.4f} ± {std:.4f}"


def write_summary_tables(summary_rows: list[dict[str, str]]) -> None:
    for name, order in [("four_class_overall_results.csv", ORDER_4), ("seven_class_overall_results.csv", ORDER_7)]:
        rows_by_name = {row["run_name"]: row for row in summary_rows}
        with (OUT_DIR / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Model", "Input", "Accuracy", "Macro-F1", "UAR"])
            for run_name in order:
                row = rows_by_name[run_name]
                writer.writerow(
                    [
                        MODEL_LABELS[run_name],
                        row["input_mode"],
                        fmt_mean_std(row, "accuracy"),
                        fmt_mean_std(row, "macro_f1"),
                        fmt_mean_std(row, "uar"),
                    ]
                )


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.small{font-size:12px}.label{font-size:13px}.title{font-size:18px;font-weight:700}.legend{font-size:13px}</style>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def grouped_bar_chart(path: Path, rows: list[dict[str, str]], order: list[str], title: str) -> None:
    width, height = 1180, 560
    left, top, right, bottom = 76, 62, 40, 118
    plot_w = width - left - right
    plot_h = height - top - bottom
    metrics = [("accuracy", "Accuracy", "#4e79a7"), ("macro_f1", "Macro-F1", "#f28e2b"), ("uar", "UAR", "#59a14f")]
    rows_by_name = {row["run_name"]: row for row in rows}
    max_v = 0.60
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{width/2}" y="30" text-anchor="middle">{html.escape(title)}</text>')
    for i in range(7):
        y = top + plot_h - i * plot_h / 6
        value = max_v * i / 6
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    group_w = plot_w / len(order)
    bar_w = group_w * 0.18
    for idx, run_name in enumerate(order):
        cx = left + group_w * idx + group_w / 2
        row = rows_by_name[run_name]
        for m_idx, (metric, _, color) in enumerate(metrics):
            value = float(row[f"{metric}_mean"])
            bar_h = value / max_v * plot_h
            x = cx + (m_idx - 1) * bar_w * 1.25 - bar_w / 2
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle">{value:.3f}</text>')
        parts.append(
            f'<text class="small" x="{cx:.1f}" y="{top+plot_h+22}" text-anchor="middle">{html.escape(MODEL_LABELS[run_name])}</text>'
        )
    legend_x = left + 20
    for i, (_, label, color) in enumerate(metrics):
        x = legend_x + i * 140
        parts.append(f'<rect x="{x}" y="{height-40}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{x+22}" y="{height-28}">{label}</text>')
    write_svg(path, parts)


def heat_color(value: float, max_value: float) -> str:
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = min(max(value / max_value, 0.0), 1.0)
    # light blue to dark blue
    r = int(239 - ratio * 170)
    g = int(246 - ratio * 125)
    b = int(255 - ratio * 75)
    return f"#{r:02x}{g:02x}{b:02x}"


def confusion_heatmap(path: Path, matrix: list[list[int]], labels: list[str], title: str) -> None:
    n = len(labels)
    cell = 72 if n == 4 else 58
    left, top = 130, 78
    width = left + cell * n + 80
    height = top + cell * n + 115
    max_v = max(max(row) for row in matrix)
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{width/2}" y="30" text-anchor="middle">{html.escape(title)}</text>')
    parts.append(f'<text class="label" x="{left + cell*n/2}" y="58" text-anchor="middle">Predicted Label</text>')
    parts.append(f'<text class="label" x="24" y="{top + cell*n/2}" transform="rotate(-90 24,{top + cell*n/2})" text-anchor="middle">True Label</text>')
    for i, label in enumerate(labels):
        parts.append(f'<text class="small" x="{left + cell*i + cell/2}" y="{top-12}" text-anchor="middle">{html.escape(label)}</text>')
        parts.append(f'<text class="small" x="{left-12}" y="{top + cell*i + cell/2 + 4}" text-anchor="end">{html.escape(label)}</text>')
    for i, row in enumerate(matrix):
        row_sum = sum(row) or 1
        for j, value in enumerate(row):
            x = left + j * cell
            y = top + i * cell
            color = heat_color(value, max_v)
            pct = value / row_sum * 100
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(f'<text class="label" x="{x+cell/2}" y="{y+cell/2-3}" text-anchor="middle">{value}</text>')
            parts.append(f'<text class="small" x="{x+cell/2}" y="{y+cell/2+15}" text-anchor="middle">{pct:.1f}%</text>')
    write_svg(path, parts)


def class_metric_chart(path: Path, class_rows: list[dict[str, str]], run_names: list[str], labels: list[str], title: str) -> None:
    width, height = 1120, 520
    left, top, right, bottom = 78, 62, 40, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = 0.70
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    rows = {(row["run_name"], row["class"]): row for row in class_rows}
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{width/2}" y="30" text-anchor="middle">{html.escape(title)}</text>')
    for i in range(8):
        y = top + plot_h - i * plot_h / 7
        value = max_v * i / 7
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    group_w = plot_w / len(labels)
    bar_w = min(22, group_w / (len(run_names) + 1.5))
    for idx, label in enumerate(labels):
        cx = left + group_w * idx + group_w / 2
        for r_idx, run_name in enumerate(run_names):
            value = float(rows[(run_name, label)]["f1_mean"])
            bar_h = value / max_v * plot_h
            x = cx + (r_idx - (len(run_names)-1)/2) * bar_w * 1.25 - bar_w / 2
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[r_idx]}"/>')
            parts.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle">{value:.2f}</text>')
        parts.append(f'<text class="small" x="{cx:.1f}" y="{top+plot_h+24}" text-anchor="middle">{html.escape(label)}</text>')
    legend_x = left + 20
    for i, run_name in enumerate(run_names):
        x = legend_x + i * 200
        parts.append(f'<rect x="{x}" y="{height-42}" width="14" height="14" fill="{colors[i]}"/>')
        parts.append(f'<text class="legend" x="{x+22}" y="{height-30}">{html.escape(MODEL_LABELS[run_name])}</text>')
    write_svg(path, parts)


def ablation_rows(summary_rows: list[dict[str, str]], order: list[str]) -> list[tuple[str, str, str, float, float]]:
    rows = {row["run_name"]: row for row in summary_rows}

    def mf1(name: str) -> float:
        return float(rows[name]["macro_f1_mean"])

    def uar(name: str) -> float:
        return float(rows[name]["uar_mean"])

    if order is ORDER_4:
        uv, depth, concat, attention, focal, masked, residual = ORDER_4
    else:
        uv, depth, concat, attention, focal, masked, residual = ORDER_7
    return [
        ("Depth vs UV", MODEL_LABELS[depth], MODEL_LABELS[uv], mf1(depth) - mf1(uv), uar(depth) - uar(uv)),
        ("Concat vs UV", MODEL_LABELS[concat], MODEL_LABELS[uv], mf1(concat) - mf1(uv), uar(concat) - uar(uv)),
        ("Attention vs Concat", MODEL_LABELS[attention], MODEL_LABELS[concat], mf1(attention) - mf1(concat), uar(attention) - uar(concat)),
        ("Masked vs Attention", MODEL_LABELS[masked], MODEL_LABELS[attention], mf1(masked) - mf1(attention), uar(masked) - uar(attention)),
        ("Residual-Masked vs Attention", MODEL_LABELS[residual], MODEL_LABELS[attention], mf1(residual) - mf1(attention), uar(residual) - uar(attention)),
        ("Focal+Sampler vs Attention", MODEL_LABELS[focal], MODEL_LABELS[attention], mf1(focal) - mf1(attention), uar(focal) - uar(attention)),
    ]


def write_ablation_tables(summary_rows: list[dict[str, str]]) -> None:
    for filename, order in [("four_class_ablation_summary.csv", ORDER_4), ("seven_class_ablation_summary.csv", ORDER_7)]:
        with (OUT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Comparison", "Compared Model", "Reference Model", "Macro-F1 Delta", "UAR Delta"])
            for comparison, compared, reference, macro_delta, uar_delta in ablation_rows(summary_rows, order):
                writer.writerow([comparison, compared, reference, f"{macro_delta:+.4f}", f"{uar_delta:+.4f}"])


def ablation_delta_chart(path: Path, summary_rows: list[dict[str, str]]) -> None:
    width, height = 1180, 600
    left, top, right, bottom = 86, 62, 40, 145
    plot_w = width - left - right
    plot_h = height - top - bottom
    items4 = ablation_rows(summary_rows, ORDER_4)
    items7 = ablation_rows(summary_rows, ORDER_7)
    labels = [item[0] for item in items4]
    max_abs = 0.14
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{width/2}" y="30" text-anchor="middle">Macro-F1 Delta in Ablation Comparisons</text>')
    zero_y = top + plot_h / 2
    for i in range(-3, 4):
        value = i * max_abs / 3
        y = zero_y - value / max_abs * plot_h / 2
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:+.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}"/>')
    group_w = plot_w / len(labels)
    bar_w = group_w * 0.20
    colors = ["#4e79a7", "#f28e2b"]
    for idx, label in enumerate(labels):
        cx = left + group_w * idx + group_w / 2
        for j, items in enumerate([items4, items7]):
            value = items[idx][3]
            h = abs(value) / max_abs * plot_h / 2
            x = cx + (j - 0.5) * bar_w * 1.25 - bar_w / 2
            y = zero_y - h if value >= 0 else zero_y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[j]}"/>')
            ty = y - 5 if value >= 0 else y + h + 14
            parts.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{ty:.1f}" text-anchor="middle">{value:+.3f}</text>')
        parts.append(
            f'<text class="small" x="{cx:.1f}" y="{top+plot_h+24}" text-anchor="middle" transform="rotate(22 {cx:.1f},{top+plot_h+24})">{html.escape(label)}</text>'
        )
    for i, text in enumerate(["Four-Class", "Seven-Class"]):
        x = left + 20 + i * 160
        parts.append(f'<rect x="{x}" y="{height-40}" width="14" height="14" fill="{colors[i]}"/>')
        parts.append(f'<text class="legend" x="{x+22}" y="{height-28}">{text}</text>')
    write_svg(path, parts)


def read_history(run_name: str) -> dict[int, dict[str, list[float]]]:
    data: dict[int, dict[str, list[float]]] = {}
    for path in sorted((PROJECT_ROOT / "artifacts" / "runs" / run_name).glob("fold_*/history.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                epoch = int(row["epoch"])
                bucket = data.setdefault(epoch, {key: [] for key in ["train_loss", "val_loss", "train_macro_f1", "val_macro_f1"]})
                for key in bucket:
                    bucket[key].append(float(row[key]))
    return data


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    chunks = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    chunks.extend(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
    return " ".join(chunks)


def training_curve(path: Path, run_name: str, title: str) -> None:
    hist = read_history(run_name)
    epochs = sorted(hist)
    width, height = 1120, 520
    left, top, right, bottom = 70, 62, 40, 72
    gap = 64
    panel_w = (width - left - right - gap) / 2
    panel_h = height - top - bottom
    parts = svg_header(width, height)
    parts.append(f'<text class="title" x="{width/2}" y="30" text-anchor="middle">{html.escape(title)}</text>')

    def draw_panel(x0: float, keys: list[str], labels: list[str], colors: list[str], y_max: float, subtitle: str) -> None:
        parts.append(f'<text class="label" x="{x0+panel_w/2:.1f}" y="{top-18}" text-anchor="middle">{subtitle}</text>')
        for i in range(6):
            y = top + panel_h - i * panel_h / 5
            value = y_max * i / 5
            parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}"/>')
            parts.append(f'<text class="small" x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{top+panel_h}"/>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{top+panel_h}" x2="{x0+panel_w}" y2="{top+panel_h}"/>')
        max_epoch = max(epochs)
        for key, color in zip(keys, colors):
            pts = []
            for epoch in epochs:
                value = mean(hist[epoch][key])
                x = x0 + (epoch - 1) / max(1, max_epoch - 1) * panel_w
                y = top + panel_h - min(value, y_max) / y_max * panel_h
                pts.append((x, y))
            parts.append(f'<path d="{line_path(pts)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, (label, color) in enumerate(zip(labels, colors)):
            x = x0 + 16 + i * 120
            parts.append(f'<line x1="{x}" y1="{top+panel_h+42}" x2="{x+24}" y2="{top+panel_h+42}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text class="legend" x="{x+32}" y="{top+panel_h+46}">{label}</text>')

    draw_panel(left, ["train_loss", "val_loss"], ["Train", "Val"], ["#4e79a7", "#f28e2b"], 2.2, "Loss")
    draw_panel(left + panel_w + gap, ["train_macro_f1", "val_macro_f1"], ["Train", "Val"], ["#4e79a7", "#f28e2b"], 0.75, "Macro-F1")
    write_svg(path, parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv_dicts(ANALYSIS_DIR / "scene_flow_run_summary.csv")
    class_rows = read_csv_dicts(ANALYSIS_DIR / "scene_flow_run_class_metrics.csv")
    confusion = json.loads((ANALYSIS_DIR / "scene_flow_run_confusion_matrices.json").read_text(encoding="utf-8"))

    write_summary_tables(summary_rows)
    write_ablation_tables(summary_rows)
    grouped_bar_chart(OUT_DIR / "fig4_1_four_class_overall_metrics.svg", summary_rows, ORDER_4, "Four-Class Overall Metrics")
    grouped_bar_chart(OUT_DIR / "fig4_3_seven_class_overall_metrics.svg", summary_rows, ORDER_7, "Seven-Class Overall Metrics")
    class_metric_chart(
        OUT_DIR / "fig4_2_four_class_f1_comparison.svg",
        class_rows,
        ["uv4_baseline_5fold", "uvd4_attention_5fold", "uvd4_residual_masked_attention_5fold"],
        LABELS_4,
        "Four-Class Class-Wise F1 Comparison",
    )
    class_metric_chart(
        OUT_DIR / "fig4_4_seven_class_f1_comparison.svg",
        class_rows,
        ["uv7_baseline_5fold", "uvd7_attention_5fold", "uvd7_residual_masked_attention_5fold"],
        LABELS_7,
        "Seven-Class Class-Wise F1 Comparison",
    )
    confusion_heatmap(
        OUT_DIR / "fig4_5_four_class_uvd_attention_confusion.svg",
        confusion["uvd4_attention_5fold"],
        LABELS_4,
        "Four-Class UVD-Attention Confusion Matrix",
    )
    confusion_heatmap(
        OUT_DIR / "fig4_6_seven_class_residual_masked_confusion.svg",
        confusion["uvd7_residual_masked_attention_5fold"],
        LABELS_7,
        "Seven-Class Residual-Masked Confusion Matrix",
    )
    ablation_delta_chart(OUT_DIR / "fig4_7_ablation_macro_f1_delta.svg", summary_rows)
    training_curve(
        OUT_DIR / "fig4_8_uvd4_attention_training_curve.svg",
        "uvd4_attention_5fold",
        "Training Curves of Four-Class UVD-Attention Model",
    )
    print(f"Wrote thesis chapter 4 assets to: {OUT_DIR}")


if __name__ == "__main__":
    main()
