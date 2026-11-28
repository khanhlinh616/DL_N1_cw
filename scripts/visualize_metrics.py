"""
Truc quan hoa qua trinh training & ma tran COCO qua tung epoch.
Doc tu:
  - outputs/logs/train_history.json
  - outputs/evaluation_results/epoch_*_metrics.json

Tao ra:
  1. outputs/visualizations/metrics/loss_curves.png
  2. outputs/visualizations/metrics/coco_ap_ar_trends.png
  3. outputs/visualizations/metrics/coco_matrix_heatmap.png
  4. outputs/evaluation_results/epochs_matrix_report.md

Chay: $env:PYTHONPATH="."; python scripts/visualize_metrics.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOG_FILE = Path("outputs/logs/train_history.json")
ABLATION_FILE = Path("outputs/logs/ablation_results.json")
EVAL_DIR = Path("outputs/evaluation_results")
OUT_VIZ_DIR = Path("outputs/visualizations/metrics")
REPORT_MD = Path("docs/epoch-matrix.md")

METRIC_ORDER = [
    ("mAP_50_95", "mAP@[.50:.95]"),
    ("mAP_50", "mAP@.50"),
    ("mAP_75", "mAP@.75"),
    ("mAP_s", "AP (small)"),
    ("mAP_m", "AP (medium)"),
    ("mAP_l", "AP (large)"),
    ("AR_1", "AR@1"),
    ("AR_10", "AR@10"),
    ("AR_100", "AR@100"),
    ("AR_s", "AR (small)"),
    ("AR_m", "AR (medium)"),
    ("AR_l", "AR (large)"),
]


def load_train_history() -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    return sorted(data, key=lambda x: int(x["epoch"]))


def load_epoch_evals() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in EVAL_DIR.glob("epoch_*_metrics.json"):
        m = re.match(r"epoch_(\d+)_metrics\.json$", p.name)
        if m:
            items.append(json.loads(p.read_text(encoding="utf-8")))
    return sorted(items, key=lambda x: int(x["epoch"]))


def plot_loss_curves(history: List[Dict[str, Any]], out_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    lrs = [h.get("lr", 0.0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    specs = [
        ("loss_total", "Total Loss (Train vs Val)", axes[0, 0]),
        ("loss_box", "Box CIoU Loss", axes[0, 1]),
        ("loss_obj", "Objectness Focal Loss", axes[1, 0]),
        ("loss_cls", "Classification BCE Loss", axes[1, 1]),
    ]
    for key, title, ax in specs:
        tr = [h["train"][key] for h in history]
        va = [h["val"][key] for h in history]
        ax.plot(epochs, tr, marker="o", linewidth=2.2, color="#2563EB", label="Train")
        ax.plot(epochs, va, marker="s", linewidth=2.2, color="#DC2626", label="Val (heldout 1k)")
        for x, y_tr, y_va in zip(epochs, tr, va):
            ax.annotate(f"{y_tr:.3f}", (x, y_tr), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#1D4ED8")
            ax.annotate(f"{y_va:.3f}", (x, y_va), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=8, color="#B91C1C")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=True, fontsize=8.5)

    lr_text = " | ".join([f"Ep{e}: LR={lr:.0e}" for e, lr in zip(epochs, lrs)])
    fig.suptitle(f"Training & Validation Loss Curves ({lr_text})", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_coco_trends(evals: List[Dict[str, Any]], out_path: Path) -> None:
    if not evals:
        return
    epochs = [e["epoch"] for e in evals]
    x = np.arange(len(epochs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    # Panel 1: Main AP & AR bar + line
    bar_keys = [
        ("mAP_50_95", "mAP@[.50:.95]", "#2563EB"),
        ("mAP_50", "mAP@.50", "#16A34A"),
        ("mAP_75", "mAP@.75", "#7C3AED"),
        ("AR_100", "AR@100", "#EA580C"),
    ]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(bar_keys))
    for off, (k, lbl, col) in zip(offsets, bar_keys):
        vals = [e["metrics"].get(k, 0.0) for e in evals]
        bars = ax1.bar(x + off, vals, width=width, label=lbl, color=col, alpha=0.9)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Epoch {ep}" for ep in epochs], fontsize=10)
    ax1.set_ylim(0, max([e["metrics"].get("AR_100", 0.25) for e in evals] + [0.25]) * 1.25)
    ax1.set_title("Core COCO Metrics (val-heldout 1k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.5)

    # Panel 2: Size breakdown (AP & AR small/medium/large)
    size_keys = [
        ("mAP_s", "AP small", "#0284C7", "-o"),
        ("mAP_m", "AP medium", "#0D9488", "-o"),
        ("mAP_l", "AP large", "#15803D", "-o"),
        ("AR_s", "AR small", "#F59E0B", "--s"),
        ("AR_m", "AR medium", "#EA580C", "--s"),
        ("AR_l", "AR large", "#DC2626", "--s"),
    ]
    for k, lbl, col, style in size_keys:
        vals = [e["metrics"].get(k, 0.0) for e in evals]
        ax2.plot(epochs, vals, style, color=col, linewidth=2.0, markersize=6, label=lbl)
        if len(epochs) >= 1:
            ax2.annotate(f"{vals[-1]:.3f}", (epochs[-1], vals[-1]),
                         textcoords="offset points", xytext=(6, 0), va="center", fontsize=7.5, color=col)
    ax2.set_xticks(epochs)
    ax2.set_xticklabels([f"Epoch {ep}" for ep in epochs], fontsize=10)
    ax2.set_xlim(min(epochs) - 0.3, max(epochs) + 0.55)
    ax2.set_title("Scale Breakdown (Small / Medium / Large)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8, ncol=2, loc="upper left")

    fig.suptitle("COCO Evaluation Progression across Epochs (val-heldout 1k)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_heatmap(evals: List[Dict[str, Any]], out_path: Path) -> None:
    if not evals:
        return
    epochs = [e["epoch"] for e in evals]
    mat = np.array([[e["metrics"].get(k, 0.0) for e in evals] for k, _ in METRIC_ORDER])

    col_labels = [f"Epoch {ep}" for ep in epochs]
    if len(evals) >= 2:
        delta = mat[:, -1] - mat[:, 0]
        col_labels.append(f"Delta (Ep{epochs[-1]}-Ep{epochs[0]})")

    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(col_labels) + 2.5), 6.2), dpi=150)
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(0.40, float(mat.max()) * 1.05))

    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels([f"Epoch {ep}" for ep in epochs], fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(METRIC_ORDER)))
    ax.set_yticklabels([lbl for _, lbl in METRIC_ORDER], fontsize=9.5, fontweight="bold")

    thresh = mat.max() * 0.58 if mat.max() > 0 else 0.5
    for i in range(len(METRIC_ORDER)):
        for j in range(len(epochs)):
            val = mat[i, j]
            txt_col = "white" if val > thresh else "#111827"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=txt_col, fontsize=9.5, fontweight="bold")

    if len(evals) >= 2:
        ax.set_xlim(-0.5, len(epochs) + 0.55)
        for i in range(len(METRIC_ORDER)):
            d = mat[i, -1] - mat[i, 0]
            sign = "+" if d >= 0 else ""
            col = "#15803D" if d > 0 else ("#DC2626" if d < 0 else "#4B5563")
            ax.text(len(epochs) + 0.02, i, f"{sign}{d:.4f}", ha="center", va="center",
                    color=col, fontsize=9, fontweight="bold")
        ax.text(len(epochs) + 0.02, -0.75, f"Ep{epochs[-1]}-Ep{epochs[0]}",
                ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1F2937")

    ax.set_title("COCO 12-Metric Matrix (val-heldout 1k, conf>=0.05, NMS=0.60)",
                 fontsize=11.5, fontweight="bold", pad=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Score", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(history: List[Dict[str, Any]], evals: List[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# Ma trận Đánh giá & Trực quan hóa qua từng Epoch",
        "",
        "> Nguồn dữ liệu thực nghiệm: `outputs/logs/train_history.json` và `outputs/evaluation_results/epoch_*_metrics.json` (`val-heldout 1k`, 1.000 ảnh).",
        "",
        "## 1. Biểu đồ Trực quan hóa",
        "",
        "![Loss Curves](../outputs/visualizations/metrics/loss_curves.png)",
        "",
        "![COCO Trends](../outputs/visualizations/metrics/coco_ap_ar_trends.png)",
        "",
        "![COCO Matrix Heatmap](../outputs/visualizations/metrics/coco_matrix_heatmap.png)",
        "",
        "## 2. Bảng Loss qua các Epoch",
        "",
        "| Epoch | LR | Train Total | Train Box | Train Obj | Train Cls | Val Total | Val Box | Val Obj | Val Cls |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for h in history:
        tr, va = h["train"], h["val"]
        lines.append(
            f"| {h['epoch']} | `{h.get('lr', 0):.0e}` | "
            f"{tr['loss_total']:.4f} | {tr['loss_box']:.4f} | {tr['loss_obj']:.4f} | {tr['loss_cls']:.4f} | "
            f"**{va['loss_total']:.4f}** | {va['loss_box']:.4f} | {va['loss_obj']:.4f} | {va['loss_cls']:.4f} |"
        )

    lines += [
        "",
        "## 3. Ma trận 12 chỉ số COCO (`val-heldout 1k`)",
        "",
    ]
    if evals:
        ep_cols = " | ".join([f"Epoch {e['epoch']}" for e in evals])
        sep_cols = " | ".join([":---:" for _ in evals])
        delta_hdr = f" | Delta (Ep{evals[-1]['epoch']}-Ep{evals[0]['epoch']})" if len(evals) >= 2 else ""
        delta_sep = " | :---:" if len(evals) >= 2 else ""
        lines.append(f"| Metric | {ep_cols}{delta_hdr} |")
        lines.append(f"|:---| {sep_cols}{delta_sep} |")
        for k, label in METRIC_ORDER:
            vals = [e["metrics"].get(k, 0.0) for e in evals]
            row_vals = " | ".join([f"{v:.4f}" for v in vals])
            delta_cell = f" | {vals[-1] - vals[0]:+.4f}" if len(evals) >= 2 else ""
            lines.append(f"| **{label}** | {row_vals}{delta_cell} |")
    else:
        lines.append("_Chưa có dữ liệu eval nào._")

    lines += [
        "",
        "## 4. Ảnh dự đoán minh họa & Lưu ý tái tạo",
        "",
        "- **Ảnh dự đoán Epoch 3** (khung xanh: Ground Truth, khung đỏ: Prediction): `outputs/visualizations/eval_samples_epoch3/`",
        "- **Cập nhật lại bảng & biểu đồ sau khi train thêm epoch:**",
        "",
        "```powershell",
        '$env:PYTHONPATH="."',
        "python scripts/eval_epochs.py --epochs 1 2 3 4 5",
        "```",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {REPORT_MD}")


def load_ablation_results() -> Dict[str, Any]:
    if not ABLATION_FILE.exists():
        return {}
    return json.loads(ABLATION_FILE.read_text(encoding="utf-8"))


def plot_m7_ablation(ab_data: Dict[str, Any], out_path: Path) -> None:
    if not ab_data:
        return
    base = ab_data.get("baseline", {})
    ab1 = ab_data.get("ablation_1_box_loss", {})
    ab2 = ab_data.get("ablation_2_obj_loss", {})
    ab3 = ab_data.get("ablation_3_fpn_scale", {})

    labels = [
        "Baseline\n(CIoU + Focal + FPN)",
        "Ablation 1\n(Smooth L1 Box)",
        "Ablation 2\n(BCE Obj)",
        "Ablation 3\n(Single P5 only)",
    ]
    rows = [base, ab1, ab2, ab3]
    totals = [float(r.get("loss_total", 0.0)) for r in rows]
    objs = [float(r.get("loss_obj", 0.0)) for r in rows]
    colors = ["#2563EB", "#0D9488", "#DC2626", "#D97706"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.4), dpi=150, gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # Left: Bar chart Loss Total vs Loss Obj
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax1.bar(x - w / 2, totals, width=w, color=colors, alpha=0.92, label="Loss Total")
    b2 = ax1.bar(x + w / 2, objs, width=w, color=colors, alpha=0.48, hatch="//", edgecolor="#1F2937", label="Loss Obj")
    for bar in b1:
        v = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 10, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b2:
        v = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 10, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, max(totals + [650]) * 1.15)
    ax1.set_ylabel("Loss (trên 100 ảnh val-heldout)")
    ax1.set_title("So sánh Loss Total & Loss Objectness (M7)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.5, loc="upper left")

    # Right: Visual comparison table
    ax2.axis("off")
    ax2.set_title("Bảng tổng hợp Ablation Studies (M7 — 100 ảnh held-out)", fontsize=11, fontweight="bold", pad=10)
    base_tot = float(base.get("loss_total", 1.0))
    base_obj = float(base.get("loss_obj", 1.0))

    def fmt_opt(d: Dict[str, Any], key: str) -> str:
        return f"{d[key]:.4f}" if key in d else "N/A"

    table_rows = [
        ["Baseline\n(CIoU+Focal+FPN)", fmt_opt(base, "loss_total"), fmt_opt(base, "loss_box"), fmt_opt(base, "loss_obj"), "Chuẩn (0%)", "Pipeline chuẩn đa tầng"],
        ["Ablation 1\n(Smooth L1)", fmt_opt(ab1, "loss_total"), fmt_opt(ab1, "loss_box"), fmt_opt(ab1, "loss_obj"), f"{(totals[1]-base_tot)/base_tot*100:+.1f}%", "Thiếu hình học 2D & tỷ lệ cạnh"],
        ["Ablation 2\n(BCE Obj)", fmt_opt(ab2, "loss_total"), "N/A", fmt_opt(ab2, "loss_obj"), f"{(totals[2]-base_tot)/base_tot*100:+.1f}%", f"Obj tăng +{(objs[2]-base_obj)/base_obj*100:.1f}% do âm nền"],
        ["Ablation 3\n(Chỉ P5)", fmt_opt(ab3, "loss_total"), "N/A", fmt_opt(ab3, "loss_obj"), f"{(totals[3]-base_tot)/base_tot*100:+.1f}%", "Chỉ còn 256/5376 ô (bỏ P3, P4)"],
    ]
    col_labels = ["Cấu hình", "Total", "Box", "Obj", "Δ Total", "Nhận xét kỹ thuật"]
    tbl = ax2.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.11, 0.10, 0.11, 0.11, 0.35],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1.0, 2.3)

    row_bg = ["#EFF6FF", "#F0FDFA", "#FEF2F2", "#FFFBEB"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(row_bg[r - 1])
            if c == 0:
                cell.set_text_props(fontweight="bold")

    fig.suptitle("Milestone M7: Trực quan hóa Kết quả Ablation Studies (val-heldout 1k)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    history = load_train_history()
    evals = load_epoch_evals()
    ab_data = load_ablation_results()

    if history:
        p1 = OUT_VIZ_DIR / "loss_curves.png"
        plot_loss_curves(history, p1)
        print(f"[SAVED] {p1}")
    if evals:
        p2 = OUT_VIZ_DIR / "coco_ap_ar_trends.png"
        p3 = OUT_VIZ_DIR / "coco_matrix_heatmap.png"
        plot_coco_trends(evals, p2)
        plot_matrix_heatmap(evals, p3)
        print(f"[SAVED] {p2}")
        print(f"[SAVED] {p3}")
    if ab_data:
        p4 = OUT_VIZ_DIR / "m7_ablation_comparison.png"
        plot_m7_ablation(ab_data, p4)
        print(f"[SAVED] {p4}")

    write_markdown_report(history, evals)


if __name__ == "__main__":
    main()
