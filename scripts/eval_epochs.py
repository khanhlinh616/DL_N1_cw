"""
Danh gia va luu ma tran COCO evaluation cho tung epoch.
Tu dong:
  - Kiem tra checkpoint hop le truoc khi chay
  - Chay `python -m src.evaluate --dataset heldout` va in truc tiep ra terminal
  - Luu raw output ra `outputs/evaluation_results/epoch_<N>_metrics.txt`
  - Parse 12 chi so COCO ra `outputs/evaluation_results/epoch_<N>_metrics.json`
  - Luu ban sao predictions ra `outputs/evaluation_results/predictions_heldout_epoch_<N>.json`
  - Goi `scripts/visualize_metrics.py` de cap nhat bieu do va bao cao

Su dung:
  $env:PYTHONPATH="."; python scripts/eval_epochs.py
  $env:PYTHONPATH="."; python scripts/eval_epochs.py --epochs 1 2 3 --max-images 1000
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


EVAL_DIR = Path("outputs/evaluation_results")
CKPT_DIR = Path("outputs/checkpoints")

COCO_PATTERNS: List[Tuple[str, str]] = [
    ("mAP_50_95", r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \] =\s*([0-9.]+)"),
    ("mAP_50", r"Average Precision\s+\(AP\) @\[ IoU=0\.50\s+\| area=\s*all \| maxDets=100 \] =\s*([0-9.]+)"),
    ("mAP_75", r"Average Precision\s+\(AP\) @\[ IoU=0\.75\s+\| area=\s*all \| maxDets=100 \] =\s*([0-9.]+)"),
    ("mAP_s", r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*small \| maxDets=100 \] =\s*([0-9.]+)"),
    ("mAP_m", r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=medium \| maxDets=100 \] =\s*([0-9.]+)"),
    ("mAP_l", r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*large \| maxDets=100 \] =\s*([0-9.]+)"),
    ("AR_1", r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=\s*1 \] =\s*([0-9.]+)"),
    ("AR_10", r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=\s*10 \] =\s*([0-9.]+)"),
    ("AR_100", r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \] =\s*([0-9.]+)"),
    ("AR_s", r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s*small \| maxDets=100 \] =\s*([0-9.]+)"),
    ("AR_m", r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=medium \| maxDets=100 \] =\s*([0-9.]+)"),
    ("AR_l", r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s*large \| maxDets=100 \] =\s*([0-9.]+)"),
]

SUMMARY_PATTERNS: List[Tuple[str, str]] = [
    ("mAP_50_95", r"mAP@\[\.50:\.95\]\s*:\s*([0-9.]+)"),
    ("mAP_50", r"mAP@\.50\s*:\s*([0-9.]+)"),
    ("mAP_75", r"mAP@\.75\s*:\s*([0-9.]+)"),
    ("mAP_s", r"mAP \(small\)\s*:\s*([0-9.]+)"),
    ("mAP_m", r"mAP \(medium\)\s*:\s*([0-9.]+)"),
    ("mAP_l", r"mAP \(large\)\s*:\s*([0-9.]+)"),
    ("AR_100", r"AR@100\s*:\s*([0-9.]+)"),
]


def parse_coco_output(raw_text: str) -> Dict[str, float]:
    """Trich xuat 12 chi so COCO (uu tien do chinh xac 4 chu so tu Summary neu co)."""
    metrics: Dict[str, float] = {}
    for key, pat in COCO_PATTERNS:
        m = re.search(pat, raw_text)
        if m:
            metrics[key] = float(m.group(1))
    for key, pat in SUMMARY_PATTERNS:
        m = re.search(pat, raw_text)
        if m:
            metrics[key] = float(m.group(1))
    m_pred = re.search(r"Saved\s+(\d+)\s+predictions", raw_text)
    if m_pred:
        metrics["num_predictions"] = int(m_pred.group(1))
    return metrics


def save_epoch_metrics(epoch: int, checkpoint: str, raw_text: str, max_images: int = 1000) -> Path:
    """Luu file .txt va .json cho 1 epoch."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = EVAL_DIR / f"epoch_{epoch}_metrics.txt"
    json_path = EVAL_DIR / f"epoch_{epoch}_metrics.json"

    header = (
        f"# Epoch {epoch} Evaluation Matrix (val-heldout1k)\n"
        f"# Checkpoint: {checkpoint}\n"
        f"# Timestamp:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 65 + "\n"
    )
    txt_path.write_text(header + raw_text.strip() + "\n", encoding="utf-8")

    parsed = parse_coco_output(raw_text)
    payload = {
        "epoch": epoch,
        "checkpoint": checkpoint,
        "dataset": "heldout",
        "max_images": max_images,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": parsed,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {txt_path}")
    print(f"[SAVED] {json_path}")
    return json_path


def check_checkpoint(ckpt_path: Path) -> bool:
    """Kiem tra checkpoint co ton tai va load duoc khong."""
    if not ckpt_path.exists():
        print(f"  [SKIP] Khong tim thay: {ckpt_path}")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        is_interim = isinstance(ckpt, dict) and ckpt.get("batch_idx") is not None
        if is_interim:
            print(f"  [WARN] {ckpt_path.name} la interim checkpoint (batch {ckpt['batch_idx'] + 1}), chua xong epoch.")
        else:
            print(f"  [OK] {ckpt_path.name} hop le.")
        return True
    except Exception as exc:
        print(f"  [CORRUPT] {ckpt_path.name} khong doc duoc: {exc}")
        return False


def run_eval_for_epoch(epoch: int, max_images: int, device: str, force: bool = False) -> Optional[Path]:
    ckpt_path = CKPT_DIR / f"epoch_{epoch}.pth"
    json_path = EVAL_DIR / f"epoch_{epoch}_metrics.json"
    if json_path.exists() and not force:
        print(f"[INFO] Da co {json_path}, bo qua (dung --force de chay lai).")
        return json_path

    print(f"\n{'=' * 65}\n  Checking EPOCH {epoch}: {ckpt_path}\n{'=' * 65}")
    if not check_checkpoint(ckpt_path):
        return None

    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--checkpoint", str(ckpt_path),
        "--dataset", "heldout",
        "--max-images", str(max_images),
        "--device", device,
    ]
    lines: List[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()

    raw_text = "".join(lines)
    if rc != 0:
        print(f"[WARN] src.evaluate exit code {rc} cho epoch {epoch}")
        return None

    pred_src = EVAL_DIR / "predictions_heldout.json"
    if pred_src.exists():
        shutil.copy2(pred_src, EVAL_DIR / f"predictions_heldout_epoch_{epoch}.json")

    return save_epoch_metrics(epoch, str(ckpt_path), raw_text, max_images=max_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and save COCO matrices")
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true", help="Chay lai ngay ca khi da co file metrics")
    args = parser.parse_args()

    for ep in args.epochs:
        run_eval_for_epoch(ep, max_images=args.max_images, device=args.device, force=args.force)

    # Cap nhat bieu do va bao cao tong hop
    viz_script = Path(__file__).parent / "visualize_metrics.py"
    if viz_script.exists():
        subprocess.run([sys.executable, str(viz_script)], check=False)


if __name__ == "__main__":
    main()
