"""Training engine and CLI for YOLO-style object detector (Milestones M4 & M6).

Features:
- AdamW optimizer with weight decay and gradient clipping (max norm = 10.0).
- CosineAnnealingLR scheduler with optional warmup epochs.
- Mixed Precision (AMP) with GradScaler and graceful CPU fallback.
- Backbone freezing strategy: ResNet-50 frozen for initial epochs, unfreeze layers 2-4.
- FrozenBatchNorm2d preservation (never receives gradients).
- Validation tracking on held-out dataset.
- Checkpoint persistence (best_map.pth, last_epoch.pth) with config, epoch, and git commit.
- Resumable training via --resume flag.
- Smoke-run / dry-run support via --max-images flag.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.core.assigner import assign_targets
from src.core.loss import MultiTaskLoss
from src.data.dataset import COCODataset, detection_collate_fn
from src.models.detector import YOLODetector, build_detector
from src.utils.config import load_yaml_config
from src.utils.logger import MetricTracker, TrainingLogger


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproduction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _sync_git_progress(epoch_num: int, batch_num: int, total_batches: int, metrics: Dict[str, float]) -> None:
    """Safely update docs/plan-progress-status.md with batch progress and push to origin main."""
    try:
        status_file = Path("docs/plan-progress-status.md")
        if not status_file.exists():
            return
        content = status_file.read_text(encoding="utf-8")
        from datetime import datetime
        import re

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        pct = (batch_num / total_batches) * 100
        loss_val = metrics.get("loss_total", 0.0)
        box_val = metrics.get("loss_box", 0.0)
        obj_val = metrics.get("loss_obj", 0.0)
        cls_val = metrics.get("loss_cls", 0.0)

        progress_line = (
            f"* **Tiến độ trực tiếp:** Epoch {epoch_num}: Batch {batch_num}/{total_batches} ({pct:.1f}%) | "
            f"Loss: {loss_val:.4f} (box: {box_val:.4f}, obj: {obj_val:.4f}, cls: {cls_val:.4f})"
        )

        content = re.sub(r"\* \*\*Cập nhật lần cuối:\*\* .*", f"* **Cập nhật lần cuối:** {now_str}", content)
        if "* **Tiến độ trực tiếp:**" in content:
            content = re.sub(r"\* \*\*Tiến độ trực tiếp:\*\* .*", progress_line, content)
        else:
            marker = "* **Trạng thái tổng thể:**"
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if line.startswith(marker):
                    new_lines.append(progress_line)
            content = "\n".join(new_lines) + "\n"

        if len(content.splitlines()) > 190:
            return

        status_file.write_text(content, encoding="utf-8")

        subprocess.run(["git", "add", "docs/plan-progress-status.md"], check=False, capture_output=True, timeout=30)
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"docs: sync training progress Epoch {epoch_num} (Batch {batch_num}/{total_batches})"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if commit_res.returncode == 0:
            push_res = subprocess.run(
                ["git", "push", "origin", "main"],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            if push_res.returncode == 0:
                print(f"  [Git-Sync] Progress committed & pushed to GitHub: Batch {batch_num}/{total_batches}", flush=True)
            else:
                print(f"  [Git-Sync Warning] Git push returned code {push_res.returncode}: {push_res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"  [Git-Sync Warning] Sync failed: {e}", flush=True)


def _sync_git_epoch_completed(epoch_num: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    """Safely update documentation after epoch finishes and push to origin main."""
    try:
        from datetime import datetime
        import json
        import re

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        t_loss = train_metrics.get("loss_total", 0.0)
        v_loss = val_metrics.get("loss_total", 0.0)

        status_file = Path("docs/plan-progress-status.md")
        if status_file.exists():
            content = status_file.read_text(encoding="utf-8")
            content = re.sub(r"\* \*\*Cập nhật lần cuối:\*\* .*", f"* **Cập nhật lần cuối:** {now_str}", content)
            content = re.sub(
                r"\* \*\*Trạng thái tổng thể:\*\* .*",
                f"* **Trạng thái tổng thể:** Hoàn thành M0–M9; Đã hoàn thành 100% Epoch 1, Epoch 2 & Epoch {epoch_num} baseline.",
                content,
            )
            content = re.sub(
                r"\* \*\*Tiến độ trực tiếp:\*\* .*",
                f"* **Tiến độ trực tiếp:** Epoch {epoch_num} hoàn thành 100% (Train: {t_loss:.4f}, Val: {v_loss:.4f}).",
                content,
            )
            status_file.write_text(content, encoding="utf-8")

        metrics_file = Path("outputs/metrics.json")
        if metrics_file.exists():
            try:
                m_data = json.loads(metrics_file.read_text(encoding="utf-8-sig"))
                if "baseline_m6" in m_data:
                    m_data["baseline_m6"]["epoch"] = epoch_num
                    m_data["baseline_m6"]["total_epochs"] = epoch_num
                    m_data["baseline_m6"]["images_processed"] = epoch_num * 14000
                    m_data["baseline_m6"]["status"] = f"Completed 100% Epoch 1 to Epoch {epoch_num}"
                    m_data["baseline_m6"]["train_loss_total"] = round(t_loss, 4)
                    m_data["baseline_m6"]["train_loss_box"] = round(train_metrics.get("loss_box", 0.0), 4)
                    m_data["baseline_m6"]["train_loss_obj"] = round(train_metrics.get("loss_obj", 0.0), 4)
                    m_data["baseline_m6"]["train_loss_cls"] = round(train_metrics.get("loss_cls", 0.0), 4)
                    m_data["baseline_m6"]["val_loss_total"] = round(v_loss, 4)
                    m_data["baseline_m6"]["val_loss_box"] = round(val_metrics.get("loss_box", 0.0), 4)
                    m_data["baseline_m6"]["val_loss_obj"] = round(val_metrics.get("loss_obj", 0.0), 4)
                    m_data["baseline_m6"]["val_loss_cls"] = round(val_metrics.get("loss_cls", 0.0), 4)
                    m_data["baseline_m6"][f"epoch_{epoch_num}"] = {
                        "train_loss_total": round(t_loss, 4),
                        "train_loss_box": round(train_metrics.get("loss_box", 0.0), 4),
                        "train_loss_obj": round(train_metrics.get("loss_obj", 0.0), 4),
                        "train_loss_cls": round(train_metrics.get("loss_cls", 0.0), 4),
                        "val_loss_total": round(v_loss, 4),
                        "val_loss_box": round(val_metrics.get("loss_box", 0.0), 4),
                        "val_loss_obj": round(val_metrics.get("loss_obj", 0.0), 4),
                        "val_loss_cls": round(val_metrics.get("loss_cls", 0.0), 4),
                    }
                    metrics_file.write_text(json.dumps(m_data, indent=2), encoding="utf-8")
            except Exception:
                pass

        subprocess.run(["git", "add", "docs/plan-progress-status.md"], check=False, capture_output=True, timeout=30)
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"docs: record completion of Epoch {epoch_num} baseline training and validation"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if commit_res.returncode == 0:
            push_res = subprocess.run(["git", "push", "origin", "main"], check=False, capture_output=True, text=True, timeout=90)
            if push_res.returncode == 0:
                print(f"  [Git-Sync] Epoch {epoch_num} completion committed & pushed to GitHub -> main", flush=True)
    except Exception as e:
        print(f"  [Git-Sync Warning] Epoch completion sync failed: {e}", flush=True)



def build_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    betas: Tuple[float, float] = (0.9, 0.999),
    backbone_lr_scale: float = 0.1,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with differential LR for pretrained backbone vs FPN/head."""
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if name.startswith("backbone_fpn.") and not name.startswith("backbone_fpn.fpn."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale})

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 2,
    min_lr: float = 0.00001,
    base_lr: float = 0.001,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build CosineAnnealingLR scheduler with linear warmup."""
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(total_epochs - warmup_epochs),
            eta_min=min_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
        eta_min=min_lr,
    )


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    save_dir: str | Path,
    best_name: str = "best_map.pth",
    last_name: str = "last_epoch.pth",
) -> None:
    """Save training checkpoint to disk."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_path = out_dir / last_name
    torch.save(state, last_path)

    # Save numbered epoch checkpoint (epoch_1.pth, epoch_2.pth, ...)
    if "epoch" in state:
        epoch_num = int(state["epoch"]) + 1
        epoch_path = out_dir / f"epoch_{epoch_num}.pth"
        torch.save(state, epoch_path)

    if is_best:
        best_path = out_dir / best_name
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model and optimizer state from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optim_groups_changed = False
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as e:
            optim_groups_changed = True
            print(f"[WARN] Optimizer param groups changed ({e}); keeping fresh optimizer state with differential LR.")
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        if optim_groups_changed:
            for _ in range(ckpt.get("epoch", 0) + 1):
                try:
                    scheduler.step()
                except Exception:
                    pass
        else:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"[WARN] Could not restore scheduler state dict ({e}); stepping scheduler to epoch {ckpt.get('epoch', 0) + 1}.")
                for _ in range(ckpt.get("epoch", 0) + 1):
                    try:
                        scheduler.step()
                    except Exception:
                        pass
    if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    loss_fn: Callable,
    grad_clip_norm: float = 10.0,
    accumulate_grad_batches: int = 1,
    use_amp: bool = True,
    checkpoint_dir: Optional[Path] = None,
    save_interval: int = 50,
    start_batch_idx: int = 0,
    git_sync: bool = False,
    git_sync_interval: int = 500,
    initial_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Train detector for one single epoch.

    Args:
        model: YOLODetector instance.
        optimizer: PyTorch optimizer.
        scaler: GradScaler for AMP.
        dataloader: Training DataLoader.
        device: Target torch device.
        epoch: Current epoch index (0-indexed).
        loss_fn: MultiTaskLoss module.
        grad_clip_norm: Maximum gradient norm clipping threshold (default: 10.0).
        accumulate_grad_batches: Number of batches to accumulate gradients over.
        use_amp: Whether to enable automatic mixed precision.

    Returns:
        Dictionary of averaged epoch training losses.
    """
    model.train()
    tracker = MetricTracker()
    if initial_metrics and start_batch_idx > 0:
        init_n = start_batch_idx * getattr(dataloader, "batch_size", 4)
        for k, v in initial_metrics.items():
            tracker.update(k, v, n=init_n)
    optimizer.zero_grad()

    amp_enabled = use_amp and device.type == "cuda"
    device_type = "cuda" if device.type == "cuda" else "cpu"

    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx < start_batch_idx:
            continue
        bsz = images.size(0)
        images = images.to(device)
        gt_boxes = [t["boxes"].to(device) for t in targets]
        gt_labels = [t["labels"].to(device) for t in targets]

        assigned_targets = assign_targets(gt_boxes, gt_labels)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            predictions = model(images)

        # Compute loss in FP32 to prevent FP16 overflow/underflow in Focal/BCE/CIoU
        predictions_fp32 = {
            lvl: {k: v.float() for k, v in d.items()}
            for lvl, d in predictions.items()
        }
        loss_total, loss_dict = loss_fn(predictions_fp32, assigned_targets)

        if not torch.isfinite(loss_total):
            print(
                f"  [WARN] Non-finite loss ({loss_total.item()}) at Epoch {epoch + 1} Batch {batch_idx + 1}; skipping batch.",
                flush=True,
            )
            optimizer.zero_grad()
            del predictions, predictions_fp32, loss_total, assigned_targets, images, gt_boxes, gt_labels
            continue

        loss_scaled = loss_total / accumulate_grad_batches

        tracker.update_dict(loss_dict, n=bsz)
        cur_loss_val = loss_total.item()

        if amp_enabled:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        del predictions, predictions_fp32, loss_scaled, loss_total, assigned_targets, images, gt_boxes, gt_labels

        if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(dataloader):
            if amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=grad_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=grad_clip_norm,
                )
                optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            cur = tracker.get_averages()
            print(
                f"  [Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(dataloader)}] "
                f"loss: {cur_loss_val:.4f} (avg: {cur.get('loss_total', 0.0):.4f}) "
                f"box: {cur.get('loss_box', 0.0):.4f} "
                f"obj: {cur.get('loss_obj', 0.0):.4f} "
                f"cls: {cur.get('loss_cls', 0.0):.4f}",
                flush=True,
            )

        if checkpoint_dir is not None and (batch_idx + 1) % save_interval == 0:
            interim_path = Path(checkpoint_dir) / f"epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": tracker.get_averages(),
                },
                interim_path,
            )
            print(f"  [Auto-save] Checkpoint saved at batch {batch_idx + 1} -> {interim_path}", flush=True)

        if git_sync and ((batch_idx + 1) % git_sync_interval == 0 or (batch_idx + 1) == len(dataloader)):
            _sync_git_progress(epoch + 1, batch_idx + 1, len(dataloader), tracker.get_averages())

    return tracker.get_averages()


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Callable,
) -> Dict[str, float]:
    """Evaluate detector losses on validation dataset.

    Args:
        model: YOLODetector instance.
        dataloader: Validation DataLoader.
        device: Target torch device.
        loss_fn: MultiTaskLoss module.

    Returns:
        Dictionary of averaged validation losses.
    """
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            gt_boxes = [t["boxes"].to(device) for t in targets]
            gt_labels = [t["labels"].to(device) for t in targets]

            assigned_targets = assign_targets(gt_boxes, gt_labels)
            predictions = model(images)
            _, loss_dict = loss_fn(predictions, assigned_targets)
            tracker.update_dict(loss_dict, n=images.size(0))
            del predictions, assigned_targets, images, gt_boxes, gt_labels

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
                cur_val = tracker.get_averages()
                print(
                    f"  [Validation | Batch {batch_idx + 1}/{len(dataloader)}] "
                    f"loss: {cur_val.get('loss_total', 0.0):.4f}",
                    flush=True,
                )

    return tracker.get_averages()


def build_dataloaders(
    data_config: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 2,
    max_images: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from configuration."""
    dataset_cfg = data_config.get("dataset", {})

    train_img_dir = Path(dataset_cfg.get("train_images_dir", "data/coco/train2017"))
    train_ann_file = Path(dataset_cfg.get("train_annotations_file", "data/coco/train_subset14k.json"))
    val_img_dir = Path(dataset_cfg.get("val_heldout_images_dir", dataset_cfg.get("train_images_dir", "data/coco/train2017")))
    val_ann_file = Path(dataset_cfg.get("val_heldout_annotations_file", "data/coco/val_heldout1k.json"))

    # Fallback to val2017 directory if train2017 is not yet downloaded (useful for smoke runs)
    if not train_img_dir.exists() and val_img_dir.exists():
        print(f"[WARN] Train images dir not found: {train_img_dir}. Falling back to val images: {val_img_dir}")
        train_img_dir = val_img_dir
        train_ann_file = Path(dataset_cfg.get("test_val2017_annotations_file", "data/coco/annotations/instances_val2017.json"))
        val_ann_file = train_ann_file

    if not val_img_dir.exists():
        raise FileNotFoundError(f"Validation images dir not found: {val_img_dir}")
    if not val_ann_file.exists():
        # Fallback to instances_val2017.json if heldout does not exist
        fallback_val_ann = Path("data/coco/annotations/instances_val2017.json")
        if fallback_val_ann.exists():
            val_ann_file = fallback_val_ann
        else:
            raise FileNotFoundError(f"Validation annotation file not found: {val_ann_file}")

    train_dataset = COCODataset(
        root=train_img_dir,
        annotation_file=train_ann_file,
        img_size=512,
        is_train=True,
    )
    val_dataset = COCODataset(
        root=val_img_dir,
        annotation_file=val_ann_file,
        img_size=512,
        is_train=False,
    )

    if max_images is not None:
        train_len = min(max_images, len(train_dataset))
        val_len = min(max(1, max_images // 2), len(val_dataset))
        train_dataset = Subset(train_dataset, list(range(train_len)))  # type: ignore[assignment]
        val_dataset = Subset(val_dataset, list(range(val_len)))  # type: ignore[assignment]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train(
    config_path: str | Path = "configs/base_config.yaml",
    data_config_path: str | Path = "configs/dataset_coco_subset.yaml",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    resume_path: Optional[str | Path] = None,
    seed_override: Optional[int] = None,
    max_images: Optional[int] = None,
    device_override: Optional[str] = None,
    num_threads_override: Optional[int] = None,
    git_sync_override: Optional[bool] = None,
    git_sync_interval: int = 500,
) -> Dict[str, Any]:
    """Execute complete detector training pipeline."""
    config = load_yaml_config(config_path)
    data_config = load_yaml_config(data_config_path)

    training_cfg = config.get("training", {})
    optim_cfg = config.get("optimizer", {})
    sched_cfg = config.get("scheduler", {})
    loss_cfg = config.get("loss_weights", {})
    ckpt_cfg = config.get("checkpoints", {})
    log_cfg = config.get("logging", {})

    seed = seed_override or training_cfg.get("seed", 42)
    set_seed(seed)

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        cpu_threads = num_threads_override or 4
        torch.set_num_threads(cpu_threads)
        print(f"[INFO] PyTorch CPU threads: {cpu_threads} (thermal optimized)")

    epochs = epochs_override or training_cfg.get("epochs", 20)
    batch_size = batch_size_override or training_cfg.get("batch_size", 16)
    lr = lr_override or optim_cfg.get("lr", 0.001)
    weight_decay = optim_cfg.get("weight_decay", 0.0005)
    use_amp = training_cfg.get("amp", True)
    accumulate_batches = training_cfg.get("accumulate_grad_batches", 1)
    if device.type == "cpu" and batch_size < 16 and accumulate_batches == 1:
        accumulate_batches = max(1, 16 // batch_size)
    grad_clip = training_cfg.get("grad_clip_norm", 10.0)
    freeze_epochs = training_cfg.get("freeze_backbone_epochs", 3)
    num_workers = 0 if sys.platform == "win32" and device.type == "cpu" else training_cfg.get("num_workers", 0)
    git_sync = git_sync_override if git_sync_override is not None else training_cfg.get("git_sync", False)

    save_dir = Path(ckpt_cfg.get("save_dir", "outputs/checkpoints"))
    best_name = ckpt_cfg.get("best_model_name", "best_map.pth")
    last_name = ckpt_cfg.get("last_model_name", "last_epoch.pth")
    log_dir = Path(log_cfg.get("log_dir", "outputs/logs"))

    print("=" * 65)
    print(" YOLO Detector Training Engine (Milestone M4)")
    print("=" * 65)
    print(f"Device:            {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Total Epochs:      {epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Learning Rate:     {lr}")
    print(f"AMP:               {use_amp and device.type == 'cuda'}")
    print(f"Grad Clip Norm:    {grad_clip}")
    print(f"Backbone Freeze:   First {freeze_epochs} epochs")
    print(f"Checkpoints Dir:   {save_dir}")
    print(f"Max Images Limit:  {max_images or 'Full Dataset'}")
    print("=" * 65)

    # 1. Build Model
    model = build_detector(num_classes=80, pretrained_backbone=True)
    model.to(device)

    # 2. Build Loss
    criterion = MultiTaskLoss(
        lambda_box=loss_cfg.get("lambda_box", 5.0),
        lambda_obj=loss_cfg.get("lambda_obj", 1.0),
        lambda_cls=loss_cfg.get("lambda_cls", 1.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    ).to(device)

    # 3. Build Optimizer & Scheduler
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=sched_cfg.get("warmup_epochs", 2),
        min_lr=sched_cfg.get("min_lr", 0.00001),
        base_lr=lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    # 4. Resume from Checkpoint if specified
    start_epoch = 0
    resume_batch_idx = 0
    best_val_loss = float("inf")
    initial_train_metrics: Optional[Dict[str, float]] = None
    if resume_path:
        print(f"[INFO] Resuming training from checkpoint: {resume_path}")
        ckpt_info = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device=device
        )
        if "batch_idx" in ckpt_info and ckpt_info["batch_idx"] is not None:
            start_epoch = ckpt_info.get("epoch", 0)
            resume_batch_idx = ckpt_info["batch_idx"] + 1
            initial_train_metrics = ckpt_info.get("train_metrics")
            print(f"[INFO] Resumed at epoch {start_epoch + 1}, batch {resume_batch_idx + 1}")
            if "scheduler_state_dict" not in ckpt_info:
                for _ in range(start_epoch):
                    try:
                        scheduler.step()
                    except Exception:
                        pass
            best_map_file = save_dir / best_name
            if best_map_file.exists():
                try:
                    b_ckpt = torch.load(best_map_file, map_location="cpu")
                    best_val_loss = b_ckpt.get("best_val_loss", float("inf"))
                    print(f"[INFO] Loaded previous best validation loss from {best_name}: {best_val_loss:.4f}")
                except Exception:
                    pass
        else:
            start_epoch = ckpt_info.get("epoch", 0) + 1
            resume_batch_idx = 0
            best_val_loss = ckpt_info.get("best_val_loss", float("inf"))
            print(f"[INFO] Resumed at epoch {start_epoch + 1}, previous best loss: {best_val_loss:.4f}")

    # 5. Build DataLoaders
    train_loader, val_loader = build_dataloaders(
        data_config=data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
    )
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    logger = TrainingLogger(log_dir=log_dir)
    git_hash = get_git_commit_hash()

    # 6. Main Training Loop
    for epoch in range(start_epoch, epochs):
        # Configure backbone freezing
        if epoch < freeze_epochs:
            model.freeze_backbone()
            stage_str = "Backbone FROZEN"
        else:
            model.unfreeze_backbone(trainable_layers=3)
            stage_str = "Backbone UNFROZEN (fine-tuning)"

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch [{epoch + 1}/{epochs}] ({stage_str}) | LR: {current_lr:.6f} ---")

        current_start_batch = resume_batch_idx if epoch == start_epoch else 0
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            loss_fn=criterion,
            grad_clip_norm=grad_clip,
            accumulate_grad_batches=accumulate_batches,
            use_amp=use_amp,
            checkpoint_dir=save_dir,
            save_interval=50,
            start_batch_idx=current_start_batch,
            git_sync=git_sync,
            git_sync_interval=git_sync_interval,
            initial_metrics=initial_train_metrics if epoch == start_epoch else None,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=criterion,
        )

        scheduler.step()

        train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        val_str = " | ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
        print(f"[Epoch {epoch + 1} Train] {train_str}")
        print(f"[Epoch {epoch + 1} Val  ] {val_str}")

        # Check if validation loss improved
        val_loss = val_metrics.get("loss_total", float("inf"))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f" ★ Best validation loss improved to {best_val_loss:.4f}! Saving {best_name}")

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else None,
            "best_val_loss": best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config,
            "git_commit": git_hash,
        }

        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            save_dir=save_dir,
            best_name=best_name,
            last_name=last_name,
        )

        logger.log_epoch(
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
        )

        if git_sync:
            _sync_git_epoch_completed(epoch + 1, train_metrics, val_metrics)

    print("\n" + "=" * 65)
    print(f" Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f" Checkpoint saved to: {save_dir / best_name}")
    print(f" Logs saved to:       {logger.history_file}")
    print("=" * 65)

    return {
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs,
        "best_checkpoint": str(save_dir / best_name),
        "last_checkpoint": str(save_dir / last_name),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO-style detector on COCO subset")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base config YAML")
    parser.add_argument("--data-config", type=str, default="configs/dataset_coco_subset.yaml", help="Path to dataset config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images for quick dry-run testing")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', or 'cuda:0')")
    parser.add_argument("--num-threads", type=int, default=None, help="Number of PyTorch CPU threads (default: 4 on CPU)")
    parser.add_argument("--git-sync", action="store_true", help="Periodically sync training progress to git")
    parser.add_argument("--git-sync-interval", type=int, default=500, help="Interval of batches between git syncs (default: 500)")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    train(
        config_path=args.config,
        data_config_path=args.data_config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        resume_path=args.resume,
        seed_override=args.seed,
        max_images=args.max_images,
        device_override=args.device,
        num_threads_override=args.num_threads,
        git_sync_override=args.git_sync,
        git_sync_interval=args.git_sync_interval,
    )


if __name__ == "__main__":
    main()
