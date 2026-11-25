"""Metric tracker and logging utilities for YOLO detector training (Milestone M4).

Provides:
- MetricTracker: computes and stores running averages for loss components.
- EpochLogger: logs training progress and persists metric history to JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricTracker:
    """Computes and stores running averages and current values for multiple metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.values: Dict[str, float] = defaultdict(float)
        self.sums: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def update(self, key: str, value: float, n: int = 1) -> None:
        """Update a metric with a new observed value.

        Args:
            key: Name of the metric (e.g. 'loss_total', 'loss_box').
            value: Scalar metric value.
            n: Number of samples or batch weight (default: 1).
        """
        self.values[key] = float(value)
        self.sums[key] += float(value) * n
        self.counts[key] += n

    def update_dict(self, metrics: Dict[str, float], n: int = 1) -> None:
        """Update multiple metrics at once from a dictionary."""
        for k, v in metrics.items():
            self.update(k, v, n=n)

    def get_avg(self, key: str) -> float:
        """Get the running average for a metric."""
        if self.counts[key] == 0:
            return 0.0
        return self.sums[key] / self.counts[key]

    def get_val(self, key: str) -> float:
        """Get the latest value for a metric."""
        return self.values[key]

    def get_averages(self) -> Dict[str, float]:
        """Return a dictionary of running averages for all metrics."""
        return {k: self.get_avg(k) for k in self.counts}

    def format_summary(self, prefix: str = "") -> str:
        """Format metrics into a human-readable string.

        Args:
            prefix: Optional prefix for the log line (e.g. 'Epoch 1 [Train]').
        """
        avgs = self.get_averages()
        parts = [f"{prefix}"] if prefix else []
        for k, v in avgs.items():
            parts.append(f"{k}: {v:.4f}")
        return " | ".join(parts)


class TrainingLogger:
    """Handles logging and persisting epoch history to disk."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "train_history.json"
        self.history: List[Dict[str, Any]] = []
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.history = data
            except Exception:
                self.history = []

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None, lr: Optional[float] = None) -> None:
        """Log one completed epoch and append to history file."""
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
        }
        if val_metrics is not None:
            entry["val"] = val_metrics
        if lr is not None:
            entry["lr"] = lr

        # Update if epoch already in history, else append
        updated = False
        for i, item in enumerate(self.history):
            if item.get("epoch") == epoch:
                self.history[i] = entry
                updated = True
                break
        if not updated:
            self.history.append(entry)

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return complete training history."""
        return self.history
