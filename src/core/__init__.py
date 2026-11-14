"""Core detection modules: label assigner, loss function, decoder."""

from src.core.assigner import assign_targets
from src.core.loss import MultiTaskLoss, compute_loss

__all__ = ["assign_targets", "MultiTaskLoss", "compute_loss"]
