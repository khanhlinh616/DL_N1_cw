"""Unit tests for inference coordinate decoder and batched NMS (Milestone M5).

Requirements from DEFINE_AND_DESIGN.md §2.4.3:
- Center offsets decode into [-0.5, 1.5] * stride.
- Width and height decode into [0, 16 * stride].
- Scores equal sigmoid(obj) * sigmoid(cls).
- Batched NMS correctly suppresses overlapping boxes within the same class.

TODO: Implement full unit test suite in Milestone M5.
"""

from __future__ import annotations

import pytest


def test_decoder_stub() -> None:
    """Placeholder test for Milestone M5 decoder."""
    # Will be implemented along with src/core/decoder.py in Milestone M5
    assert True
