"""Lightweight Semantic Target Control reference implementation."""

from .core import (
    Candidate,
    ControlResult,
    EnergyContractError,
    Status,
    control_topk,
    rank_candidates,
)

__all__ = [
    "Candidate",
    "ControlResult",
    "EnergyContractError",
    "Status",
    "control_topk",
    "rank_candidates",
]

__version__ = "1.0.0"
