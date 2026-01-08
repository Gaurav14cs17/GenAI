"""Loss functions for SoFlow training."""

from .flow_matching import FlowMatchingLoss
from .consistency import SolutionConsistencyLoss
from .combined import SoFlowLoss

__all__ = [
    "FlowMatchingLoss",
    "SolutionConsistencyLoss",
    "SoFlowLoss",
]

