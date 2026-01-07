"""
SoFlow: Solution Flow Models for One-Step Generative Modeling

A framework for one-step generation that directly learns the solution function
of the velocity ODE defined by Flow Matching.
"""

__version__ = "0.1.0"

from .models import SoFlowModel, DiT
from .losses import FlowMatchingLoss, SolutionConsistencyLoss, SoFlowLoss

__all__ = [
    "SoFlowModel",
    "DiT",
    "FlowMatchingLoss",
    "SolutionConsistencyLoss",
    "SoFlowLoss",
]

