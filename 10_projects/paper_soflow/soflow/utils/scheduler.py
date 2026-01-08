"""
Schedule functions for SoFlow training.

These schedules control the intermediate time l in the consistency loss,
implementing curriculum learning from easy to hard targets.
"""

import math
from typing import Callable


def get_schedule_fn(schedule_type: str = "linear", **kwargs) -> Callable[[int, int], float]:
    """
    Get a schedule function by name.
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", "constant").
        **kwargs: Additional arguments for the schedule.
        
    Returns:
        Schedule function r(k, K) -> float.
    """
    if schedule_type == "linear":
        return LinearSchedule(**kwargs)
    elif schedule_type == "cosine":
        return CosineSchedule(**kwargs)
    elif schedule_type == "constant":
        value = kwargs.get("value", 0.5)
        return lambda k, K: value
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class LinearSchedule:
    """
    Linear decay schedule: r(k, K) = max(min_value, 1 - k/K).
    
    Args:
        min_value: Minimum value to decay to.
        warmup_steps: Number of warmup steps with r = 1.
    """
    
    def __init__(self, min_value: float = 0.0, warmup_steps: int = 0):
        self.min_value = min_value
        self.warmup_steps = warmup_steps

    def __call__(self, k: int, K: int) -> float:
        if k < self.warmup_steps:
            return 1.0
        
        effective_k = k - self.warmup_steps
        effective_K = K - self.warmup_steps
        
        if effective_K <= 0:
            return self.min_value
        
        ratio = 1.0 - effective_k / effective_K
        return max(self.min_value, ratio)


class CosineSchedule:
    """
    Cosine annealing schedule: r(k, K) = min_value + (1 - min_value) * (1 + cos(πk/K)) / 2.
    
    Args:
        min_value: Minimum value at the end.
        warmup_steps: Number of warmup steps.
    """
    
    def __init__(self, min_value: float = 0.0, warmup_steps: int = 0):
        self.min_value = min_value
        self.warmup_steps = warmup_steps

    def __call__(self, k: int, K: int) -> float:
        if k < self.warmup_steps:
            # Linear warmup from min_value to 1
            return self.min_value + (1.0 - self.min_value) * k / self.warmup_steps
        
        effective_k = k - self.warmup_steps
        effective_K = K - self.warmup_steps
        
        if effective_K <= 0:
            return self.min_value
        
        progress = min(1.0, effective_k / effective_K)
        cosine_value = (1 + math.cos(math.pi * progress)) / 2
        return self.min_value + (1.0 - self.min_value) * cosine_value


class ExponentialSchedule:
    """
    Exponential decay schedule: r(k, K) = max(min_value, exp(-decay_rate * k/K)).
    
    Args:
        min_value: Minimum value.
        decay_rate: Rate of exponential decay.
    """
    
    def __init__(self, min_value: float = 0.0, decay_rate: float = 5.0):
        self.min_value = min_value
        self.decay_rate = decay_rate

    def __call__(self, k: int, K: int) -> float:
        if K <= 0:
            return self.min_value
        
        progress = k / K
        value = math.exp(-self.decay_rate * progress)
        return max(self.min_value, value)


class StepSchedule:
    """
    Step schedule that decreases at specified milestones.
    
    Args:
        milestones: List of (step_fraction, value) pairs.
        initial_value: Initial value before first milestone.
    """
    
    def __init__(
        self,
        milestones: list = None,
        initial_value: float = 1.0,
    ):
        if milestones is None:
            # Default: decrease at 25%, 50%, 75% of training
            milestones = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
        self.milestones = sorted(milestones, key=lambda x: x[0])
        self.initial_value = initial_value

    def __call__(self, k: int, K: int) -> float:
        if K <= 0:
            return self.initial_value
        
        progress = k / K
        value = self.initial_value
        
        for milestone_progress, milestone_value in self.milestones:
            if progress >= milestone_progress:
                value = milestone_value
        
        return value

