from __future__ import annotations

import math
from typing import Callable

from config import SchedulerConfig, is_teacher


def constant_schedule(step: int, total_steps: int) -> float:
    return 1.0


def cosine_schedule(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    progress = min(max(step - 1, 0), total_steps - 1) / (total_steps - 1)
    min_scale = 0.05
    return min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * progress))


def warmup_cosine_schedule(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    warmup_steps = max(1, int(0.1 * total_steps))
    if step <= warmup_steps:
        return step / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(max(step - warmup_steps, 0), decay_steps) / decay_steps
    min_scale = 0.05
    return min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * progress))


def step_decay_schedule(step: int, total_steps: int) -> float:
    del total_steps
    step_size = 50
    gamma = 0.5
    drops = max(step - 1, 0) // step_size
    return gamma ** drops


def build_scheduler(config: SchedulerConfig) -> Callable[[int, int], float]:
    name = config.name

    if name == "constant":
        return constant_schedule

    if is_teacher():
        from _teacher_solutions.schedulers import build_teacher_scheduler
        return build_teacher_scheduler(config)

    if name == "cosine":
        return cosine_schedule
    if name == "warmup_cosine":
        return warmup_cosine_schedule
    if name == "step_decay":
        return step_decay_schedule

    raise ValueError(f"Unknown scheduler name: {name}")
