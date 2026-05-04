from __future__ import annotations

import torch

from config import OptimizerConfig, is_teacher
from models import Gaussian2DModel


def _iter_params(param_groups: list[dict]):
    for group in param_groups:
        for param in group["params"]:
            yield group, param


def _zero_grad(param_groups: list[dict]) -> None:
    for _, param in _iter_params(param_groups):
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def build_torch_adam(param_groups: list[dict], lr: float) -> torch.optim.Optimizer:
    torch_groups = [{"params": g["params"], "lr": g["lr"], "base_lr": g["base_lr"]} for g in param_groups]
    return torch.optim.Adam(torch_groups, lr=lr)


class StudentSGD:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups

    def zero_grad(self) -> None:
        _zero_grad(self.param_groups)

    def step(self) -> None:
        with torch.no_grad():
            for group, param in _iter_params(self.param_groups):
                if param.grad is None:
                    continue
                param.add_(param.grad, alpha=-group["lr"])


class StudentMomentum:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.velocity: dict[int, torch.Tensor] = {}
        self.momentum = 0.9

    def zero_grad(self) -> None:
        _zero_grad(self.param_groups)

    def step(self) -> None:
        with torch.no_grad():
            for group, param in _iter_params(self.param_groups):
                if param.grad is None:
                    continue
                pid = id(param)
                if pid not in self.velocity:
                    self.velocity[pid] = torch.zeros_like(param)
                velocity = self.velocity[pid]
                velocity.mul_(self.momentum).add_(param.grad)
                param.add_(velocity, alpha=-group["lr"])


class StudentAdam:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def zero_grad(self) -> None:
        _zero_grad(self.param_groups)

    def step(self) -> None:
        self.step_count += 1
        with torch.no_grad():
            for group, param in _iter_params(self.param_groups):
                if param.grad is None:
                    continue
                pid = id(param)
                if pid not in self.state:
                    self.state[pid] = {
                        "m": torch.zeros_like(param),
                        "v": torch.zeros_like(param),
                    }
                state = self.state[pid]
                grad = param.grad
                state["m"].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                state["v"].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                m_hat = state["m"] / (1.0 - self.beta1 ** self.step_count)
                v_hat = state["v"] / (1.0 - self.beta2 ** self.step_count)
                param.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-group["lr"])


class StudentAdamW:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 1e-2

    def zero_grad(self) -> None:
        _zero_grad(self.param_groups)

    def step(self) -> None:
        self.step_count += 1
        with torch.no_grad():
            for group, param in _iter_params(self.param_groups):
                if param.grad is None:
                    continue
                lr = group["lr"]
                if self.weight_decay != 0.0:
                    param.mul_(1.0 - lr * self.weight_decay)

                pid = id(param)
                if pid not in self.state:
                    self.state[pid] = {
                        "m": torch.zeros_like(param),
                        "v": torch.zeros_like(param),
                    }
                state = self.state[pid]
                grad = param.grad
                state["m"].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                state["v"].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                m_hat = state["m"] / (1.0 - self.beta1 ** self.step_count)
                v_hat = state["v"] / (1.0 - self.beta2 ** self.step_count)
                param.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-lr)


class StudentMuon:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.buffers: dict[int, torch.Tensor] = {}
        self.momentum = 0.95
        self.nesterov = True
        self.ns_steps = 5

    def zero_grad(self) -> None:
        _zero_grad(self.param_groups)

    def step(self) -> None:
        with torch.no_grad():
            for group, param in _iter_params(self.param_groups):
                if param.grad is None:
                    continue
                pid = id(param)
                if pid not in self.buffers:
                    self.buffers[pid] = torch.zeros_like(param)
                buf = self.buffers[pid]
                buf.mul_(self.momentum).add_(param.grad)
                update = param.grad.add(buf, alpha=self.momentum) if self.nesterov else buf
                update = _zeropower_via_newton_schulz(update, steps=self.ns_steps)
                param.add_(update, alpha=-group["lr"])


def _zeropower_via_newton_schulz(update: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if update.ndim < 2:
        return update / update.norm().clamp_min(1e-12)

    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.t()

    norm = matrix.norm().clamp_min(1e-12)
    x = matrix / norm
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.t()
        x = a * x + b * (xx_t @ x) + c * (xx_t @ xx_t @ x)

    if transposed:
        x = x.t()
    return x.reshape(original_shape)


def build_optimizer(model: Gaussian2DModel, config: OptimizerConfig):
    name = config.name
    base_lr = config.lr
    param_groups = model.get_param_groups(base_lr, config.param_groups)

    if name == "torch_adam":
        return build_torch_adam(param_groups=param_groups, lr=base_lr)

    if name == "student_sgd":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentSGD as TeacherStudentSGD
            return TeacherStudentSGD(param_groups=param_groups)
        return StudentSGD(param_groups=param_groups)

    if name == "student_momentum":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentMomentum as TeacherStudentMomentum
            return TeacherStudentMomentum(param_groups=param_groups)
        return StudentMomentum(param_groups=param_groups)

    if name == "student_adam":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentAdam as TeacherStudentAdam
            return TeacherStudentAdam(param_groups=param_groups)
        return StudentAdam(param_groups=param_groups)

    if name == "student_adamw":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentAdamW as TeacherStudentAdamW
            return TeacherStudentAdamW(param_groups=param_groups)
        return StudentAdamW(param_groups=param_groups)

    if name == "student_muon":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentMuon as TeacherStudentMuon
            return TeacherStudentMuon(param_groups=param_groups)
        return StudentMuon(param_groups=param_groups)

    raise ValueError(f"Unknown optimizer name: {name}")
