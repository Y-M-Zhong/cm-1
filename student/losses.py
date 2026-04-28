from __future__ import annotations

import torch

from config import LossConfig, is_teacher


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = prediction - target
    return torch.mean(diff * diff)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = prediction - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def mse_l1_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.2) -> torch.Tensor:
    diff = prediction - target
    mse = torch.mean(diff * diff)
    l1 = torch.mean(torch.abs(diff))
    return mse + l1_weight * l1


def mse_edge_loss(prediction: torch.Tensor, target: torch.Tensor, edge_weight: float = 0.1) -> torch.Tensor:
    diff = prediction - target
    mse = torch.mean(diff * diff)

    pred_dx = prediction[:, 1:, :] - prediction[:, :-1, :]
    target_dx = target[:, 1:, :] - target[:, :-1, :]
    pred_dy = prediction[1:, :, :] - prediction[:-1, :, :]
    target_dy = target[1:, :, :] - target[:-1, :, :]

    edge_dx = torch.mean((pred_dx - target_dx) ** 2)
    edge_dy = torch.mean((pred_dy - target_dy) ** 2)
    return mse + edge_weight * (edge_dx + edge_dy)


def build_loss(config: LossConfig):
    name = config.name

    if name == "mse":
        return mse_loss

    if is_teacher():
        from _teacher_solutions import losses as ref

        if name == "l1":
            return ref.l1_loss
        if name == "charbonnier":
            return lambda p, t: ref.charbonnier_loss(p, t, eps=1e-3)
        if name == "mse_l1":
            return lambda p, t: ref.mse_l1_loss(p, t, l1_weight=0.2)
        if name == "mse_edge":
            return lambda p, t: ref.mse_edge_loss(p, t, edge_weight=0.1)

    if name == "l1":
        return l1_loss
    if name == "charbonnier":
        return lambda prediction, target: charbonnier_loss(prediction, target, eps=1e-3)
    if name == "mse_l1":
        return lambda prediction, target: mse_l1_loss(prediction, target, l1_weight=0.2)
    if name == "mse_edge":
        return lambda prediction, target: mse_edge_loss(prediction, target, edge_weight=0.1)

    raise ValueError(f"Unknown loss name: {name}")
