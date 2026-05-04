from __future__ import annotations

import math

import torch

from config import Config, is_teacher
from models import Gaussian2DModel, inverse_sigmoid, inverse_softplus


class RandomGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        del target_image
        num_gaussians = model.num_gaussians
        device = model.center_raw.device

        center_init = torch.rand(num_gaussians, 2, device=device)
        sigma_init = 0.01 + 0.01 * torch.rand(num_gaussians, 1, device=device)
        scale_init = sigma_init.repeat(1, 2)
        rotation_init = torch.zeros(num_gaussians, 2, device=device)
        rotation_init[:, 0] = 1.0
        alpha_value = 0.2 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        color_init = 0.5 + 0.3 * torch.randn(num_gaussians, 3, device=device)
        color_init = color_init.clamp(0.05, 0.95)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )


class GridGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        del target_image
        num_gaussians = model.num_gaussians
        device = model.center_raw.device

        cols = math.ceil(math.sqrt(num_gaussians))
        rows = math.ceil(num_gaussians / cols)
        xs = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols
        ys = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        center_init = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)[:num_gaussians]

        scale_x = 0.45 / cols
        scale_y = 0.45 / rows
        scale_init = torch.empty(num_gaussians, 2, device=device)
        scale_init[:, 0] = scale_x
        scale_init[:, 1] = scale_y if self.config.model.use_anisotropic else scale_x

        rotation_init = torch.zeros(num_gaussians, 2, device=device)
        rotation_init[:, 0] = 1.0
        alpha_value = 0.35 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        color_init = torch.full((num_gaussians, 3), 0.5, device=device)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )


class ImageSampleGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        if target_image is None:
            RandomGaussianInitializer(self.config).initialize(model, target_image=None)
            return

        num_gaussians = model.num_gaussians
        device = model.center_raw.device
        image = target_image.detach().to(device).clamp(0.0, 1.0)
        height, width, _ = image.shape

        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        edge = torch.zeros_like(luminance)
        edge[:, 1:] += torch.abs(luminance[:, 1:] - luminance[:, :-1])
        edge[1:, :] += torch.abs(luminance[1:, :] - luminance[:-1, :])
        weights = 0.05 + luminance + 2.0 * edge
        probabilities = weights.reshape(-1)
        probabilities = probabilities / probabilities.sum().clamp_min(1e-12)

        replacement = num_gaussians > probabilities.numel()
        indices = torch.multinomial(probabilities, num_gaussians, replacement=replacement)
        ys = torch.div(indices, width, rounding_mode="floor")
        xs = indices - ys * width

        jitter_x = (torch.rand(num_gaussians, device=device) - 0.5) / max(width - 1, 1)
        jitter_y = (torch.rand(num_gaussians, device=device) - 0.5) / max(height - 1, 1)
        center_x = xs.to(torch.float32) / max(width - 1, 1) + jitter_x
        center_y = ys.to(torch.float32) / max(height - 1, 1) + jitter_y
        center_init = torch.stack([center_x, center_y], dim=-1).clamp(1e-4, 1.0 - 1e-4)

        base_scale = 0.35 / math.sqrt(num_gaussians)
        scale_noise = 0.75 + 0.5 * torch.rand(num_gaussians, 2, device=device)
        scale_init = (base_scale * scale_noise).clamp(0.004, 0.02)
        if not self.config.model.use_anisotropic:
            scale_init[:, 1] = scale_init[:, 0]

        angles = 2.0 * math.pi * torch.rand(num_gaussians, device=device)
        rotation_init = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        alpha_value = 0.2 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        color_init = image[ys, xs].clamp(0.02, 0.98)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )


def build_initializer(config: Config):
    name = config.initializer.name

    if name == "random":
        return RandomGaussianInitializer(config)
    if name == "grid":
        if is_teacher():
            from _teacher_solutions.grid_init import GridGaussianInitializer as TeacherGridGaussianInitializer
            return TeacherGridGaussianInitializer(config)
        return GridGaussianInitializer(config)
    if name == "image_sample":
        if is_teacher():
            from _teacher_solutions.image_sample_init import ImageSampleGaussianInitializer as TeacherImageSampleGaussianInitializer
            return TeacherImageSampleGaussianInitializer(config)
        return ImageSampleGaussianInitializer(config)

    raise ValueError(f"Unknown initializer name: {name}")
