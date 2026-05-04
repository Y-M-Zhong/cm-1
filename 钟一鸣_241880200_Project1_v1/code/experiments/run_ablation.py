from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, set_mode
from models import Gaussian2DModel
from renderer import GaussianRenderer
from student.initializers import build_initializer
from student.losses import build_loss
from student.optimizers import build_optimizer
from student.schedulers import build_scheduler
from target_generators import build_target_generator
from train import evaluate_prediction, save_comparison_visual
from utils import ensure_dir, resolve_device, save_image, set_seed


def build_default_config() -> Config:
    config = Config()
    config.train.save_video = False
    config.system.output_dir = "outputs_ablation"
    return config


def run_case(name: str, group: str, configure: Callable[[Config], None], output_dir: Path) -> dict[str, str | float]:
    config = build_default_config()
    configure(config)

    set_seed(config.system.seed)
    device = resolve_device(config.system.device)
    target = build_target_generator(config).generate(project_root=PROJECT_ROOT, device=device)

    model = Gaussian2DModel(num_gaussians=config.model.num_gaussians).to(device)
    build_initializer(config).initialize(model, target_image=target)

    renderer = GaussianRenderer(
        image_size=config.target.image_size,
        bg_color=config.render.bg_color,
        use_anisotropic=config.model.use_anisotropic,
        use_alpha=config.model.use_alpha,
    )
    optimizer = build_optimizer(model=model, config=config.optimizer)
    loss_fn = build_loss(config.loss)
    scheduler = build_scheduler(config.scheduler)

    losses: list[float] = []
    for step in range(1, config.train.num_steps + 1):
        lr_scale = scheduler(step, config.train.num_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["base_lr"] * lr_scale

        optimizer.zero_grad()
        prediction = renderer.render(model.get_render_params())
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    prediction = renderer.render(model.get_render_params())
    result = evaluate_prediction(prediction=prediction, target=target)

    case_dir = ensure_dir(output_dir / name)
    save_image(prediction, case_dir / "prediction.png")
    save_comparison_visual(target=target, prediction=prediction, path=case_dir / "comparison.png")
    (case_dir / "losses.csv").write_text(
        "step,loss\n" + "\n".join(f"{idx},{value:.8f}" for idx, value in enumerate(losses, start=1)) + "\n",
        encoding="utf-8",
    )

    return {
        "group": group,
        "case": name,
        "loss": config.loss.name,
        "initializer": config.initializer.name,
        "optimizer": config.optimizer.name,
        "scheduler": config.scheduler.name,
        "use_anisotropic": str(config.model.use_anisotropic),
        "use_alpha": str(config.model.use_alpha),
        "mse": result.mse,
        "mae": result.mae,
        "psnr": result.psnr,
    }


def build_cases() -> list[tuple[str, str, Callable[[Config], None]]]:
    def noop(config: Config) -> None:
        del config

    cases: list[tuple[str, str, Callable[[Config], None]]] = [
        ("baseline", "baseline", noop),
        ("loss_l1", "1.2A_loss", lambda c: setattr(c.loss, "name", "l1")),
        ("loss_charbonnier", "1.2A_loss", lambda c: setattr(c.loss, "name", "charbonnier")),
        ("init_grid", "1.2B_initializer", lambda c: setattr(c.initializer, "name", "grid")),
        ("init_image_sample", "1.2B_initializer", lambda c: setattr(c.initializer, "name", "image_sample")),
        ("opt_student_sgd", "1.2C_optimizer", lambda c: setattr(c.optimizer, "name", "student_sgd")),
        ("opt_student_momentum", "1.2C_optimizer", lambda c: setattr(c.optimizer, "name", "student_momentum")),
        ("opt_student_adam", "1.2C_optimizer", lambda c: setattr(c.optimizer, "name", "student_adam")),
        ("opt_student_adamw", "1.2C_optimizer", lambda c: setattr(c.optimizer, "name", "student_adamw")),
        ("opt_student_muon", "1.2C_optimizer", lambda c: setattr(c.optimizer, "name", "student_muon")),
        ("model_iso_no_alpha", "1.2D_model", lambda c: (setattr(c.model, "use_anisotropic", False), setattr(c.model, "use_alpha", False))),
        ("model_aniso_no_alpha", "1.2D_model", lambda c: setattr(c.model, "use_alpha", False)),
        ("model_iso_alpha", "1.2D_model", lambda c: setattr(c.model, "use_anisotropic", False)),
        ("model_aniso_alpha", "1.2D_model", noop),
        ("sched_cosine", "1.2E_scheduler", lambda c: setattr(c.scheduler, "name", "cosine")),
        ("sched_warmup_cosine", "1.2E_scheduler", lambda c: setattr(c.scheduler, "name", "warmup_cosine")),
        ("sched_step_decay", "1.2E_scheduler", lambda c: setattr(c.scheduler, "name", "step_decay")),
    ]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task 1.2 ablation experiments.")
    parser.add_argument("--output", default="outputs_ablation", help="Directory for ablation outputs.")
    parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    args = parser.parse_args()

    set_mode(args.mode)
    output_dir = ensure_dir(PROJECT_ROOT / args.output)
    rows = []
    for name, group, configure in build_cases():
        print(f"Running {name}...")
        row = run_case(name=name, group=group, configure=configure, output_dir=output_dir)
        rows.append(row)
        print(f"{name:24s} PSNR={row['psnr']:.4f} MSE={row['mse']:.8f} MAE={row['mae']:.8f}")

    csv_path = output_dir / "summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
