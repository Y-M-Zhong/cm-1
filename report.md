# CM2026 Project 1 实验报告草稿

## 实验设置

本实验在默认配置基础上完成 2D Gaussian 图像拟合。任务 1.2 消融实验统一使用 `data/real_images/r1_flamingo_128.png`、`128 x 128`、`1000` 个高斯、`200` 步、随机种子 `42`。默认基线为 `mse` loss、`random` 初始化、`torch_adam` 优化器、`constant` 学习率、开启各向异性与 alpha。

已实现模块包括：

- Loss：`l1`、`charbonnier`、`mse_l1`、`mse_edge`
- Optimizer：`student_sgd`、`student_momentum`、`student_adam`、`student_adamw`、`student_muon`
- Initializer：`grid`、`image_sample`
- Scheduler：`cosine`、`warmup_cosine`、`step_decay`

## 任务 1.2 消融结果

结果由 `python experiments/run_ablation.py --output outputs_ablation_gpu` 在 GPU 上生成，完整图片和 loss 曲线数据位于 `outputs_ablation_gpu/`。

| 实验 | 配置 | PSNR | MSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Baseline | mse + random + torch_adam + constant | 30.5935 | 0.00087226 | 0.01990399 |
| 1.2A Loss | l1 | 29.6914 | 0.00107364 | 0.01967952 |
| 1.2A Loss | charbonnier | 29.7096 | 0.00106915 | 0.01959821 |
| 1.2B Init | grid | 31.3897 | 0.00072616 | 0.01820099 |
| 1.2B Init | image_sample | 30.7968 | 0.00083238 | 0.02033302 |
| 1.2C Optimizer | student_sgd | 10.5861 | 0.08737649 | 0.20757923 |
| 1.2C Optimizer | student_momentum | 10.9455 | 0.08043657 | 0.19914857 |
| 1.2C Optimizer | student_adam | 30.5219 | 0.00088677 | 0.02007134 |
| 1.2C Optimizer | student_adamw | 30.2050 | 0.00095389 | 0.02062576 |
| 1.2C Optimizer | student_muon | 20.5155 | 0.00888071 | 0.07253133 |
| 1.2D Model | isotropic + no alpha | 26.3870 | 0.00229775 | 0.03550264 |
| 1.2D Model | anisotropic + no alpha | 28.1458 | 0.00153256 | 0.02941568 |
| 1.2D Model | isotropic + alpha | 27.5348 | 0.00176407 | 0.02745778 |
| 1.2D Model | anisotropic + alpha | 30.5935 | 0.00087226 | 0.01990399 |
| 1.2E Scheduler | cosine | 29.0972 | 0.00123107 | 0.02333520 |
| 1.2E Scheduler | warmup_cosine | 28.3661 | 0.00145676 | 0.02514262 |
| 1.2E Scheduler | step_decay | 28.7539 | 0.00133232 | 0.02418082 |

### 结果分析

Loss 消融中，`l1` 和 `charbonnier` 的 MAE 略低于 MSE 基线，但 PSNR/MSE 更差。PSNR 本质由 MSE 决定，因此直接优化 MSE 更符合评测目标；鲁棒损失会削弱大残差区域的惩罚，使最终均方误差不占优。

初始化消融中，`grid` 最好，说明均匀覆盖目标图像可以减少随机初始化造成的空洞区域，前期更稳定。`image_sample` 使用目标颜色初始化，最终也优于随机基线，但由于本渲染器是加性累积，采样点过密区域容易初期偏亮，因此在该单图上略弱于均匀网格。

优化器消融中，`student_adam` 与 `torch_adam` 非常接近，说明手写 Adam 的一阶/二阶矩和偏差修正实现正确。SGD 与 Momentum 在此任务中表现很差，主要因为不同参数组的梯度尺度差异大，位置、尺度、颜色和 alpha 需要自适应步长。AdamW 的权重衰减略微降低效果；Muon 针对大矩阵权重设计，而本任务参数多为 `[N, 2]` 或 `[N, 3]`，正交化收益有限。

模型设计消融显示各向异性和 alpha 都有效，二者同时开启最佳。各向异性提升了细长结构的表达能力，alpha 允许每个高斯控制贡献强度，减少颜色与密度耦合；二者互补。

学习率调度器在默认 Adam 设置下没有超过 constant。该任务使用全量图像优化，不存在 minibatch 噪声；Adam 已经提供自适应更新，过早衰减学习率反而降低后期修正能力。

## 任务 2 配置与结果

任务 2 最终使用 `image_sample` 初始化、`mse` loss、`torch_adam` 优化器、`constant` scheduler，开启各向异性和 alpha。100 步 sprint 使用 `lr=0.08`；500 步 standard 使用 `lr=0.05`。

最终自测命令：

```bash
python experiments/run_assignment2.py --track both --output outputs_assignment2_final_gpu
```

| Track | R1 | R2 | R3 | S1 | S2 | S3 | 平均 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task2A 100 steps | 29.2814 | 26.9206 | 27.7189 | 27.6807 | 31.6878 | 28.8327 | 28.6870 |
| Task2B 500 steps | 32.8823 | 28.9432 | 31.6249 | 32.9704 | 38.5795 | 35.3107 | 33.3852 |

100 步设置侧重快速收敛，因此使用更大学习率。500 步设置使用较稳的 `0.05`，在合成图上达到较高 PSNR，同时真实图保持稳定。尝试更大的 `0.08` 会导致合成图明显过冲，平均 PSNR 下降；`step_decay` 也会过早降低学习率，影响最终质量。

## 总结

本实验中最关键的设计是初始化、Adam 自适应优化和模型表达能力。`grid` 对单张火烈鸟消融最优，`image_sample` 在任务 2 多图平均上更好，尤其能利用合成目标的颜色先验。各向异性和 alpha 明显提升表达能力；学习率调度器在该全量优化问题中收益不明显。
