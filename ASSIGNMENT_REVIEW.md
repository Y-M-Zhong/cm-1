# CM2026 Project 1 作业 Review 说明

本文档用于 review 当前作业实现：先解释这个作业要做什么，再说明我改了哪些文件、每个实现做了什么、当前验证结果如何、每条 Git 记录对应什么修改，以及如何重新复现实验。项目目录已经按提交要求清理，训练输出目录 `outputs*/` 和 Python 缓存 `__pycache__/` 都属于可再生成文件，不应作为代码提交内容保留。

## 1. 作业完整介绍

### 1.1 作业目标

这个项目是一个 2D Gaussian Splatting 图像拟合作业。给定一张目标图像，程序会维护一组可训练的 2D Gaussian 点，每个 Gaussian 有位置、尺度、旋转、透明度和颜色等参数。训练时，渲染器把这些 Gaussian 渲染成一张预测图像，然后通过 loss 比较预测图和目标图，再用优化器反向传播更新 Gaussian 参数。

简化后的训练流程是：

```text
目标图像 -> 初始化 Gaussian 参数 -> 可微渲染 -> 计算 loss -> 反向传播 -> 优化器更新 -> 重复迭代 -> 输出重建图和指标
```

最终评价指标主要是：

- `MSE`：预测图与目标图的均方误差，越低越好。
- `MAE`：预测图与目标图的平均绝对误差，越低越好。
- `PSNR`：由 MSE 换算得到，越高越好。任务 2 直接按平均 PSNR 阈值评分。

### 1.2 作业任务

作业分为两个大任务。

任务 1 是代码实现和消融实验，占 70 分：

- 任务 1.1：补全 `student/` 目录里的核心算法实现。
- 任务 1.2：做消融实验，比较不同 loss、初始化、优化器、模型开关和学习率调度器对结果的影响。

任务 2 是配置优化，占 30 分：

- 在固定约束下调 loss、初始化器、优化器、学习率、scheduler、模型开关等配置。
- 分别优化 100 步快速收敛任务和 500 步最终质量任务。
- 在 6 张测试图上取平均 PSNR 评分。

### 1.3 项目主要文件怎么配合

| 文件 | 作用 |
| --- | --- |
| `config.py` | 全局配置，包括目标图、模型、训练步数、loss、optimizer、scheduler、initializer |
| `train.py` | 默认训练主循环 |
| `models.py` | 2D Gaussian 参数容器和参数约束 |
| `renderer.py` | 可微 2D Gaussian 渲染器 |
| `target_generators.py` | 目标图像或合成 Gaussian 数据生成 |
| `student/losses.py` | 作业要求实现的 loss |
| `student/optimizers.py` | 作业要求实现的优化器 |
| `student/initializers.py` | 作业要求实现的初始化策略 |
| `student/schedulers.py` | 作业要求实现的学习率调度器 |
| `experiments/assignment2_settings.py` | 任务 2 最终配置 |
| `experiments/run_assignment2.py` | 任务 2 自测脚本 |
| `experiments/run_ablation.py` | 我新增的任务 1.2 消融实验脚本 |

### 1.4 评分和提交重点

代码实现部分要求不能直接调用 PyTorch 中对应的现成 loss 或 optimizer。可以使用 PyTorch 张量运算和自动求导，但核心公式要自己写。

提交重点是：

- `student/losses.py`
- `student/optimizers.py`
- `student/initializers.py`
- `student/schedulers.py`
- `experiments/assignment2_settings.py`
- 实验报告或 review 文档中记录的消融实验和任务 2 结果

## 2. 当前 Git 记录说明

当前主线分支应为 `main`。修复前仓库一度处于 detached HEAD 状态，这是因为工作区被切到了初始提交 `dd15f5c`，后续又在游离 HEAD 上产生了两个提交。现在已经切回 `main`，并把需要保留的 review 文档接回了 `main`。

当前 `main` 的核心提交记录如下：

```text
dd15f5c Initial local project snapshot
e3dce4b Implement student losses and schedulers
6f2eb13 Implement student optimizers
796c8ce Implement Gaussian initializers
460397c Tune assignment 2 settings
a41e60f Add ablation experiment runner
29de021 Add experiment report
```

### `dd15f5c Initial local project snapshot`

这是作业项目的初始快照，包含老师/项目提供的原始代码、数据、文档和 TODO stub。这个提交里主要内容包括：

- `README.md`、`docs/`：作业说明、任务 1/2 说明、问题公式化说明。
- `config.py`：默认训练配置。
- `train.py`：训练主循环。
- `models.py`、`renderer.py`：Gaussian 模型和渲染器。
- `target_generators.py`、`data/`：目标图像和合成数据。
- `student/*.py`：作业需要学生补全的 TODO 文件。
- `experiments/run_assignment2.py`、`experiments/assignment2_settings.py`：任务 2 自测入口和待调配置。

这个提交可以理解为“没做作业之前的原始项目状态”。

### `e3dce4b Implement student losses and schedulers`

这一提交完成了两个相对独立、公式较短的模块：

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `student/losses.py` | 实现 `l1`、`charbonnier`、`mse_l1`、`mse_edge` | 满足任务 1.1 至少 3 个 loss 的要求，并为任务 1.2 loss 消融提供可选项 |
| `student/schedulers.py` | 实现 `cosine`、`warmup_cosine`、`step_decay` | 满足任务 1.1 三个 scheduler 的要求，并支持任务 1.2E 消融 |

验证方式：

- `py_compile` 语法检查通过。
- 小张量反传 smoke test 通过。
- scheduler 输出范围合理，能够被训练循环作为学习率倍率使用。

### `6f2eb13 Implement student optimizers`

这一提交补全了所有学生优化器：

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `student/optimizers.py` | 新增 `_iter_params`、`_zero_grad` 辅助函数 | 统一遍历参数组和手动清梯度，避免依赖 `torch.optim.Optimizer` |
| `student/optimizers.py` | 实现 `StudentSGD` | 满足 SGD 实现要求 |
| `student/optimizers.py` | 实现 `StudentMomentum` | 满足 Momentum 实现要求 |
| `student/optimizers.py` | 实现 `StudentAdam` | 满足 Adam 实现要求，并作为和 `torch_adam` 对齐的关键正确性检查 |
| `student/optimizers.py` | 实现 `StudentAdamW` | 满足 AdamW 实现要求，使用 decoupled weight decay |
| `student/optimizers.py` | 实现 `StudentMuon` | 满足 Muon 实现要求，使用 Newton-Schulz 近似正交化 |

验证方式：

- `StudentAdam` 和 `torch.optim.Adam` 在小张量优化测试中 20 步后最大参数差异约 `1.19e-7`。
- 5 个优化器都通过 finite-value smoke test。
- `student_adam` 可以跑通 2 步训练并降低 loss。

### `796c8ce Implement Gaussian initializers`

这一提交补全初始化策略：

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `student/initializers.py` | 实现 `GridGaussianInitializer` | 让 Gaussian 初始时均匀覆盖图像，减少随机空洞 |
| `student/initializers.py` | 实现 `ImageSampleGaussianInitializer` | 利用目标图像颜色和边缘信息初始化位置与颜色，加速早期收敛 |

验证方式：

- 检查初始化后的中心在 `[0, 1]` 范围内、scale 为正、颜色形状正确。
- `grid` 和 `image_sample` 都能跑通短训练。

### `460397c Tune assignment 2 settings`

这一提交调整任务 2 配置：

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `experiments/assignment2_settings.py` | Task2A 使用 `image_sample`，`lr=0.08` | 100 步任务更看重快速收敛，较大学习率在多图平均上更好 |
| `experiments/assignment2_settings.py` | Task2B 使用 `image_sample`，`lr=0.05` | 500 步任务更看重稳定最终质量，`0.05` 比 `0.08` 更稳 |

验证方式：

- 使用 `experiments/run_assignment2.py --track both` 在 GPU 上验证。
- 最终记录：Task2A 平均 `28.6870 dB`，Task2B 平均 `33.3852 dB`。

### `a41e60f Add ablation experiment runner`

这一提交新增任务 1.2 自动消融脚本：

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `experiments/run_ablation.py` | 新增统一消融实验 runner | 避免手动改配置导致实验条件不一致；自动输出 `summary.csv`、预测图、对比图和 loss 曲线数据 |

验证方式：

- 使用 `python experiments/run_ablation.py --output outputs_ablation_gpu` 跑完整 1.2A-E 消融。
- 输出结果已整理在本文档第 5.3 节。

### `29de021 Add experiment report`

这一提交新增实验报告：

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `report.md` | 新增报告草稿 | 记录实验设置、任务 1.2 消融、任务 2 结果和分析 |
| `report.pdf` | 由 `report.md` 转换生成 | README 要求实验报告以 PDF 格式提交 |

### 本次新增的 review 文档提交

在修复 Git 状态后，`ASSIGNMENT_REVIEW.md` 会作为一个新的 `main` 提交加入。它不是作业核心代码，而是帮助 review 的说明文档，包含：

| 文件 | 修改类型 | 具体内容 |
| --- | --- | --- |
| `ASSIGNMENT_REVIEW.md` | 新增 | 作业介绍、文件级实现说明、每次 Git 提交说明、验证结果、复现命令、目录清理说明 |

### 为什么 `ASSIGNMENT_REVIEW.md` 不在 Git 更改里

如果它已经提交，那么不出现在 `git status --short` 中是正常的。原因是：

- `git status` 只显示当前工作区相对于最新提交的“未提交变化”。
- 如果一个文件已经提交，并且当前没有继续修改，它就不会出现在 `git status --short` 里。

可以用下面命令确认它已经被 Git 跟踪：

```bash
git ls-files ASSIGNMENT_REVIEW.md
```

当前输出是：

```text
ASSIGNMENT_REVIEW.md
```

如果刚修改完但还没提交，它会显示为 `M ASSIGNMENT_REVIEW.md`；如果已经提交且工作区干净，它不会显示。

## 3. 当前交付文件概览

作业要求中的必交代码文件已覆盖：

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `student/losses.py` | 已实现 | 新增 4 个 loss：`l1`、`charbonnier`、`mse_l1`、`mse_edge` |
| `student/optimizers.py` | 已实现 | 新增 5 个手写优化器：SGD、Momentum、Adam、AdamW、Muon |
| `student/initializers.py` | 已实现 | 新增 `grid` 和 `image_sample` 初始化 |
| `student/schedulers.py` | 已实现 | 新增 `cosine`、`warmup_cosine`、`step_decay` |
| `experiments/assignment2_settings.py` | 已调参 | 配置任务 2 的 100 步和 500 步最终设置 |
| `experiments/run_ablation.py` | 新增 | 自动运行任务 1.2 的消融实验并输出 CSV/图片 |
| `ASSIGNMENT_REVIEW.md` | 新增 | 本 review 文档 |

## 4. 代码实现细节

### 4.1 Loss 实现：`student/losses.py`

实现内容：

- `l1_loss`：直接计算 `mean(abs(prediction - target))`。
- `charbonnier_loss`：计算 `mean(sqrt(diff^2 + eps^2))`，是 L1 的平滑版本，避免零点不可导问题。
- `mse_l1_loss`：组合 MSE 与 L1，形式为 `mse + l1_weight * l1`。
- `mse_edge_loss`：在 MSE 基础上加入水平和垂直一阶差分的边缘误差，用于鼓励局部结构一致。

实现约束：

- 没有调用 `torch.nn.functional` 的现成 loss。
- 只使用 PyTorch 张量运算，支持 autograd。

### 4.2 Optimizer 实现：`student/optimizers.py`

新增了公共辅助函数：

- `_iter_params(param_groups)`：统一遍历参数组。
- `_zero_grad(param_groups)`：手动清空梯度，避免依赖 `torch.optim.Optimizer`。

实现内容：

- `StudentSGD`：按 `param -= lr * grad` 更新。
- `StudentMomentum`：维护每个参数的速度 `v = momentum * v + grad`，再用速度更新参数。
- `StudentAdam`：维护一阶矩 `m` 和二阶矩 `v`，实现 bias correction，并按 Adam 公式更新。
- `StudentAdamW`：在 Adam 更新前做 decoupled weight decay，即 `param *= (1 - lr * weight_decay)`。
- `StudentMuon`：维护动量 buffer，并对矩阵形状更新量使用 Newton-Schulz 近似正交化；非矩阵参数退化为归一化更新。

重要校验：

- `StudentAdam` 与 `torch.optim.Adam` 在小张量 20 步测试后的最大差异约 `1.19e-7`，说明 Adam 公式和 bias correction 实现正确。

### 4.3 Initializer 实现：`student/initializers.py`

`GridGaussianInitializer`：

- 将 `N` 个 Gaussian 按接近正方形的网格均匀铺在图像平面。
- 初始 scale 根据网格宽高设置，使 Gaussian 覆盖相对均匀。
- alpha 开启时初值为 `0.35`，颜色设为中灰 `0.5`。

`ImageSampleGaussianInitializer`：

- 使用目标图像亮度和边缘强度构造采样权重。
- 按权重从像素中采样 Gaussian 中心，并加入半像素级 jitter。
- 颜色直接取目标图像采样像素颜色。
- scale 设为较小范围，alpha 为 `0.2`，避免加性渲染初期过曝。

设计考虑：

- `grid` 更强调均匀覆盖，适合作为稳定初始化。
- `image_sample` 更强调目标颜色先验，在任务 2 的多图平均上表现更好。

### 4.4 Scheduler 实现：`student/schedulers.py`

实现内容：

- `cosine_schedule`：从 `1.0` 余弦退火到 `0.05`。
- `warmup_cosine_schedule`：前 `10%` 步数线性 warmup 到 `1.0`，之后余弦退火到 `0.05`。
- `step_decay_schedule`：每 50 步乘以 `0.5`。

注意：

- 这些 scheduler 返回的是学习率倍率，训练循环会用它乘以每个参数组的 `base_lr`。

### 4.5 任务 2 配置：`experiments/assignment2_settings.py`

最终配置：

| Track | steps | loss | initializer | optimizer | lr | scheduler | anisotropic | alpha |
| --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| Task2A sprint | 100 | `mse` | `image_sample` | `torch_adam` | `0.08` | `constant` | True | True |
| Task2B standard | 500 | `mse` | `image_sample` | `torch_adam` | `0.05` | `constant` | True | True |

保留硬约束：

- seed = `42`
- image size = `128`
- num gaussians = `1000`
- background = `(0.0, 0.0, 0.0)`
- steps 分别为 `100` 和 `500`

### 4.6 消融脚本：`experiments/run_ablation.py`

新增该脚本是为了让任务 1.2 的实验可复现、配置一致。

脚本行为：

- 默认使用 README 中规定的任务 1.2 基线。
- 每次只改变一个模块。
- 跑完后保存：
  - `summary.csv`
  - 每个 case 的 `prediction.png`
  - 每个 case 的 `comparison.png`
  - 每个 case 的 `losses.csv`

运行方式：

```bash
python experiments/run_ablation.py --output outputs_ablation_gpu
```

## 5. 验证记录

### 5.1 静态检查

已运行：

```bash
python -m py_compile student/losses.py student/optimizers.py student/initializers.py student/schedulers.py experiments/assignment2_settings.py experiments/run_assignment2.py experiments/run_ablation.py train.py
```

结果：通过。

### 5.2 Adam 数值对齐

已用一个小张量问题对比 `StudentAdam` 与 `torch.optim.Adam`，20 步后参数最大差异：

```text
1.1920928955078125e-07
```

结论：`StudentAdam` 与 PyTorch Adam 对齐到浮点误差量级。

### 5.3 任务 1.2 消融结果

运行命令：

```bash
python experiments/run_ablation.py --output outputs_ablation_gpu
```

结果：

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

简要解读：

- Loss：MSE 的 PSNR 最好，因为 PSNR 直接由 MSE 决定；L1/Charbonnier 的 MAE 略好但 MSE 更差。
- Initializer：`grid` 在单张 flamingo 消融上最好，说明均匀覆盖很重要；`image_sample` 也优于 random。
- Optimizer：`student_adam` 接近 `torch_adam`；SGD/Momentum 很差，说明该问题需要自适应学习率。
- Model：各向异性和 alpha 都有明显帮助，同时开启最佳。
- Scheduler：constant 最好；全量优化 + Adam 下，过早衰减学习率会损害后期拟合。

### 5.4 任务 2 最终结果

运行命令：

```bash
python experiments/run_assignment2.py --track both --output outputs_assignment2_final_gpu
```

Task2A 100 steps：

| Image | PSNR |
| --- | ---: |
| R1_flamingo | 29.2814 |
| R2_starry_night | 26.9206 |
| R3_parkour | 27.7189 |
| S1_night_cityscape | 27.6807 |
| S2_mandala | 31.6878 |
| S3_coral_reef | 28.8327 |
| Average | 28.6870 |

Task2B 500 steps：

| Image | PSNR |
| --- | ---: |
| R1_flamingo | 32.8823 |
| R2_starry_night | 28.9432 |
| R3_parkour | 31.6249 |
| S1_night_cityscape | 32.9704 |
| S2_mandala | 38.5795 |
| S3_coral_reef | 35.3107 |
| Average | 33.3852 |

按 README 阈值估算：

- Task2A 平均 `28.6870 dB`，达到 `>= 28.0 dB` 档。
- Task2B 平均 `33.3852 dB`，达到 `>= 33.0 dB` 档，距离 `33.5 dB` 满分档很近。

## 6. 当前目录清理说明

已清理内容：

- 所有 `outputs*` 目录：训练结果、调参结果、消融输出，均可重新生成。
- `__pycache__` 目录：Python 编译缓存，可自动生成。

保留内容：

- 源代码与作业配置。
- `ASSIGNMENT_REVIEW.md`，用于 review 当前实现和结果。

注意：

- 若重新运行验证命令，会再次生成 `outputs_assignment2_final_gpu/`、`outputs_ablation_gpu/` 等目录；这些目录已被 `.gitignore` 的 `outputs*/` 规则忽略，不需要提交。

## 7. 建议 review 顺序

1. 先看 `student/losses.py` 和 `student/schedulers.py`，确认公式实现简单直接。
2. 再看 `student/optimizers.py`，重点检查 Adam bias correction、AdamW decoupled weight decay、Muon 的 Newton-Schulz 更新。
3. 看 `student/initializers.py`，确认 `grid` 和 `image_sample` 没有读取隐藏答案或数据作弊。
4. 看 `experiments/assignment2_settings.py`，确认任务 2 没有修改硬约束。
5. 如需复现结果，优先用 GPU 跑：

```bash
python experiments/run_assignment2.py --track both --output outputs_assignment2_final_gpu
python experiments/run_ablation.py --output outputs_ablation_gpu
```

在当前环境中，沙箱内 PyTorch 看不到 CUDA；需要在沙箱外运行时才会自动使用 GPU。此前验证时 GPU 可见，`torch.cuda.is_available()` 为 `True`。
