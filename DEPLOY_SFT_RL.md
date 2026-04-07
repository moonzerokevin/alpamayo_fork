# Alpamayo SFT/RL Deployment

本目录把上游 `NVlabs/alpamayo` 的训练能力落到当前机器上，并且补了适合单卡 `H100 80G` 的最小可行脚本。

## 当前结论

- 这份 checkout 是上游 `alpamayo` 的独立训练仓，当前提交为 `ff067c438dc58cc98e6a677d4e98e6ca583c17b9`。
- 环境已在本机验证通过：
  - `uv 0.10.4`
  - `Python 3.12` 虚拟环境位于 `alpamayo/.venv`
  - `CUDA 12.8` / `nvcc` 可用
  - `flash-attn==2.8.3` 可安装
  - `torch.cuda.is_available() == True`
- 上游官方边界：
  - SFT 文档验证环境是 `8x H100 80G`
  - RL 本地验证文档要求至少 `5x 80G GPU`
- 单卡（或 1×policy + 1×rollout）上的 RL 在 10B 规模下易在 policy 前向 OOM，**不作为可行验证路径**。
- 要做「RL 是否跑通」的验证，请用 **`finetune/rl/README.md` 推荐拓扑**（多卡 FSDP policy + 独立 rollout），数据上只取 **1 条 clip overfit** 即可缩短时间（见下方 `prepare_rl_overfit_one_scene.sh` / `run_rl_overfit_one_scene.sh`）。

## 目录与脚本

- 环境变量初始化：`scripts/setup_local_env.sh`
- 推理 smoke test：`scripts/run_inference_smoke.sh`
- 下载训练资产：`scripts/prepare_r1_training_assets.sh`
- 单卡 Stage 1：`scripts/run_sft_stage1_single_h100.sh`
- 单卡 Stage 2：`scripts/run_sft_stage2_single_h100.sh`
- 单卡 Stage 2 评估：`scripts/eval_sft_stage2_single_h100.sh`
- RL 单场景 overfit 索引：`scripts/prepare_rl_overfit_one_scene.sh`
- RL overfit 试跑（多卡，对齐上游 README）：`scripts/run_rl_overfit_one_scene.sh`
- （弃用）单卡 RL：`scripts/run_rl_single_h100_experimental.sh`
- 导出 RL checkpoint：`scripts/export_rl_checkpoint.sh`

新增 Hydra/TOML 配置：

- `finetune/sft/configs/sft_stage1_single_h100.yaml`
- `finetune/sft/configs/sft_stage2_single_h100.yaml`
- `finetune/rl/toml/alpamayo_rvla_rl_overfit_one_scene.toml`
- （弃用）`finetune/rl/toml/alpamayo_rvla_rl_single_h100_experimental.toml`

## 1. 初始化环境

在 `alpamayo/` 下执行：

```bash
source scripts/setup_local_env.sh
. .venv/bin/activate
```

默认会建立这些目录：

- `artifacts/models/alpamayo_r1_training`
- `artifacts/pai`
- `artifacts/sft_stage1`
- `artifacts/sft_stage2`
- `artifacts/cosmos_rl_logs`
- `artifacts/cosmos_rl_train_output`（RL checkpoint 输出，`ALPAMAYO_RL_OUTPUT_DIR`）
- `artifacts/rl_exported_model`

## 2. Hugging Face 权限

先确保你已经获得以下 gated 资源权限：

- [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- 转换训练态目录时还会拉取 VLM 权重 [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)（gated）
- 如要试 `1.5` 的 RL，也要有 [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

然后登录：

```bash
hf auth login
```

如果你更习惯环境变量，也可以同时导出：

```bash
export HF_TOKEN=...
```

## 3. 推理 smoke test

先验证独立仓能正常加载模型：

```bash
bash scripts/run_inference_smoke.sh
```

这一步会下载示例数据和权重，首次运行较慢。

## 4. 准备训练资产

下载 R1 训练用模型目录，并拉一小份 PAI 数据：

```bash
bash scripts/prepare_r1_training_assets.sh
```

默认行为：

- 通过 `scripts/convert_release_config_to_training.py` 把 `nvidia/Alpamayo-R1-10B` 转成训练态目录
- 下载 `chunk 3116`
- 生成 `clip_index_mini.parquet`

可选环境变量：

```bash
export PAI_CHUNKS=3116
export PAI_MINI_SAMPLES=16
```

## 5. SFT 最小可行流程

### Stage 1

```bash
bash scripts/run_sft_stage1_single_h100.sh
```

这个脚本做了这些降配：

- `torchrun --nproc_per_node=1`
- 关闭 DeepSpeed
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `max_steps=20`

成功判据：

- 日志里能持续输出 `loss`
- 目录 `artifacts/sft_stage1/` 下生成 `checkpoint-*`
- 没有 CUDA OOM / tokenizer / dataset 路径错误

### Stage 2

先选择一个 Stage 1 checkpoint：

```bash
export STAGE1_CKPT_DIR=/home/ubuntu/kevin/Alpamayo_15/alpamayo/artifacts/sft_stage1/checkpoint-...
bash scripts/run_sft_stage2_single_h100.sh
```

成功判据：

- `artifacts/sft_stage2/` 下生成 `checkpoint-*`
- 可以继续执行评估脚本

### Stage 2 评估

```bash
export STAGE2_CKPT_DIR=/home/ubuntu/kevin/Alpamayo_15/alpamayo/artifacts/sft_stage2/checkpoint-...
bash scripts/eval_sft_stage2_single_h100.sh
```

## 6. RL 验证流程（推荐：单场景 overfit + 官方多卡拓扑）

权威步骤与参数说明以 **`finetune/rl/README.md`** 为准：本地验证需要 **至少 5×80GB GPU**（1 个 policy 副本、`policy.parallelism.dp_shard_size=4`，再加 1 张卡跑 vLLM rollout）。

若只想确认「RL 管线能跑、reward/GRPO 在转」，不必用完整 mini 集：先只保留 **1 条 clip** 做 overfit，仍使用与 README **相同的** `cosmos-rl ... alpamayo_cosmos_rl_post_training_entry.py` 启动方式。

运行前请确认：

- 已完成 §4，`ALPAMAYO_MODEL_DIR` 与 `ALPAMAYO_PAI_LOCAL_DIR` 有效
- 机器 GPU 数：**推荐 5 张**；若只有 **4×80G**，脚本会默认使用 `ALPAMAYO_RL_POLICY_DP_SHARD=3`（仍要求 4 张卡：3 policy + 1 rollout），属减_shard 折中，不保证与官方数值一致
- （可选）`ALPAMAYO_RL_OUTPUT_DIR`、`ALPAMAYO_RL_HYDRA_CONFIG`（R1：`alpamayo1_rvla_rl_pai`；1.5：`alpamayo1_5_rvla_rl_pai`）
- （可选）`ALPAMAYO_RL_CLIP_INDEX`：默认 `clip_index_overfit_one.parquet`

执行：

```bash
bash scripts/prepare_rl_overfit_one_scene.sh
bash scripts/run_rl_overfit_one_scene.sh
```

成功时可观察 `artifacts/cosmos_rl_logs/logs_*/` 下的 `controller.log`、`policy_0.log`、`rollout_0.log`（与 README 描述一致）。

**不再推荐**：`run_rl_single_h100_experimental.sh` / `alpamayo_rvla_rl_single_h100_experimental.toml`（单卡 policy 易 OOM）。

## 7. 导出 RL checkpoint

如果 RL 试跑成功并产出 policy shard：

```bash
export COSMOS_POLICY_CKPT_DIR=/path/to/checkpoints/step_N/policy
bash scripts/export_rl_checkpoint.sh
```

导出的 Hugging Face 目录位于：

- `artifacts/rl_exported_model`

## 8. 回接到 AlpaSim

### 结论先说

- `alpasim` 当前的 `ar1` wrapper 直接调用 `AlpamayoR1.from_pretrained(...)`
- 因此它最自然消费的是“完整可推理模型目录”
- 这与 SFT Stage 2 产物天然兼容
- RL 导出的 checkpoint 仅覆盖 VLM 主干，不是完整 action expert，因此不能直接当作最终 `ar1` 推理模型替掉 `checkpoint_path`

### 推荐回接路径

1. 先把 `SFT Stage 2` 产物作为 `alpasim` 的 `checkpoint_path`
2. 如果你后面要利用 RL 结果，建议把 RL 作为 VLM 后训练，再继续做 Stage 2 / expert 训练，产出完整可推理目录后再接入 `alpasim`

### 在 `alpasim` 中本地验证

`alpasim` 的默认 `ar1` 配置支持直接改成本地路径：

```yaml
model:
  checkpoint_path: "/mnt/drivers/ar1/Alpamayo-R1-10B"
```

你可以把本地导出的完整模型目录放到 host 的 `drivers` 挂载目录，或直接在生成后的 `driver-config.yaml` 里把 `checkpoint_path` 改成容器可见路径。

当前工作区里的关键参考：

- `alpasim/src/wizard/configs/driver/ar1.yaml`
- `alpasim/src/driver/src/alpasim_driver/models/ar1_model.py`
- `ALPASIM_RUNBOOK.md`

### 单场景仿真建议

先按现有 runbook 走最小场景：

```bash
cd /home/ubuntu/kevin/Alpamayo_15/alpasim/src/wizard
uv run alpasim_wizard +deploy=local \
  wizard.run_method=NONE \
  wizard.log_dir="$PWD/../../eval_ar1_custom" \
  driver=[ar1,ar1_runtime_configs] \
  runtime.simulation_config.n_rollouts=1 \
  scenes.scene_ids=[clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6]
```

然后把生成的 `driver-config.yaml` 中 `checkpoint_path` 指向你的完整模型目录，再执行：

```bash
docker compose -f ../../eval_ar1_custom/docker-compose.yaml up --abort-on-container-exit
```

## 9. 你当前最稳的主线

如果你的目标是“尽快在 `alpasim` 中跑起来自己的 Alpamayo 版本”，推荐顺序如下：

1. `run_inference_smoke.sh`
2. `prepare_r1_training_assets.sh`
3. `run_sft_stage1_single_h100.sh`
4. `run_sft_stage2_single_h100.sh`
5. `eval_sft_stage2_single_h100.sh`
6. 将 Stage 2 产物接入 `alpasim`
7. 把 RL 作为单独实验线推进；验证请用 README 多卡拓扑 + 必要时单场景 overfit（`run_rl_overfit_one_scene.sh`）
