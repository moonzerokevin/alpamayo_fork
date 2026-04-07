#!/usr/bin/env bash
# 与 finetune/rl/README.md §6 一致的 Cosmos-RL 启动方式；数据为单 clip overfit。
# 硬件：推荐 5×80GB（policy dp_shard=4 + rollout 1）。仅 4×80GB 时可 export ALPAMAYO_RL_POLICY_DP_SHARD=3。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

export ALPAMAYO_RL_HYDRA_CONFIG="${ALPAMAYO_RL_HYDRA_CONFIG:-alpamayo1_rvla_rl_pai}"
export ALPAMAYO_RL_CLIP_INDEX="${ALPAMAYO_RL_CLIP_INDEX:-clip_index_overfit_one.parquet}"

if [[ ! -d "$ALPAMAYO_MODEL_DIR" ]]; then
  echo "错误: ALPAMAYO_MODEL_DIR 不存在: $ALPAMAYO_MODEL_DIR"
  echo "请先 bash scripts/prepare_r1_training_assets.sh"
  exit 1
fi

if [[ ! -f "$ALPAMAYO_PAI_LOCAL_DIR/$ALPAMAYO_RL_CLIP_INDEX" ]]; then
  echo "错误: 未找到 $ALPAMAYO_PAI_LOCAL_DIR/$ALPAMAYO_RL_CLIP_INDEX"
  echo "请先 bash scripts/prepare_rl_overfit_one_scene.sh"
  exit 1
fi

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "${ALPAMAYO_RL_POLICY_DP_SHARD:-}" ]]; then
  if [[ "$NGPU" -ge 5 ]]; then
    export ALPAMAYO_RL_POLICY_DP_SHARD=4
  elif [[ "$NGPU" -eq 4 ]]; then
    export ALPAMAYO_RL_POLICY_DP_SHARD=3
  else
    echo "错误: 检测到 $NGPU 张 GPU。RL overfit 至少需要 4×80GB（推荐 5 张，见 finetune/rl/README.md）。"
    exit 1
  fi
fi

MIN_NEED=$((ALPAMAYO_RL_POLICY_DP_SHARD + 1))
if [[ "$NGPU" -lt "$MIN_NEED" ]]; then
  echo "错误: policy dp_shard=$ALPAMAYO_RL_POLICY_DP_SHARD 需要至少 $MIN_NEED 张 GPU（另 1 张给 rollout）。当前 $NGPU 张。"
  exit 1
fi

RL_CFG_SRC="$WORKSPACE_DIR/finetune/rl/toml/alpamayo_rvla_rl_overfit_one_scene.toml"
RL_CFG_RUN="$(mktemp)"
trap 'rm -f "$RL_CFG_RUN"' EXIT

sed \
  -e "s|__ALPAMAYO_MODEL_DIR__|${ALPAMAYO_MODEL_DIR}|g" \
  -e "s|__ALPAMAYO_RL_OUTPUT_DIR__|${ALPAMAYO_RL_OUTPUT_DIR}|g" \
  -e "s|__POLICY_DP_SHARD__|${ALPAMAYO_RL_POLICY_DP_SHARD}|g" \
  "$RL_CFG_SRC" >"$RL_CFG_RUN"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export COSMOS_LOG_LEVEL="${COSMOS_LOG_LEVEL:-INFO}"

echo "ALPAMAYO_RL_HYDRA_CONFIG=$ALPAMAYO_RL_HYDRA_CONFIG"
echo "ALPAMAYO_RL_CLIP_INDEX=$ALPAMAYO_RL_CLIP_INDEX"
echo "ALPAMAYO_RL_POLICY_DP_SHARD=$ALPAMAYO_RL_POLICY_DP_SHARD (检测到 GPU 数: $NGPU)"
echo "Cosmos-RL 日志: $ALPAMAYO_LOG_DIR"
echo "训练输出: $ALPAMAYO_RL_OUTPUT_DIR"

exec cosmos-rl \
  --config "$RL_CFG_RUN" \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
