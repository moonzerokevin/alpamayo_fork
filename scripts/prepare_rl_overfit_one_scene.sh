#!/usr/bin/env bash
# 从已下载的 PAI 中只保留 1 条 clip，供 RL overfit 验证（见 finetune/rl/README.md）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR%/scripts}"

cd "$WORKSPACE_DIR"
source scripts/setup_local_env.sh
. .venv/bin/activate

CHUNK="${PAI_CHUNKS:-3116}"
OUT_NAME="${ALPAMAYO_RL_OVERFIT_INDEX:-clip_index_overfit_one.parquet}"

if [[ ! -f "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" ]]; then
  echo "未找到 $ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet，请先: bash scripts/prepare_r1_training_assets.sh"
  exit 1
fi

python scripts/curate_pai_samples.py \
  --clip-index-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" \
  --chunk "$CHUNK" \
  --num-samples 1 \
  --output-path "$ALPAMAYO_PAI_LOCAL_DIR/$OUT_NAME"

echo "已写入单场景索引: $ALPAMAYO_PAI_LOCAL_DIR/$OUT_NAME"
echo "启动 RL: export ALPAMAYO_RL_CLIP_INDEX=$OUT_NAME && bash scripts/run_rl_overfit_one_scene.sh"
