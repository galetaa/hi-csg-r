#!/usr/bin/env bash
set -euo pipefail

TRAIN_MANIFEST="data/experiments/htr_graph_v1/local_graph_ready/tri10k_mixed_v2/train.jsonl"
VAL_MANIFEST="data/experiments/htr_graph_v1/local_graph_ready/tri10k_mixed_v2/val.jsonl"
VOCAB="data/experiments/htr_graph_v1/local_graph_ready/tri10k_mixed_v2/vocab.json"

for MODE in gray gray_fg gray_skel gray_dist gray_fg_skel gray_fg_skel_dist
do
  OUT="outputs/htr_graph_v1/tri10k_local_ablation_${MODE}_v1"
  echo ""
  echo "=============================="
  echo "TRAIN $MODE"
  echo "=============================="

  rm -rf "$OUT"

  python -m tools.train_local_channel_ctc train \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --vocab "$VOCAB" \
    --out_dir "$OUT" \
    --epochs 80 \
    --batch_size 32 \
    --num_workers 2 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --dropout 0.1 \
    --blank_bias_init -1.0 \
    --blank_logit_penalty_start -2.0 \
    --blank_logit_penalty_end -0.4 \
    --height_bins 4 \
    --feature_size 256 \
    --channel_mode "$MODE" \
    --log_every 50 \
    --seed 60 --amp
done