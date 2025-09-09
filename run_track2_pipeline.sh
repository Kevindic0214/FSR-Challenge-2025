#!/bin/bash

# FSR-2025 Hakka ASR Track 2 Complete Pipeline (using infer_track2.py)
# Usage: ./run_track2_pipeline.sh [stage_start] [stage_end]
# Stages: 1=prepare, 2=train, 3=infer, 4=eval

set -euo pipefail

# Configuration
ROOT_DIR="HAT-Vol2"
MANIFESTS_DIR="${ROOT_DIR}/manifests_track2"
MODEL_DIR="exp_track2_whisper_large_lora"   # train_whisper_lora_track2.py default
PRED_FILE="predictions_track2_pinyin.csv"
EVAL_KEY_DIR="FSR-2025-Hakka-evaluation-key"
EVAL_DATA_DIR="FSR-2025-Hakka-evaluation"
BASE_MODEL="openai/whisper-large-v2"

# Parse args
STAGE_START=${1:-1}
STAGE_END=${2:-4}

echo "=== FSR-2025 Hakka ASR Track 2 Pipeline ==="
echo "Running stages $STAGE_START to $STAGE_END"
echo "Root directory: $ROOT_DIR"
echo "Model output: $MODEL_DIR"
echo ""

if ! command -v python &>/dev/null; then
  echo "ERROR: Python is not installed"; exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
  echo "ERROR: HAT-Vol2 directory not found."; exit 1
fi

# Stage 1: Prepare
if [ $STAGE_START -le 1 ] && [ $STAGE_END -ge 1 ]; then
  echo "=== Stage 1: Data Preparation ==="
  python prepare_hakka_track2.py \
    --data_root "$ROOT_DIR" \
    --out_dir "$MANIFESTS_DIR" \
    --dev_speakers 12 \
    --dev_ratio 0.10 \
    --exclude_mispronounced \
    --seed 1337
  echo "Data preparation completed. Check $MANIFESTS_DIR/"
  echo ""
fi

# Stage 2: Train
if [ $STAGE_START -le 2 ] && [ $STAGE_END -ge 2 ]; then
  echo "=== Stage 2: Training ==="
  if [ ! -f "${MANIFESTS_DIR}/train.jsonl" ]; then
    echo "ERROR: ${MANIFESTS_DIR}/train.jsonl not found. Run stage 1 first."; exit 1
  fi
  python train_whisper_lora_track2.py
  echo "Training completed. Model saved to $MODEL_DIR/"
  echo ""
fi

# Stage 3: Inference
if [ $STAGE_START -le 3 ] && [ $STAGE_END -ge 3 ]; then
  echo "=== Stage 3: Inference ==="
  if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model dir not found: $MODEL_DIR. Run stage 2 first."; exit 1
  fi
  if [ ! -d "$EVAL_DATA_DIR" ]; then
    echo "ERROR: Eval data dir not found: $EVAL_DATA_DIR"; exit 1
  fi
  python infer_track2.py \
    --eval_root "$EVAL_DATA_DIR" \
    --outfile "$PRED_FILE" \
    --model "$BASE_MODEL" \
    --lora_dir "$MODEL_DIR" \
    --beams 1 \
    --batch 1
  echo "Inference completed. Results saved to $PRED_FILE"
  echo ""
fi

# Stage 4: Eval
if [ $STAGE_START -le 4 ] && [ $STAGE_END -ge 4 ]; then
  echo "=== Stage 4: Evaluation ==="
  if [ ! -f "$PRED_FILE" ]; then
    echo "ERROR: Prediction CSV not found: $PRED_FILE. Run stage 3 first."; exit 1
  fi
  if [ ! -d "$EVAL_KEY_DIR" ]; then
    echo "ERROR: Key dir not found: $EVAL_KEY_DIR"; exit 1
  fi
  python eval_track2_ser.py \
    --key_dir "$EVAL_KEY_DIR" \
    --pred_csv "$PRED_FILE" \
    --aligned_out "aligned_track2_ser.csv"
  echo "Evaluation completed. Aligned -> aligned_track2_ser.csv"
  echo ""
fi

echo "=== Pipeline Complete ==="
echo "All requested stages finished."
