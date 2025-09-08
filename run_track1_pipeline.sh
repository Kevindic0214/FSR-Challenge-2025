#!/bin/bash

# FSR-2025 Hakka ASR Track 1 Complete Pipeline
# Usage: ./run_track1_pipeline.sh [stage_start] [stage_end]
# Stages: 1=prepare, 2=train, 3=infer, 4=eval

set -euo pipefail  # Safer bash: exit on error/undef, and fail on pipeline errors

# Configuration (adjust as needed)
ROOT_DIR="HAT-Vol2"
MANIFESTS_DIR="${ROOT_DIR}/manifests_track1"
MODEL_DIR="runs/track1/whisper_v2_lora"
PRED_FILE="predictions_track1.csv"
EVAL_KEY_DIR="FSR-2025-Hakka-evaluation-key"

# Parse arguments
STAGE_START=${1:-1}  # Default start from stage 1
STAGE_END=${2:-4}    # Default end at stage 4

echo "=== FSR-2025 Hakka ASR Track 1 Pipeline ==="
echo "Running stages $STAGE_START to $STAGE_END"
echo "Root directory: $ROOT_DIR"
echo "Model output: $MODEL_DIR"
echo ""

# Check dependencies
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed"
    exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: HAT-Vol2 directory not found. Please ensure it's in the current directory."
    exit 1
fi

# Stage 1: Data Preparation
if [ $STAGE_START -le 1 ] && [ $STAGE_END -ge 1 ]; then
    echo "=== Stage 1: Data Preparation ==="
    python prepare_hakka_track1.py \
        --root "$ROOT_DIR" \
        --out_dir "$MANIFESTS_DIR" \
        --dev_speakers 12 \
        --drop_mispronounce \
        --relative_audio_path \
        --seed 1337
    
    echo "Data preparation completed. Check $MANIFESTS_DIR/"
    echo ""
fi

# Stage 2: Training
if [ $STAGE_START -le 2 ] && [ $STAGE_END -ge 2 ]; then
    echo "=== Stage 2: Training ==="
    
    # Check if training data exists
    if [ ! -f "${MANIFESTS_DIR}/train.jsonl" ]; then
        echo "ERROR: Training data not found. Please run stage 1 first."
        exit 1
    fi
    
    python train_whisper_lora_track1.py \
        --train_jsonl "${MANIFESTS_DIR}/train.jsonl" \
        --dev_jsonl "${MANIFESTS_DIR}/dev.jsonl" \
        --root "$ROOT_DIR" \
        --out_dir "$MODEL_DIR" \
        --base_model "openai/whisper-large-v2" \
        --epochs 3 \
        --lr 1e-4 \
        --batch_size 2 \
        --grad_accum 16 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05
    
    echo "Training completed. Model saved to $MODEL_DIR/"
    echo ""
fi

# Stage 3: Inference
if [ $STAGE_START -le 3 ] && [ $STAGE_END -ge 3 ]; then
    echo "=== Stage 3: Inference ==="
    
    # Check if model exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Trained model not found. Please run stage 2 first."
        exit 1
    fi
    
    # Check if evaluation data exists
    if [ ! -d "FSR-2025-Hakka-evaluation" ]; then
        echo "ERROR: Evaluation data directory not found."
        echo "Please ensure FSR-2025-Hakka-evaluation is in the current directory."
        exit 1
    fi
    
    python infer_hakka_hanzi_warmup.py \
        --eval_root "FSR-2025-Hakka-evaluation" \
        --model "openai/whisper-large-v2" \
        --lora_dir "$MODEL_DIR" \
        --outfile "$PRED_FILE" \
        --beams 5 \
        --temperature 0.0 \
        --length_penalty 1.0 \
        --max_new_tokens 256
    
    echo "Inference completed. Results saved to $PRED_FILE"
    echo ""
fi

# Stage 4: Evaluation
if [ $STAGE_START -le 4 ] && [ $STAGE_END -ge 4 ]; then
    echo "=== Stage 4: Evaluation ==="
    
    # Check if predictions exist
    if [ ! -f "$PRED_FILE" ]; then
        echo "ERROR: Prediction file not found. Please run stage 3 first."
        exit 1
    fi
    
    # Check if evaluation keys exist
    if [ ! -d "$EVAL_KEY_DIR" ]; then
        echo "ERROR: Evaluation key directory not found."
        echo "Please ensure $EVAL_KEY_DIR is in the current directory."
        exit 1
    fi
    
    python eval_track1_cer.py \
        --key_dir "$EVAL_KEY_DIR" \
        --hyp "$PRED_FILE" \
        --dump_err "errors_track1.jsonl" \
        --aligned_out "aligned_track1.csv"
    
    echo "Evaluation completed."
    echo "- Error analysis: errors_track1.jsonl"
    echo "- Aligned results: aligned_track1.csv"
    echo ""
fi

echo "=== Pipeline Complete ==="
echo "All stages completed successfully!"

# Display final results if evaluation was run
if [ $STAGE_END -ge 4 ] && [ -f "aligned_track1.csv" ]; then
    echo ""
    echo "=== Summary ==="
    echo "Prediction file: $PRED_FILE"
    echo "Error analysis: errors_track1.jsonl"
    echo "Aligned results: aligned_track1.csv"
fi
