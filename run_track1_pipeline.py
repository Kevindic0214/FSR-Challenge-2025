#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FSR-2025 Hakka ASR Track 1 Complete Pipeline

Usage:
    python run_track1_pipeline.py [--stage_start 1] [--stage_end 4] [--config config.json]

Stages:
    1: Data preparation
    2: Training 
    3: Inference
    4: Evaluation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    "root_dir": "HAT-Vol2",
    "manifests_dir": "HAT-Vol2/manifests_track1",
    "model_dir": "runs/track1/whisper_v2_lora",
    "pred_file": "predictions_track1.csv",
    "eval_key_dir": "FSR-2025-Hakka-evaluation-key",
    "eval_data_dir": "FSR-2025-Hakka-evaluation",
    
    # Stage 1: Data preparation
    "prepare": {
        "dev_speakers": 12,
        "drop_mispronounce": True,
        "relative_audio_path": True,
        "seed": 1337
    },
    
    # Stage 2: Training
    "training": {
        "base_model": "openai/whisper-large-v2",
        "epochs": 3,
        "lr": 1e-4,
        "batch_size": 2,
        "grad_accum": 16,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    
    # Stage 3: Inference
    "inference": {
        "beams": 1,
        "temperature": 0.0,
        "length_penalty": 1.0,
        "max_new_tokens": 256
    },
    
    # Stage 4: Evaluation
    "evaluation": {
        "dump_err": "errors_track1.jsonl",
        "aligned_out": "aligned_track1.csv"
    }
}

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False

def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    print("=== Checking Dependencies ===")
    
    # Check Python
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found. Please install: pip install torch")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not found. Please install: pip install transformers")
        return False
        
    try:
        import peft
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError:
        print("✗ PEFT not found. Please install: pip install peft")
        return False
    
    print("")
    return True

def check_data_files(config: Dict[str, Any]) -> bool:
    """Check if required data files exist."""
    print("=== Checking Data Files ===")
    
    root_dir = Path(config["root_dir"])
    if not root_dir.exists():
        print(f"✗ Root directory not found: {root_dir}")
        print("Please ensure HAT-Vol2 is in the current directory.")
        return False
    print(f"✓ Root directory found: {root_dir}")
    
    eval_data = Path(config["eval_data_dir"])
    if not eval_data.exists():
        print(f"✗ Evaluation data not found: {eval_data}")
        print("Warning: Inference stage will fail without evaluation data.")
    else:
        print(f"✓ Evaluation data found: {eval_data}")
    
    eval_key = Path(config["eval_key_dir"])
    if not eval_key.exists():
        print(f"✗ Evaluation keys not found: {eval_key}")
        print("Warning: Evaluation stage will fail without evaluation keys.")
    else:
        print(f"✓ Evaluation keys found: {eval_key}")
    
    print("")
    return True

def stage_1_prepare(config: Dict[str, Any]) -> bool:
    """Stage 1: Data preparation."""
    print("=== Stage 1: Data Preparation ===")
    
    cfg = config["prepare"]
    cmd = [
        "python", "prepare_hakka_track1.py",
        "--root", config["root_dir"],
        "--out_dir", config["manifests_dir"],
        "--dev_speakers", str(cfg["dev_speakers"]),
        "--seed", str(cfg["seed"])
    ]
    
    if cfg["drop_mispronounce"]:
        cmd.append("--drop_mispronounce")
    if cfg["relative_audio_path"]:
        cmd.append("--relative_audio_path")
    
    success = run_command(cmd, "Data preparation")
    if success:
        print(f"Data preparation completed. Check {config['manifests_dir']}/")
    print("")
    return success

def stage_2_training(config: Dict[str, Any]) -> bool:
    """Stage 2: Training."""
    print("=== Stage 2: Training ===")
    
    # Check if training data exists
    train_file = Path(config["manifests_dir"]) / "train.jsonl"
    dev_file = Path(config["manifests_dir"]) / "dev.jsonl"
    
    if not train_file.exists():
        print(f"✗ Training data not found: {train_file}")
        print("Please run stage 1 first.")
        return False
    
    cfg = config["training"]
    cmd = [
        "python", "train_whisper_lora_track1.py",
        "--train_jsonl", str(train_file),
        "--dev_jsonl", str(dev_file),
        "--root", config["root_dir"],
        "--out_dir", config["model_dir"],
        "--base_model", cfg["base_model"],
        "--epochs", str(cfg["epochs"]),
        "--lr", str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--grad_accum", str(cfg["grad_accum"]),
        "--lora_r", str(cfg["lora_r"]),
        "--lora_alpha", str(cfg["lora_alpha"]),
        "--lora_dropout", str(cfg["lora_dropout"])
    ]
    
    success = run_command(cmd, "Training")
    if success:
        print(f"Training completed. Model saved to {config['model_dir']}/")
    print("")
    return success

def stage_3_inference(config: Dict[str, Any]) -> bool:
    """Stage 3: Inference."""
    print("=== Stage 3: Inference ===")
    
    # Check if model exists
    model_dir = Path(config["model_dir"])
    if not model_dir.exists():
        print(f"✗ Trained model not found: {model_dir}")
        print("Please run stage 2 first.")
        return False
    
    # Check if evaluation data exists
    eval_data = Path(config["eval_data_dir"])
    if not eval_data.exists():
        print(f"✗ Evaluation data not found: {eval_data}")
        return False
    
    cfg = config["inference"]
    cmd = [
        "python", "infer_track1.py",
        "--eval_root", config["eval_data_dir"],
        "--outfile", config["pred_file"],
        "--model", config["training"]["base_model"],
        "--lora_dir", config["model_dir"],
        "--beams", str(cfg["beams"]),
        "--temperature", str(cfg["temperature"]),
        "--length_penalty", str(cfg["length_penalty"]),
        "--max_new_tokens", str(cfg["max_new_tokens"])
    ]
    
    success = run_command(cmd, "Inference")
    if success:
        print(f"Inference completed. Results saved to {config['pred_file']}")
    print("")
    return success

def stage_4_evaluation(config: Dict[str, Any]) -> bool:
    """Stage 4: Evaluation."""
    print("=== Stage 4: Evaluation ===")
    
    # Check if predictions exist
    pred_file = Path(config["pred_file"])
    if not pred_file.exists():
        print(f"✗ Prediction file not found: {pred_file}")
        print("Please run stage 3 first.")
        return False
    
    # Check if evaluation keys exist
    eval_key_dir = Path(config["eval_key_dir"])
    if not eval_key_dir.exists():
        print(f"✗ Evaluation key directory not found: {eval_key_dir}")
        return False
    
    cfg = config["evaluation"]
    cmd = [
        "python", "eval_track1_cer.py",
        "--key_dir", config["eval_key_dir"],
        "--hyp", config["pred_file"],
        "--dump_err", cfg["dump_err"],
        "--aligned_out", cfg["aligned_out"]
    ]
    
    success = run_command(cmd, "Evaluation")
    if success:
        print("Evaluation completed.")
        print(f"- Error analysis: {cfg['dump_err']}")
        print(f"- Aligned results: {cfg['aligned_out']}")
    print("")
    return success

def main():
    parser = argparse.ArgumentParser(description="FSR-2025 Hakka ASR Track 1 Pipeline")
    parser.add_argument("--stage_start", type=int, default=1, choices=[1,2,3,4],
                        help="Starting stage (1=prepare, 2=train, 3=infer, 4=eval)")
    parser.add_argument("--stage_end", type=int, default=4, choices=[1,2,3,4],
                        help="Ending stage")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (optional)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be run without executing")
    args = parser.parse_args()
    
    if args.stage_start > args.stage_end:
        print("ERROR: stage_start must be <= stage_end")
        sys.exit(1)
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
        print(f"Loaded config from: {args.config}")
    
    print("=== FSR-2025 Hakka ASR Track 1 Pipeline ===")
    print(f"Running stages {args.stage_start} to {args.stage_end}")
    print(f"Root directory: {config['root_dir']}")
    print(f"Model output: {config['model_dir']}")
    print("")
    
    if args.dry_run:
        print("DRY RUN MODE - Commands will not be executed")
        print("")
    
    # Check dependencies and data
    if not check_dependencies():
        print("Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    if not check_data_files(config):
        print("Data file check failed. Some stages may not work.")
    
    # Execute stages
    stages = {
        1: stage_1_prepare,
        2: stage_2_training,
        3: stage_3_inference,
        4: stage_4_evaluation
    }
    
    success_count = 0
    for stage_num in range(args.stage_start, args.stage_end + 1):
        if args.dry_run:
            print(f"Would run stage {stage_num}")
            success_count += 1
        else:
            if stages[stage_num](config):
                success_count += 1
            else:
                print(f"Stage {stage_num} failed. Stopping pipeline.")
                break
    
    total_stages = args.stage_end - args.stage_start + 1
    print("=== Pipeline Summary ===")
    print(f"Completed {success_count}/{total_stages} stages successfully")
    
    if success_count == total_stages:
        print("✓ All stages completed successfully!")
        if args.stage_end >= 4 and Path(config["evaluation"]["aligned_out"]).exists():
            print(f"\nFinal results: {config['pred_file']}")
    else:
        print("✗ Pipeline incomplete due to errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
