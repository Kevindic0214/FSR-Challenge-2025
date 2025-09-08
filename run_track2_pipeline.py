#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FSR-2025 Hakka ASR Track 2 Complete Pipeline

Usage:
    python run_track2_pipeline.py [--stage_start 1] [--stage_end 4] [--config config.json]

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

# Default configuration (aligned to existing Track 2 scripts)
DEFAULT_CONFIG = {
    "root_dir": "HAT-Vol2",
    "manifests_dir": "HAT-Vol2/manifests_track2",
    # train_whisper_lora_track2.py saves LoRA adapter here by default
    "model_dir": "exp_track2_whisper_large_lora",
    "base_model": "openai/whisper-large-v2",
    "pred_file": "predictions_track2_pinyin.csv",
    "eval_key_dir": "FSR-2025-Hakka-evaluation-key",
    "eval_data_dir": "FSR-2025-Hakka-evaluation",

    # Stage 1: Data preparation
    "prepare": {
        "dev_speakers": 12,
        "dev_ratio": 0.10,
        "exclude_mispronounced": True,
        "relative_audio_path": False,
        "stats_out": "HAT-Vol2/manifests_track2/stats.json",
        "dev_list_out": "HAT-Vol2/manifests_track2/dev_speakers.txt",
        "seed": 1337
    },

    # Stage 3: Inference (train script is self-configured)
    "inference": {
        "beams": 1,            # greedy for speed; change if needed
        "batch": 1             # infer_hakka_pinyin_warmup.py default
    },

    # Stage 4: Evaluation
    "evaluation": {
        "aligned_out": "aligned_track2_ser.csv"
    }
}


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def check_dependencies() -> bool:
    print("=== Checking Dependencies ===")
    try:
        import torch  # noqa: F401
        print("✓ PyTorch installed")
    except ImportError:
        print("✗ PyTorch not found. Please install: pip install torch")
        return False
    try:
        import transformers  # noqa: F401
        print("✓ Transformers installed")
    except ImportError:
        print("✗ Transformers not found. Please install: pip install transformers")
        return False
    try:
        import peft  # noqa: F401
        print("✓ PEFT installed")
    except ImportError:
        print("✗ PEFT not found. Please install: pip install peft")
        return False
    print("")
    return True


def check_data_files(config: Dict[str, Any]) -> bool:
    print("=== Checking Data Files ===")
    root_dir = Path(config["root_dir"])  # HAT-Vol2
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
    print("=== Stage 1: Data Preparation (Track 2) ===")
    cfg = config["prepare"]
    cmd = [
        "python", "prepare_hakka_track2.py",
        "--data_root", config["root_dir"],
        "--out_dir", config["manifests_dir"],
        "--dev_speakers", str(cfg["dev_speakers"]),
        "--dev_ratio", str(cfg["dev_ratio"]),
        "--seed", str(cfg["seed"]),
    ]
    if cfg.get("exclude_mispronounced", False):
        cmd.append("--exclude_mispronounced")
    if cfg.get("relative_audio_path", False):
        cmd.append("--relative_audio_path")
        cmd += ["--audio_root", config["root_dir"]]
    if cfg.get("stats_out"):
        cmd += ["--stats_out", cfg["stats_out"]]
    if cfg.get("dev_list_out"):
        cmd += ["--dev_list_out", cfg["dev_list_out"]]

    ok = run_command(cmd, "Data preparation")
    if ok:
        print(f"Data preparation completed. Check {config['manifests_dir']}/")
    print("")
    return ok


def stage_2_training(config: Dict[str, Any]) -> bool:
    print("=== Stage 2: Training (Track 2) ===")
    mani_dir = Path(config["manifests_dir"]).resolve()
    if not (mani_dir / "train.jsonl").exists():
        print(f"✗ Training data not found: {mani_dir / 'train.jsonl'}")
        print("Please run stage 1 first.")
        return False
    # Pass explicit paths; if relative audio used in manifests, provide audio_root
    cmd = [
        "python", "train_whisper_lora_track2.py",
        "--train_jsonl", str(mani_dir / "train.jsonl"),
        "--dev_jsonl", str(mani_dir / "dev.jsonl"),
        "--out_dir", config["model_dir"],
        "--base_model", config["base_model"],
    ]
    if config["prepare"].get("relative_audio_path", False):
        cmd += ["--audio_root", config["root_dir"]]
    ok = run_command(cmd, "Training")
    if ok:
        print(f"Training completed. Model saved to {config['model_dir']}/")
    print("")
    return ok


def stage_3_inference(config: Dict[str, Any]) -> bool:
    print("=== Stage 3: Inference (Track 2) ===")
    model_dir = Path(config["model_dir"]).resolve()
    if not model_dir.exists():
        print(f"✗ Trained model not found: {model_dir}")
        print("Please run stage 2 first.")
        return False
    eval_data = Path(config["eval_data_dir"]).resolve()
    if not eval_data.exists():
        print(f"✗ Evaluation data not found: {eval_data}")
        return False
    cfg = config["inference"]
    cmd = [
        "python", "infer_hakka_pinyin_warmup.py",
        "--eval_root", str(eval_data),
        "--lora_dir", str(model_dir),
        "--outfile", config["pred_file"],
        "--model", config["base_model"],
        "--beams", str(cfg["beams"]),
        "--batch", str(cfg["batch"]),
    ]
    ok = run_command(cmd, "Inference")
    if ok:
        print(f"Inference completed. Results saved to {config['pred_file']}")
    print("")
    return ok


def stage_4_evaluation(config: Dict[str, Any]) -> bool:
    print("=== Stage 4: Evaluation (Track 2) ===")
    the_pred = Path(config["pred_file"]).resolve()
    if not the_pred.exists():
        print(f"✗ Prediction file not found: {the_pred}")
        print("Please run stage 3 first.")
        return False
    key_dir = Path(config["eval_key_dir"]).resolve()
    if not key_dir.exists():
        print(f"✗ Evaluation key directory not found: {key_dir}")
        return False
    cfg = config["evaluation"]
    cmd = [
        "python", "eval_track2_ser.py",
        "--key_dir", str(key_dir),
        "--pred_csv", str(the_pred),
        "--aligned_out", cfg["aligned_out"],
    ]
    ok = run_command(cmd, "Evaluation")
    if ok:
        print("Evaluation completed.")
        print(f"- Aligned results: {cfg['aligned_out']}")
    print("")
    return ok


def main():
    ap = argparse.ArgumentParser(description="FSR-2025 Hakka ASR Track 2 Pipeline")
    ap.add_argument("--stage_start", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Starting stage (1=prepare, 2=train, 3=infer, 4=eval)")
    ap.add_argument("--stage_end", type=int, default=4, choices=[1, 2, 3, 4],
                    help="Ending stage")
    ap.add_argument("--config", type=str, default=None,
                    help="JSON config file (optional)")
    ap.add_argument("--dry_run", action="store_true", help="Show planned commands only")
    args = ap.parse_args()

    if args.stage_start > args.stage_end:
        print("ERROR: stage_start must be <= stage_end")
        sys.exit(1)

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            user_cfg = json.load(f)
            # shallow update is fine here
            config.update(user_cfg)
        print(f"Loaded config from: {args.config}")

    print("=== FSR-2025 Hakka ASR Track 2 Pipeline ===")
    print(f"Running stages {args.stage_start} to {args.stage_end}")
    print(f"Root directory: {config['root_dir']}")
    print(f"Model output: {config['model_dir']}")
    print("")

    if args.dry_run:
        print("DRY RUN MODE - Commands will not be executed\n")

    if not check_dependencies():
        print("Dependency check failed. Please install missing packages.")
        sys.exit(1)
    if not check_data_files(config):
        print("Data file check failed. Some stages may not work.")

    stages = {1: stage_1_prepare, 2: stage_2_training, 3: stage_3_inference, 4: stage_4_evaluation}

    success = 0
    for s in range(args.stage_start, args.stage_end + 1):
        if args.dry_run:
            print(f"Would run stage {s}")
            success += 1
        else:
            if stages[s](config):
                success += 1
            else:
                print(f"Stage {s} failed. Stopping pipeline.")
                break

    total = args.stage_end - args.stage_start + 1
    print("=== Pipeline Summary ===")
    print(f"Completed {success}/{total} stages successfully")
    if success == total:
        print("✓ All stages completed successfully!")
        if args.stage_end >= 4 and Path(DEFAULT_CONFIG["evaluation"]["aligned_out"]).exists():
            print(f"\nFinal results: {config['pred_file']}")
    else:
        print("✗ Pipeline incomplete due to errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
