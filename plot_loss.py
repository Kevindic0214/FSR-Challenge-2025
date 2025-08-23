#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot training loss & eval metrics from HuggingFace Trainer's trainer_state.json.

- 掃描 exp 目錄下的 trainer_state.json（根目錄或 checkpoint-*/ 之中最新一個）
- 匯出 metrics.csv（step, epoch, loss, eval_loss, eval_cer, learning_rate）
- 另存兩張圖：training_loss.png、eval_metric.png
- 預設實驗目錄：exp_track1_whisper_large_lora（可用 --exp_dir 覆寫）
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import csv

def find_trainer_state(exp_dir: Path) -> Path:
    cand = []
    # 根目錄
    if (exp_dir / "trainer_state.json").exists():
        cand.append(exp_dir / "trainer_state.json")
    # checkpoint-* 下的
    for p in exp_dir.glob("checkpoint-*"):
        ts = p / "trainer_state.json"
        if ts.exists():
            cand.append(ts)
    if not cand:
        raise SystemExit(f"[ERROR] 找不到 trainer_state.json（{exp_dir} 底下）")
    # 按修改時間取最新
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

def load_logs(state_path: Path):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    logs = state.get("log_history", [])
    rows = []
    for x in logs:
        rows.append({
            "step": x.get("step"),
            "epoch": x.get("epoch"),
            "loss": x.get("loss"),
            "eval_loss": x.get("eval_loss"),
            "eval_cer": x.get("eval_cer"),
            "learning_rate": x.get("learning_rate"),
        })
    # 去掉沒有 step 的行
    rows = [r for r in rows if r["step"] is not None]
    rows.sort(key=lambda r: r["step"])
    return rows

def save_csv(rows, out_csv: Path):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step","epoch","loss","eval_loss","eval_cer","learning_rate"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def plot_training_loss(rows, out_png: Path):
    steps = [r["step"] for r in rows if r["loss"] is not None]
    losses = [r["loss"] for r in rows if r["loss"] is not None]
    if not steps:
        print("[WARN] 找不到訓練 loss 紀錄，略過 training_loss.png")
        return
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_png}")

def plot_eval_metric(rows, out_png: Path):
    # 優先 eval_cer，否則退回 eval_loss
    steps, vals, ylabel, title = [], [], None, None
    if any(r.get("eval_cer") is not None for r in rows):
        steps = [r["step"] for r in rows if r.get("eval_cer") is not None]
        vals  = [r["eval_cer"] for r in rows if r.get("eval_cer") is not None]
        ylabel = "CER (%)"
        title = "Validation CER"
    elif any(r.get("eval_loss") is not None for r in rows):
        steps = [r["step"] for r in rows if r.get("eval_loss") is not None]
        vals  = [r["eval_loss"] for r in rows if r.get("eval_loss") is not None]
        ylabel = "Eval Loss"
        title = "Validation Loss"

    if not steps:
        print("[WARN] 找不到 eval 指標（eval_cer / eval_loss），略過 eval_metric.png")
        return

    plt.figure()
    plt.plot(steps, vals)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_png}")

def print_summary(rows):
    # 簡要摘要
    best_loss = None
    best_cer = None
    for r in rows:
        if r.get("loss") is not None:
            best_loss = r["loss"] if best_loss is None else min(best_loss, r["loss"])
        if r.get("eval_cer") is not None:
            best_cer = r["eval_cer"] if best_cer is None else min(best_cer, r["eval_cer"])
    print("\n=== Summary ===")
    if best_loss is not None:
        print(f"Best training loss: {best_loss:.4f}")
    if best_cer is not None:
        print(f"Best eval CER: {best_cer:.2f}%")
    print("===============")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=Path, default=Path("exp_track1_whisper_large_lora"),
                    help="實驗資料夾（含 trainer_state.json 或 checkpoint-*/trainer_state.json）")
    args = ap.parse_args()

    state_path = find_trainer_state(args.exp_dir)
    print(f"[INFO] Using trainer_state.json: {state_path}")

    rows = load_logs(state_path)
    if not rows:
        raise SystemExit("[ERROR] log_history 內沒有可用紀錄。")

    save_csv(rows, args.exp_dir / "metrics.csv")
    print(f"[OK] Saved CSV: {args.exp_dir / 'metrics.csv'}")

    plot_training_loss(rows, args.exp_dir / "training_loss.png")
    plot_eval_metric(rows, args.exp_dir / "eval_metric.png")
    print_summary(rows)

if __name__ == "__main__":
    main()
