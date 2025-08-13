#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import re
from pathlib import Path
from collections import defaultdict

# --------- 預設路徑（可用參數覆寫） ---------
DEF_ROOT = Path("HAT-Vol2")
DEF_OUT  = DEF_ROOT / "manifests_track1"

# 允許的 CSV 欄位名稱（中文）
COL_FN      = "檔名"
COL_HANZI   = "客語漢字"
COL_REMARKS = "備註"

# 會被視為「讀音有問題」而剔除（可用 --keep-mispronounce 放回）
MISPRONOUNCE_KEY = "正確讀音"

# --------- 工具函式 ---------
def load_csv_mapping(csv_path: Path):
    """
    讀一個 *_edit.csv，回傳：
    - mapping: { wav_filename -> {"hanzi": str, "remarks": str} }
    - kept, dropped_empty_text, dropped_mispronounced
    """
    mapping = {}
    kept = dropped_empty = dropped_mis = 0

    # 以 utf-8-sig 可自動忽略 BOM
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get(COL_FN) or "").strip()
            hanzi = (row.get(COL_HANZI) or "").strip()
            remarks = (row.get(COL_REMARKS) or "").strip()
            if not fn:
                continue
            if not hanzi:
                dropped_empty += 1
                continue
            mapping[fn] = {"hanzi": hanzi, "remarks": remarks}
            kept += 1
    return mapping, kept, dropped_empty, dropped_mis

def scan_all_csvs(root: Path):
    """
    在 root 下找所有 *_edit.csv，合併成一個 mapping。
    """
    glob_csvs = sorted(root.rglob("*_edit.csv"))
    all_map = {}
    stats = {"kept": 0, "dropped_empty_text": 0, "files": 0}
    for c in glob_csvs:
        m, kept, dropped_empty, _ = load_csv_mapping(c)
        all_map.update(m)
        stats["kept"] += kept
        stats["dropped_empty_text"] += dropped_empty
        stats["files"] += 1
    return all_map, glob_csvs, stats

def iter_training_wavs(root: Path):
    """
    走訪訓練音檔資料夾：
      - 例如：訓練_大埔腔30H/**/<utt>.wav、訓練_詔安腔30H/**/<utt>.wav
    """
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("訓練_")]
    for d in dirs:
        for wav in d.rglob("*.wav"):
            yield wav

def speaker_id_from_path(wav_path: Path):
    """
    說話人 = 第二層資料夾名，例如：
      訓練_大埔腔30H/DF101K2001/DF101K2001_001.wav -> DF101K2001
    """
    # wav_path.parts = (..., 訓練_大埔腔30H, DF101K2001, file.wav)
    if len(wav_path.parts) < 2:
        return "unknown"
    return wav_path.parent.name

def group_tag_from_speaker(spk: str):
    """
    用說話人前兩碼分群：DF/DM/ZF/ZM（若無法判斷就 'XX'）
    """
    return spk[:2] if len(spk) >= 2 else "XX"

def normalize_hanzi(text: str, strip_spaces=True, keep_asterisk=True):
    """
    文字正規化：
      - 預設移除所有空白（CER 對空白敏感，通常不需要）
      - 預設保留合音用的 '*'（可選移除）
    """
    t = text
    if strip_spaces:
        t = re.sub(r"\s+", "", t)
    if not keep_asterisk:
        t = t.replace("*", "")
    return t

# --------- 主流程 ---------
def main():
    ap = argparse.ArgumentParser(description="Prepare Track-1 (客語漢字) manifests (train/dev) from HAT-Vol2")
    ap.add_argument("--root", type=Path, default=DEF_ROOT, help="HAT-Vol2 根目錄")
    ap.add_argument("--out_dir", type=Path, default=DEF_OUT, help="輸出目錄（會建立）")
    ap.add_argument("--dev_speakers", type=int, default=12, help="dev 說話人數量（預設 12）")
    ap.add_argument("--seed", type=int, default=1337)
    
    # 讀音問題濾除：互斥
    misgrp = ap.add_mutually_exclusive_group()
    misgrp.add_argument("--drop_mispronounce", action="store_true",
                    help="過濾備註含『正確讀音』的樣本（建議開啟）")
    misgrp.add_argument("--keep_mispronounce", action="store_true",
                    help="保留備註含『正確讀音』的樣本（與 --drop_mispronounce 互斥）")
    
    # 合音星號：互斥
    astgrp = ap.add_mutually_exclusive_group()
    astgrp.add_argument("--keep_asterisk", action="store_true",
                    help="保留合音 '*'（預設保留）")
    astgrp.add_argument("--strip_asterisk", action="store_true",
                    help="移除合音 '*'")
    args = ap.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 讀取所有 *_edit.csv → 建立 <檔名> -> {hanzi, remarks} 對照
    mapping, csv_list, csv_stats = scan_all_csvs(args.root)
    if not mapping:
        raise SystemExit(f"[ERROR] 在 {args.root} 底下找不到 *_edit.csv 或內容為空。")

    # 掃描訓練用 wav
    entries_by_spk = defaultdict(list)
    kept, missing_audio, dropped_mis, total = 0, 0, 0, 0

    # 說話人統計（for 平衡選 dev）
    all_speakers = set()

    # 正規化參數
    keep_ast = True if args.keep_asterisk else (not args.strip_asterisk)  # 預設保留；若使用 --strip_asterisk 則 False
    drop_mispron = args.drop_mispronounce and not args.keep_mispronounce

    for wav in iter_training_wavs(args.root):
        total += 1
        fn = wav.name  # e.g., DF101K2001_001.wav
        item = mapping.get(fn)
        if item is None:
            # 沒在 CSV mapping 中，跳過
            continue

        hanzi = normalize_hanzi(item["hanzi"], strip_spaces=True, keep_asterisk=keep_ast)

        # 過濾「正確讀音」樣本
        if drop_mispron and MISPRONOUNCE_KEY in (item.get("remarks") or ""):
            dropped_mis += 1
            continue

        if not wav.exists():
            missing_audio += 1
            continue

        spk = speaker_id_from_path(wav)
        all_speakers.add(spk)

        utt_id = wav.stem
        entries_by_spk[spk].append({
            "utt_id": utt_id,
            "audio": str(wav.resolve()),
            "hanzi": hanzi,
            "text": hanzi,  # 方便 downstream（直接當 text 用）
            "group": group_tag_from_speaker(spk),
        })
        kept += 1

    # ---- 挑 dev 說話人：嘗試在 DF/DM/ZF/ZM 平衡 ----
    spk_by_group = defaultdict(list)
    for spk in sorted(all_speakers):
        spk_by_group[group_tag_from_speaker(spk)].append(spk)

    # 預設希望各組平均；若不足再回補
    desired_per_group = {}
    groups = ["DF", "DM", "ZF", "ZM"]
    base = args.dev_speakers // len(groups)
    rem  = args.dev_speakers % len(groups)
    for g in groups:
        desired_per_group[g] = base
    # 把餘數依序發給前幾組
    for g in groups[:rem]:
        desired_per_group[g] += 1

    dev_speakers = []
    for g in groups:
        cand = spk_by_group.get(g, [])
        random.shuffle(cand)
        take = min(len(cand), desired_per_group[g])
        dev_speakers.extend(cand[:take])

    # 如果還不夠（某組太少），從其他組再補滿
    if len(dev_speakers) < args.dev_speakers:
        remaining = [s for s in sorted(all_speakers) if s not in dev_speakers]
        need = args.dev_speakers - len(dev_speakers)
        random.shuffle(remaining)
        dev_speakers.extend(remaining[:need])

    dev_speakers = sorted(dev_speakers)[:args.dev_speakers]

    # ---- 切分 train/dev 並寫出 JSONL ----
    train_out = args.out_dir / "train.jsonl"
    dev_out   = args.out_dir / "dev.jsonl"

    n_train = n_dev = 0
    with train_out.open("w", encoding="utf-8") as ft, dev_out.open("w", encoding="utf-8") as fd:
        for spk, items in entries_by_spk.items():
            is_dev = spk in dev_speakers
            for ex in items:
                line = json.dumps(ex, ensure_ascii=False)
                if is_dev:
                    fd.write(line + "\n"); n_dev += 1
                else:
                    ft.write(line + "\n"); n_train += 1

    # ---- 輸出統計 ----
    stats = {
        "csv_files": csv_stats["files"],
        "csv_kept_rows": csv_stats["kept"],
        "csv_dropped_empty_text": csv_stats["dropped_empty_text"],
        "kept": kept,
        "dropped_mispronounced": dropped_mis,
        "missing_audio": missing_audio,
        "speakers_total": len(all_speakers),
        "speakers_dev": len(dev_speakers),
        "dev_speakers": dev_speakers,
        "train_utt": n_train,
        "dev_utt": n_dev,
        "keep_asterisk": keep_ast,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
