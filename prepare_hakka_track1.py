#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare manifests for FSR-2025 Track 1 (Hakka Hanzi).

What it does (aligned with Track 2):
- Robust text normalization (NFKC, zero-width removal, optional punctuation stripping).
- CSV duplicate-key detection and coverage auditing.
- Balanced dev speaker selection across DF/DM/ZF/ZM (with XX fallback), reproducible via --seed.
- Portable audio paths via --relative_audio_path (relative to --root).
- Persisted diagnostics: prepare_report.json (+ optional --stats_out) and dev_speakers.txt.
- JSONL schema: {"utt_id","audio","hanzi","text","speaker","group"} (text == normalized hanzi).

Notes:
- Mispronounced filtering can be toggled by --drop_mispronounce (alias: --exclude_mispronounced).
- Asterisk handling can be toggled by --strip_asterisk or --keep_asterisk.
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

# --------- Default paths (overridable via CLI arguments) ---------
DEF_ROOT = Path("HAT-Vol2")
DEF_OUT  = DEF_ROOT / "manifests_track1"

# Allowed CSV column names (in Chinese, must match source files)
COL_FN      = "檔名"
COL_HANZI   = "客語漢字"
COL_REMARKS = "備註"

# Marker considered as "mispronunciation" and filtered out (unless --keep_mispronounce is used)
MISPRONOUNCE_KEY = "正確讀音"

# Zero-width chars and punctuation normalization
_ZW_CHARS_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
_PUNCT_TABLE = str.maketrans({
    "，": "，", "。": "。", "、": "、", "！": "！", "？": "？", "；": "；", "：": "：",
    "（": "（", "）": "）", "「": "「", "」": "」", "『": "『", "』": "』",
    ",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：",
    "(": "（", ")": "）", "[": "（", "]": "）", "{": "（", "}" : "）",
    "—": "－", "–": "－", "-": "－",
})

# --------- Helper functions ---------
def normalize_hanzi(
    text: str,
    strip_spaces: bool = True,
    keep_asterisk: bool = True,
    strip_punct: bool = False,
) -> str:
    """
    Robust text normalization for Track-1 Hanzi:
      - Unicode NFKC normalization (unify full/half width & compatibility forms)
      - Remove zero-width characters
      - Default: remove all whitespaces
      - Default: keep '*' (co-articulation marker), optionally strip
      - Optional: strip punctuation (keep off by default to match evaluation unless aligned)
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZW_CHARS_RE.sub("", t)
    t = t.translate(_PUNCT_TABLE)
    if strip_spaces:
        t = re.sub(r"\s+", "", t)
    if not keep_asterisk:
        t = t.replace("*", "")
    if strip_punct:
        t = re.sub(r"[，。、！？」；：「『』（）－,\.!\?:;\[\]\{\}\(\)\"']", "", t)
    return t


def load_csv_mapping(csv_path: Path):
    """
    Read one *_edit.csv and return:
    - mapping: { wav_filename -> {"hanzi": str, "remarks": str} }
    - kept, dropped_empty_text
    """
    mapping = {}
    kept = dropped_empty = 0

    # Using utf-8-sig automatically skips BOM
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
    return mapping, kept, dropped_empty


def scan_all_csvs(root: Path):
    """
    Find all *_edit.csv under root and merge into one mapping.
    Detect duplicate filename keys across CSVs and keep the first occurrence.
    """
    glob_csvs = sorted(root.rglob("*_edit.csv"))
    all_map = {}
    stats = {"kept": 0, "dropped_empty_text": 0, "files": 0, "duplicates_in_csv": 0}
    for c in glob_csvs:
        m, kept, dropped_empty = load_csv_mapping(c)
        for k, v in m.items():
            if k in all_map:
                stats["duplicates_in_csv"] += 1
                # Keep the first seen; comment the line below to overwrite instead.
                continue
            all_map[k] = v
        stats["kept"] += kept
        stats["dropped_empty_text"] += dropped_empty
        stats["files"] += 1
    return all_map, glob_csvs, stats


def iter_training_wavs(root: Path):
    """
    Iterate through training audio directories, e.g.:
        - 訓練_大埔腔30H/**/<utt>.wav
        - 訓練_詔安腔30H/**/<utt>.wav
    """
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("訓練_")]
    for d in dirs:
        for wav in d.rglob("*.wav"):
            yield wav


def speaker_id_from_path(wav_path: Path):
    """
    Speaker ID = immediate parent directory name, e.g.:
        訓練_大埔腔30H/DF101K2001/DF101K2001_001.wav -> DF101K2001
    """
    if len(wav_path.parts) < 2:
        return "unknown"
    return wav_path.parent.name


def group_tag_from_speaker(spk: str):
    """
    Group tag: use first two chars of speaker ID -> DF/DM/ZF/ZM (else 'XX').
    """
    return spk[:2] if len(spk) >= 2 else "XX"


# --------- Main pipeline ---------
def main():
    ap = argparse.ArgumentParser(description="Prepare Track-1 (Hakka Hanzi) manifests (train/dev) from HAT-Vol2")
    ap.add_argument("--root", type=Path, default=DEF_ROOT, help="HAT-Vol2 root directory")
    ap.add_argument("--out_dir", type=Path, default=DEF_OUT, help="Output directory (will be created)")
    ap.add_argument("--dev_speakers", type=int, default=12, help="Number of dev speakers (default 12)")
    ap.add_argument("--seed", type=int, default=1337)

    # Mispronunciation filtering: mutually exclusive
    misgrp = ap.add_mutually_exclusive_group()
    misgrp.add_argument("--drop_mispronounce", action="store_true",
                        help="Filter samples whose 備註 contain '正確讀音' (recommended)")
    # alias for consistency with Track 2
    misgrp.add_argument("--exclude_mispronounced", dest="drop_mispronounce", action="store_true",
                        help="Alias of --drop_mispronounce (Track 2 naming)")
    misgrp.add_argument("--keep_mispronounce", action="store_true",
                        help="Keep samples whose 備註 contain '正確讀音'")

    # Asterisk handling (co-articulation marker): mutually exclusive
    astgrp = ap.add_mutually_exclusive_group()
    astgrp.add_argument("--keep_asterisk", action="store_true",
                        help="Keep co-articulation '*' (default behavior)")
    astgrp.add_argument("--strip_asterisk", action="store_true",
                        help="Strip co-articulation '*' from text")

    # Normalization / I/O / audit options
    ap.add_argument("--strip_punct", action="store_true",
                    help="Strip punctuation during normalization (default: keep)")
    ap.add_argument("--relative_audio_path", action="store_true",
                    help="Store audio path as relative to --root (default: absolute path)")
    ap.add_argument("--stats_out", type=Path, default=None,
                    help="Optional path to write stats json (e.g., manifests_track1/stats.json)")
    ap.add_argument("--dev_list_out", type=Path, default=None,
                    help="Optional path to write picked dev speakers list (txt)")

    args = ap.parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read all *_edit.csv -> build <filename> -> {hanzi, remarks} mapping
    mapping, csv_list, csv_stats = scan_all_csvs(args.root)
    if not mapping:
        raise SystemExit(f"[ERROR] No *_edit.csv found under {args.root} or all are empty.")

    # Scan training wav files
    entries_by_spk = defaultdict(list)
    kept, missing_audio, dropped_mis, total = 0, 0, 0, 0

    # Coverage / audit sets
    used_csv_keys = set()   # CSV rows that are actually used (by filename)
    seen_audio_fns = set()  # wav basenames seen in training dirs

    # Speaker statistics (for balanced dev selection)
    all_speakers = set()

    # Normalization parameters
    keep_ast = True if args.keep_asterisk else (not args.strip_asterisk)  # default keep; set False if --strip_asterisk
    drop_mispron = args.drop_mispronounce and not args.keep_mispronounce
    strip_punct = bool(args.strip_punct)

    for wav in iter_training_wavs(args.root):
        total += 1
        fn = wav.name  # e.g., DF101K2001_001.wav
        seen_audio_fns.add(fn)
        item = mapping.get(fn)
        if item is None:
            # Audio exists but CSV doesn't have a mapping -> record via audit counters later
            continue

        hanzi = normalize_hanzi(item["hanzi"], strip_spaces=True,
                                keep_asterisk=keep_ast, strip_punct=strip_punct)
        # Filter mispronunciation samples containing the key string
        if drop_mispron and MISPRONOUNCE_KEY in (item.get("remarks") or ""):
            dropped_mis += 1
            continue

        if not wav.exists():
            missing_audio += 1
            continue

        used_csv_keys.add(fn)
        spk = speaker_id_from_path(wav)
        all_speakers.add(spk)

        # Relative or absolute path
        if args.relative_audio_path:
            try:
                audio_path = str(wav.resolve().relative_to(args.root.resolve()))
            except Exception:
                audio_path = wav.name  # conservative fallback
        else:
            audio_path = str(wav.resolve())

        utt_id = wav.stem
        entries_by_spk[spk].append({
            "utt_id": utt_id,
            "audio": audio_path,
            "hanzi": hanzi,
            "text": hanzi,  # downstream convenience; keep 'hanzi' for auditing
            "speaker": spk,
            "group": group_tag_from_speaker(spk),
        })
        kept += 1

    # ---- Select dev speakers: attempt balanced DF/DM/ZF/ZM distribution ----
    spk_by_group = defaultdict(list)
    for spk in sorted(all_speakers):
        spk_by_group[group_tag_from_speaker(spk)].append(spk)

    groups = ["DF", "DM", "ZF", "ZM"]
    base = args.dev_speakers // len(groups)
    rem  = args.dev_speakers % len(groups)
    desired_per_group = {g: base for g in groups}
    for g in groups[:rem]:
        desired_per_group[g] += 1

    dev_speakers = []
    for g in groups:
        cand = spk_by_group.get(g, [])
        random.shuffle(cand)
        take = min(len(cand), desired_per_group[g])
        dev_speakers.extend(cand[:take])

    # If still insufficient (some groups too small), fill from remaining speakers (including 'XX')
    if len(dev_speakers) < args.dev_speakers:
        remaining = [s for s in sorted(all_speakers) if s not in dev_speakers]
        random.shuffle(remaining)
        need = args.dev_speakers - len(dev_speakers)
        dev_speakers.extend(remaining[:need])

    dev_speakers = sorted(dev_speakers)[:args.dev_speakers]

    # Persist selected dev speaker list (optional but recommended)
    if args.dev_list_out:
        args.dev_list_out.parent.mkdir(parents=True, exist_ok=True)
        with args.dev_list_out.open("w", encoding="utf-8") as fdev:
            for s in dev_speakers:
                fdev.write(s + "\n")

    # ---- Split into train/dev and write JSONL ----
    train_out = args.out_dir / "train.jsonl"
    dev_out   = args.out_dir / "dev.jsonl"

    n_train = n_dev = 0
    with train_out.open("w", encoding="utf-8") as ft, dev_out.open("w", encoding="utf-8") as fd:
        for spk, items in entries_by_spk.items():
            is_dev = spk in dev_speakers
            for ex in items:
                line = json.dumps(ex, ensure_ascii=False)
                if is_dev:
                    fd.write(line + "\n")
                    n_dev += 1
                else:
                    ft.write(line + "\n")
                    n_train += 1

    # ---- Coverage & distribution statistics ----
    csv_but_no_audio = len(set(mapping.keys()) - used_csv_keys)
    audio_not_in_csv = len(seen_audio_fns - set(mapping.keys()))

    # Group distribution
    all_groups = ["DF", "DM", "ZF", "ZM", "XX"]
    group_distribution = {g: {"speakers": 0, "utts": 0} for g in all_groups}
    dev_utt_by_group   = {g: 0 for g in all_groups}

    for spk, items in entries_by_spk.items():
        g = group_tag_from_speaker(spk)
        if g not in group_distribution:
            group_distribution[g] = {"speakers": 0, "utts": 0}
            dev_utt_by_group[g] = 0
        group_distribution[g]["speakers"] += 1
        group_distribution[g]["utts"]     += len(items)
        if spk in set(dev_speakers):
            dev_utt_by_group[g] += len(items)

    # ---- Final stats (aligned with Track2 + extended fields) ----
    stats = {
        "csv_files": csv_stats["files"],
        "csv_kept_rows": csv_stats["kept"],
        "csv_dropped_empty_text": csv_stats["dropped_empty_text"],
        "duplicates_in_csv": csv_stats.get("duplicates_in_csv", 0),
        "kept": kept,
        "dropped_mispronounced": dropped_mis,
        "missing_audio": missing_audio,
        "csv_but_no_audio": csv_but_no_audio,
        "audio_not_in_csv": audio_not_in_csv,
        "speakers_total": len(all_speakers),
        "speakers_dev": len(dev_speakers),
        "dev_speakers": dev_speakers,
        "train_utt": n_train,
        "dev_utt": n_dev,
        "keep_asterisk": keep_ast,
        "strip_punct": strip_punct,
        "relative_audio_path": bool(args.relative_audio_path),
        "group_distribution": group_distribution,
        "dev_utt_by_group": dev_utt_by_group,
    }

    # Console print
    # print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ---- Save report JSON (match Track2 behavior) ----
    report_path = args.out_dir / "prepare_report.json"
    with report_path.open("w", encoding="utf-8") as fr:
        json.dump(stats, fr, ensure_ascii=False, indent=2)

    # Optional extra stats path for CI or custom location
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_out.open("w", encoding="utf-8") as fs:
            json.dump(stats, fs, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
