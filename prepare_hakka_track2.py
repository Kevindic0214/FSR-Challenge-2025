#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare manifests for FSR-2025 Track 2 (Hakka Pinyin).

What it does (aligned with Track 1):
- Recursively read CSVs with headers: 檔名, 中文, 客語漢字, 客語拼音, 標籤, 備註.
- Clean 客語拼音: remove starred syllables (e.g., "*ki53") by default, lower-case, keep [a-z0-9 ] and squeeze spaces.
- Optionally drop rows whose 備註 contains "正確讀音".
- Verify audio existence by recursively searching under --audio_root.
- Speaker ID: filename stem before '_' (e.g., DF101J2003_001.wav -> DF101J2003), group tag = first 2 chars (DF/DM/ZF/ZM).
- Dev split by speakers: dev = max(--dev_speakers, ceil(#speakers*--dev_ratio)) with balanced allocation across groups.
- JSONL schema: {"utt_id","audio","pinyin","text","speaker","group"} (text == normalized pinyin).
- Persisted diagnostics: prepare_report.json (+ optional --stats_out) and dev_speakers.txt.

Notes:
- Asterisk handling flags mirror Track 1 naming: --strip_asterisk/--keep_asterisk (aliases of drop/keep star syllables).
- Mispronunciation filtering also supports --drop_mispronounce (alias of --exclude_mispronounced).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict
from random import Random
from typing import Optional, List, Dict

DEF_ROOT = Path("HAT-Vol2")
DEF_OUT = DEF_ROOT / "manifests_track2"

STAR_SYL_RE = re.compile(r'\s*\*[a-z]+[0-9]+\s*', flags=re.I)
_ZW_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
ALLOW_RE = re.compile(r'[^a-z0-9\s]')

def norm_pinyin(s: str, drop_star_syllables: bool = True) -> str:
    if s is None:
        return ""
    # Unicode normalize and remove zero-width chars to align with inference
    s = unicodedata.normalize("NFKC", s).strip()
    s = _ZW_RE.sub("", s)
    s = s.lower()
    # optionally remove starred (unpronounced) syllables like "*ki53"
    if drop_star_syllables:
        s = STAR_SYL_RE.sub(' ', s)
    # keep only ascii letters/digits/spaces
    s = ALLOW_RE.sub(' ', s)
    # squeeze spaces
    s = ' '.join(s.split())
    return s

def guess_speaker_id(wavname: str) -> str:
    # DF101J2003_001.wav -> DF101J2003
    stem = Path(wavname).stem
    return stem.split('_')[0]

def group_tag_from_speaker(spk: str) -> str:
    return spk[:2] if isinstance(spk, str) and len(spk) >= 2 else "XX"

def build_audio_index(audio_root: Path) -> Dict[str, Path]:
    """Index all *.wav under audio_root by basename for fast lookup."""
    idx: Dict[str, Path] = {}
    dup = 0
    for p in audio_root.rglob("*.wav"):
        name = p.name
        if name not in idx:
            idx[name] = p
        else:
            dup += 1
    if dup > 0:
        print(f"[WARN] Audio index found {dup} duplicate basenames under {audio_root}. Using first occurrence.")
    return idx

def find_audio(index: Dict[str, Path], audio_root: Path, wavname: str) -> Optional[Path]:
    # fast path: direct join
    direct = audio_root / wavname
    if direct.exists():
        return direct
    # indexed lookup by basename
    return index.get(Path(wavname).name)

def read_csv_rows(csv_paths: List[Path]):
    for p in csv_paths:
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            # expected headers (Chinese)
            for row in reader:
                yield p, row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', '--root', dest='data_root', type=Path, default=DEF_ROOT,
                    help='Your HAT-Vol2 root folder (contains CSVs & audio subfolders)')
    ap.add_argument('--audio_root', type=Path, required=False,
                    help='If audio is not under data_root, set this explicitly')
    ap.add_argument('--out_dir', type=Path, default=DEF_OUT)
    ap.add_argument('--dev_speakers', type=int, default=10,
                    help='minimum #speakers to hold out for dev')
    ap.add_argument('--dev_ratio', type=float, default=0.10,
                    help='ratio of speakers for dev split')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--exclude_mispronounced', action='store_true',
                    help='drop rows whose 備註 contains "正確讀音"')
    # alias to align with Track 1 naming
    ap.add_argument('--drop_mispronounce', dest='exclude_mispronounced', action='store_true',
                    help='Alias of --exclude_mispronounced (Track 1 naming)')
    ap.add_argument('--relative_audio_path', action='store_true',
                    help='store audio path relative to --audio_root (default: absolute)')
    # Starred syllables handling (mutually exclusive)
    ast = ap.add_mutually_exclusive_group()
    ast.add_argument('--drop_star_syllables', action='store_true', default=True,
                     help='drop tokens like *ki53 (default)')
    ast.add_argument('--keep_star_syllables', dest='drop_star_syllables', action='store_false',
                     help='keep tokens prefixed by *')
    # aliases for Track 1 style naming
    ast.add_argument('--strip_asterisk', dest='drop_star_syllables', action='store_true',
                     help='Alias of --drop_star_syllables')
    ast.add_argument('--keep_asterisk', dest='drop_star_syllables', action='store_false',
                     help='Alias of --keep_star_syllables')
    # Diagnostics / outputs
    ap.add_argument('--stats_out', type=Path, default=None,
                    help='optional path to write stats json (e.g., manifests_track2/stats.json)')
    ap.add_argument('--dev_list_out', type=Path, default=None,
                    help='optional path to write selected dev speakers list (txt)')
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    audio_root = (args.audio_root or data_root).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect CSV files recursively
    csv_paths = sorted(data_root.rglob('*_edit.csv'))
    if not csv_paths:
        # fallback to any csv (older dumps). Filter by header later.
        csv_paths = sorted(data_root.rglob('*.csv'))
    if not csv_paths:
        print(f'[ERROR] No CSV found under {data_root}', file=sys.stderr)
        sys.exit(1)
    print(f'[INFO] Found {len(csv_paths)} CSV files.')

    rng = Random(args.seed)

    kept, dropped_empty, dropped_mispron, missing_audio = 0, 0, 0, 0
    items = []
    spk2items = defaultdict(list)
    seen_ids = set()
    dup_ids = 0

    # Build audio index once (performance)
    audio_index = build_audio_index(audio_root)

    csv_wav_names = set()

    for csv_path, row in read_csv_rows(csv_paths):
        wav = (row.get('檔名') or '').strip()
        text_raw = (row.get('客語拼音') or '').strip()
        note = (row.get('備註') or '').strip()

        if not wav:
            continue
        csv_wav_names.add(Path(wav).name)

        if args.exclude_mispronounced and ('正確讀音' in note):
            dropped_mispron += 1
            continue

        text = norm_pinyin(text_raw, drop_star_syllables=bool(args.drop_star_syllables))
        if not text:
            dropped_empty += 1
            continue

        apath = find_audio(audio_index, audio_root, wav)
        if apath is None:
            missing_audio += 1
            continue

        uttid = Path(wav).stem
        if uttid in seen_ids:
            # avoid duplicates across CSVs
            dup_ids += 1
            continue
        seen_ids.add(uttid)
        spk = guess_speaker_id(wav)
        grp = group_tag_from_speaker(spk)
        # relative or absolute audio path
        audio_path_str = str(apath)
        if args.relative_audio_path:
            try:
                audio_path_str = str(apath.resolve().relative_to(audio_root.resolve()))
            except Exception:
                audio_path_str = apath.name
        ex = {
            'utt_id': uttid,
            'audio': audio_path_str,
            'pinyin': text,
            'text': text,
            'speaker': spk,
            'group': grp,
        }
        items.append(ex)
        spk2items[spk].append(ex)
        kept += 1

    if kept == 0:
        print('[ERROR] No valid samples after cleaning.', file=sys.stderr)
        sys.exit(1)

    speakers = list(spk2items.keys())
    rng.shuffle(speakers)
    # Balanced dev selection across groups (DF/DM/ZF/ZM; fallback 'XX')
    target_dev = max(args.dev_speakers, int(len(speakers) * args.dev_ratio + 0.5))
    grp2speakers: Dict[str, List[str]] = defaultdict(list)
    for spk in speakers:
        grp2speakers[group_tag_from_speaker(spk)].append(spk)
    # shuffle within group for determinism
    for g in grp2speakers.keys():
        rng.shuffle(grp2speakers[g])
    # proportional allocation with floor, then fill remainder by largest remainder
    totals = {g: len(v) for g, v in grp2speakers.items()}
    total_spk = sum(totals.values()) or 1
    alloc = {g: int(target_dev * (totals[g] / total_spk)) for g in totals}
    # Ensure at least 1 if group exists and target_dev >= number of groups
    if target_dev >= len(totals):
        for g in totals:
            if totals[g] > 0:
                alloc[g] = max(1, alloc[g])
    # adjust to meet target_dev
    assigned = sum(alloc.values())
    # compute remainders for largest remainder method
    rema = sorted(((target_dev * (totals[g] / total_spk)) - alloc[g], g) for g in totals)
    # add or remove to match target
    if assigned < target_dev:
        need = target_dev - assigned
        for _ in range(need):
            if not rema:
                break
            _, g = rema.pop()  # largest remainder
            alloc[g] = min(totals[g], alloc[g] + 1)
    elif assigned > target_dev:
        over = assigned - target_dev
        for _ in range(over):
            # remove from smallest remainder first
            if not rema:
                break
            _, g = rema.pop(0)
            alloc[g] = max(0, alloc[g] - 1)
    # pick dev speakers based on allocation
    dev_spks = set()
    for g, n_take in alloc.items():
        if n_take <= 0:
            continue
        pool = grp2speakers.get(g, [])
        dev_spks.update(pool[:n_take])

    train, dev = [], []
    for spk, exs in spk2items.items():
        (dev if spk in dev_spks else train).extend(exs)

    # sort by id for determinism
    train.sort(key=lambda x: x['utt_id'])
    dev.sort(key=lambda x: x['utt_id'])

    # write jsonl
    def dump_jsonl(path: Path, data: List[dict]):
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for ex in data:
                line = {
                    'utt_id': ex['utt_id'],
                    'audio': ex['audio'],
                    'pinyin': ex.get('pinyin', ex['text']),
                    'text': ex['text'],
                    'speaker': ex.get('speaker', ''),
                    'group': ex.get('group', 'XX')
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

    dump_jsonl(out_dir / 'train.jsonl', train)
    dump_jsonl(out_dir / 'dev.jsonl', dev)

    # group-wise stats
    speakers_total_by_grp = {g: len(v) for g, v in grp2speakers.items()}
    speakers_dev_by_grp = {g: len([s for s in grp2speakers.get(g, []) if s in dev_spks]) for g in grp2speakers.keys()}
    def count_utts_by_group(lst):
        out: Dict[str, int] = defaultdict(int)
        for ex in lst:
            out[ex.get('group', 'XX')] += 1
        return dict(out)
    train_utt_by_grp = count_utts_by_group(train)
    dev_utt_by_grp = count_utts_by_group(dev)
    all_utt_by_grp = count_utts_by_group(items)

    # build group_distribution aligned with Track 1
    all_groups = set(list(speakers_total_by_grp.keys()) + list(all_utt_by_grp.keys()) + ['DF','DM','ZF','ZM','XX'])
    group_distribution = {
        g: {
            'speakers': int(speakers_total_by_grp.get(g, 0)),
            'utts': int(all_utt_by_grp.get(g, 0))
        }
        for g in sorted(all_groups)
    }

    # write a small report
    rep = {
        'kept': kept,
        'dropped_empty_text': dropped_empty,
        'dropped_mispronounced': dropped_mispron,
        'missing_audio': missing_audio,
        'duplicate_ids': dup_ids,
        'duplicates_in_csv': dup_ids,
        'csv_files': len(csv_paths),
        'csv_but_no_audio': missing_audio,
        'audio_not_in_csv': int(max(0, len(set(audio_index.keys()) - csv_wav_names))),
        'speakers_total': len(speakers),
        'speakers_dev': len(dev_spks),
        'dev_target': target_dev,
        'dev_speakers': sorted(list(dev_spks)),
        'relative_audio_path': bool(args.relative_audio_path),
        'speakers_total_by_group': speakers_total_by_grp,
        'speakers_dev_by_group': speakers_dev_by_grp,
        'train_utt_by_group': train_utt_by_grp,
        'dev_utt_by_group': dev_utt_by_grp,
        'group_distribution': group_distribution,
        'train_utt': len(train),
        'dev_utt': len(dev),
    }
    with (out_dir / 'prepare_report.json').open('w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # Optional extra outputs
    if args.stats_out is not None:
        try:
            args.stats_out.parent.mkdir(parents=True, exist_ok=True)
            with args.stats_out.open('w', encoding='utf-8') as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    if args.dev_list_out is not None:
        try:
            with args.dev_list_out.open('w', encoding='utf-8') as f:
                for spk in sorted(dev_spks):
                    f.write(f"{spk}\n")
        except Exception:
            pass

    # quick stats
    def avg_len(exs): 
        return sum(len(e['text'].split()) for e in exs) / max(1, len(exs))
    print('[DONE] Manifest written:')
    print(f'  train.jsonl: {len(train)} utts (avg syllables {avg_len(train):.2f})')
    print(f'  dev.jsonl  : {len(dev)} utts (avg syllables {avg_len(dev):.2f})')
    print('[REPORT]', json.dumps(rep, ensure_ascii=False))

if __name__ == '__main__':
    main()
