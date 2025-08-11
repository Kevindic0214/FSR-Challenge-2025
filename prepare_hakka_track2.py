#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare HAT-Vol2 (FSR-2025-Hakka-train) for Track 2 (pinyin)
- Recursively read CSVs with headers: 檔名, 中文, 客語漢字, 客語拼音, 標籤, 備註
- Clean "客語拼音":
    * remove starred syllables (e.g., "*ki53")
    * lower-case; keep only [a-z0-9 ] ; squeeze spaces
- Optionally drop rows with 備註 containing "正確讀音"
- Verify audio existence by recursively searching under --audio_root
- Speaker ID: filename stem before '_' (e.g., DF101J2003_001.wav -> DF101J2003)
- Split by speaker: dev = max(--dev_speakers, ceil(#speakers*--dev_ratio))
- Output JSONL: {"id","audio","text"} for train/dev
- Also write simple reports
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from random import Random

STAR_SYL_RE = re.compile(r'\s*\*[a-z]+[0-9]+\s*', flags=re.I)
ALLOW_RE = re.compile(r'[^a-z0-9\s]')

def norm_pinyin(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # remove starred (unpronounced) syllables like "*ki53"
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

def find_audio(audio_root: Path, wavname: str) -> Path | None:
    # common fast path
    direct = audio_root / wavname
    if direct.exists():
        return direct
    # recursive search (slow but robust)
    cands = list(audio_root.rglob(wavname))
    return cands[0] if cands else None

def read_csv_rows(csv_paths: list[Path]):
    for p in csv_paths:
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            # expected headers (Chinese)
            for row in reader:
                yield p, row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=Path, required=True,
                    help='Your HAT-Vol2 root folder (contains CSVs & audio subfolders)')
    ap.add_argument('--audio_root', type=Path, required=False,
                    help='If audio is not under data_root, set this explicitly')
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--dev_speakers', type=int, default=10,
                    help='minimum #speakers to hold out for dev')
    ap.add_argument('--dev_ratio', type=float, default=0.10,
                    help='ratio of speakers for dev split')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--exclude_mispronounced', action='store_true',
                    help='drop rows whose 備註 contains "正確讀音"')
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    audio_root = (args.audio_root or data_root).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect CSV files recursively
    csv_paths = sorted(data_root.rglob('*.csv'))
    if not csv_paths:
        print(f'[ERROR] No CSV found under {data_root}', file=sys.stderr)
        sys.exit(1)
    print(f'[INFO] Found {len(csv_paths)} CSV files.')

    rng = Random(args.seed)

    kept, dropped_empty, dropped_mispron, missing_audio = 0, 0, 0, 0
    items = []
    spk2items = defaultdict(list)

    for csv_path, row in read_csv_rows(csv_paths):
        wav = (row.get('檔名') or '').strip()
        text_raw = (row.get('客語拼音') or '').strip()
        note = (row.get('備註') or '').strip()

        if not wav:
            continue

        if args.exclude_mispronounced and ('正確讀音' in note):
            dropped_mispron += 1
            continue

        text = norm_pinyin(text_raw)
        if not text:
            dropped_empty += 1
            continue

        apath = find_audio(audio_root, wav)
        if apath is None:
            missing_audio += 1
            continue

        uttid = Path(wav).stem
        spk = guess_speaker_id(wav)
        ex = {'id': uttid, 'audio': str(apath), 'text': text, 'speaker': spk}
        items.append(ex)
        spk2items[spk].append(ex)
        kept += 1

    if kept == 0:
        print('[ERROR] No valid samples after cleaning.', file=sys.stderr)
        sys.exit(1)

    speakers = list(spk2items.keys())
    rng.shuffle(speakers)
    min_dev_spk = max(args.dev_speakers, int(len(speakers) * args.dev_ratio + 0.5))
    dev_spks = set(speakers[:min_dev_spk])

    train, dev = [], []
    for spk, exs in spk2items.items():
        (dev if spk in dev_spks else train).extend(exs)

    # sort by id for determinism
    train.sort(key=lambda x: x['id'])
    dev.sort(key=lambda x: x['id'])

    # write jsonl
    def dump_jsonl(path: Path, data: list[dict]):
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for ex in data:
                f.write(json.dumps({'id': ex['id'], 'audio': ex['audio'], 'text': ex['text']}, ensure_ascii=False) + '\n')

    dump_jsonl(out_dir / 'train.jsonl', train)
    dump_jsonl(out_dir / 'dev.jsonl', dev)

    # write a small report
    rep = {
        'kept': kept,
        'dropped_empty_text': dropped_empty,
        'dropped_mispronounced': dropped_mispron,
        'missing_audio': missing_audio,
        'speakers_total': len(speakers),
        'speakers_dev': len(dev_spks),
        'train_utt': len(train),
        'dev_utt': len(dev),
    }
    with (out_dir / 'prepare_report.json').open('w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # quick stats
    def avg_len(exs): 
        return sum(len(e['text'].split()) for e in exs) / max(1, len(exs))
    print('[DONE] Manifest written:')
    print(f'  train.jsonl: {len(train)} utts (avg syllables {avg_len(train):.2f})')
    print(f'  dev.jsonl  : {len(dev)} utts (avg syllables {avg_len(dev):.2f})')
    print('[REPORT]', json.dumps(rep, ensure_ascii=False))

if __name__ == '__main__':
    main()
