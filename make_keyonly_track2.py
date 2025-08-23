#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, glob, argparse
from collections import OrderedDict

def read_key_ids(key_dir: str):
    """Collect IDs (basename without extension) from all *_edit.csv under key_dir, preserving order."""
    ids = []
    seen = set()
    paths = sorted(glob.glob(os.path.join(key_dir, "*_edit.csv")))
    if not paths:
        raise SystemExit(f"No *_edit.csv found under: {key_dir}")
    for p in paths:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            if not rdr.fieldnames or ("檔名" not in rdr.fieldnames):
                raise SystemExit(f"Expected header '檔名' in {p}")
            for row in rdr:
                fn = (row["檔名"] or "").strip()
                uid = os.path.splitext(os.path.basename(fn))[0]
                if uid and uid not in seen:
                    seen.add(uid)
                    ids.append(uid)
    return ids

def read_pred(pred_csv: str):
    """
    Read your prediction CSV: first column filename, second column hypothesis.
    Return dict uid -> (original_filename_with_ext, hypothesis).
    """
    mp = OrderedDict()
    with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    # optional header skip
    start = 0
    if rows:
        head = "".join(rows[0]).lower()
        if any(k in head for k in ["檔名","錄音","file","filename","id","sent"]):
            start = 1
    for r in rows[start:]:
        if not r: 
            continue
        fname = r[0].strip()
        hyp = ",".join(r[1:]).strip() if len(r) > 1 else ""
        uid = os.path.splitext(os.path.basename(fname))[0]
        mp[uid] = (fname, hyp)
    return mp

def main():
    ap = argparse.ArgumentParser(description="Filter predictions to only keys present in evaluation key.")
    ap.add_argument("--key_dir", required=True, help="FSR-2025-Hakka-evaluation-key directory")
    ap.add_argument("--pred_csv", required=True, help="Level-Up_拼音.csv")
    ap.add_argument("--out", default="Level-Up_pinyin_keyonly.csv", help="Filtered output CSV")
    args = ap.parse_args()

    key_ids = read_key_ids(args.key_dir)
    pred_map = read_pred(args.pred_csv)

    kept, missing, extra = 0, 0, 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for uid in key_ids:  # write in key order
            if uid in pred_map:
                fname, hyp = pred_map[uid]
                w.writerow([fname, hyp])
                kept += 1
            else:
                missing += 1
    # count extras that were dropped
    extra = sum(1 for uid in pred_map.keys() if uid not in set(key_ids))

    print(f"[OK] Wrote {args.out}")
    print(f"kept={kept}, missing_from_pred={missing}, dropped_extras={extra}")

if __name__ == "__main__":
    main()
