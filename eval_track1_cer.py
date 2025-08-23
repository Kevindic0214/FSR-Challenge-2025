#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os, sys, glob, re
from collections import OrderedDict

def read_two_col_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for r in reader:
            if not r:
                continue
            rows.append(r)
    # drop header if first row looks like header
    if rows and (("錄音" in rows[0][0]) or ("file" in rows[0][0].lower())):
        rows = rows[1:]
    out = []
    for r in rows:
        if len(r) < 2:
            r = (r[0], "")
        out.append((r[0].strip(), r[1].strip()))
    return out

def ingest_key_dir(key_dir):
    cand = []
    for ext in ("*.csv", "*.tsv"):
        cand.extend(glob.glob(os.path.join(key_dir, ext)))
    if not cand:
        print(f"[ERROR] No key csv/tsv found in {key_dir}", file=sys.stderr)
        sys.exit(1)
    key_path = sorted(cand)[0]
    rows = read_two_col_csv(key_path)
    mapping = OrderedDict((k, v) for k, v in rows)
    print(f"[INFO] Loaded {len(mapping)} refs from {key_path}")
    return mapping

def norm_hanzi(s: str, keep_star: bool = False) -> str:
    s = re.sub(r"\s+", "", s)
    if not keep_star:
        s = s.replace("*", "")
    return s

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost))
        prev = cur
    return prev[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key_dir", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--keep_star", action="store_true")
    ap.add_argument("--aligned_out", default=None)
    args = ap.parse_args()

    refs = ingest_key_dir(args.key_dir)           # id -> ref
    preds_rows = read_two_col_csv(args.pred_csv)  # [(id, hyp), ...]
    preds = OrderedDict(preds_rows)

    missing = [k for k in refs if k not in preds]
    extra   = [k for k in preds if k not in refs]
    if missing:
        print(f"[WARN] Missing {len(missing)} ids not in predictions.", file=sys.stderr)
    if extra:
        print(f"[WARN] Found {len(extra)} preds not in key set (ignored).", file=sys.stderr)

    total_ref, total_edits, exact = 0, 0, 0
    aligned = []

    for utt, ref_raw in refs.items():
        ref = norm_hanzi(ref_raw, keep_star=args.keep_star)
        hyp = norm_hanzi(preds.get(utt, ""), keep_star=args.keep_star)
        e = levenshtein(ref, hyp)
        total_ref += len(ref)
        total_edits += e
        if ref == hyp:
            exact += 1
        if args.aligned_out:
            aligned.append([utt, ref, hyp, e, len(ref), f"{(e/len(ref)) if len(ref) else 0:.6f}"])

    cer = (total_edits / total_ref) if total_ref else 0.0
    em  = exact / len(refs) if refs else 0.0

    print("==== Track1 Hanzi CER ====")
    print(f"UTT            = {len(refs)}")
    print(f"REF_CHARS      = {total_ref}")
    print(f"TOTAL_EDITS    = {total_edits}")
    print(f"CER            = {cer:.4f} ({cer*100:.2f}%)")
    print(f"Exact-Match    = {em*100:.2f}%")

    if args.aligned_out:
        with open(args.aligned_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id","ref","hyp","edits","ref_len","cer"])
            w.writerows(aligned)
        print(f"[INFO] Wrote aligned details to {args.aligned_out}")

if __name__ == "__main__":
    main()
