#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track1 (Hanzi) CER evaluator for FSR-2025 Hakka ASR warm-up.
- Aggregates ALL *_edit.csv under --key_dir
- Uses the '客語漢字' column as reference
- Aligns by basename with extension (e.g., 'xxx.wav'); also tolerates stem-only IDs from predictions
- Normalization: NFKC -> remove zero-width -> remove all whitespace -> (default) remove '*'
- Missing predictions are treated as empty string (counted as deletions)
- Outputs micro-averaged CER plus optional per-utterance details
"""

import argparse
import csv
import glob
import os
import re
import sys
import unicodedata
from collections import OrderedDict

# ---------- Normalization helpers ----------

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")

def is_punct(ch: str) -> bool:
    """Unicode punctuation check."""
    cat = unicodedata.category(ch)
    if cat and cat.startswith("P"):
        return True
    # Some common symbols which are not strictly 'P*' but often treated as punct
    return ch in "·•…—–‐-·•()[]{}<>《》〈〉「」『』“”\"'、，。！？；：．"

def norm_hanzi(s: str, keep_star: bool = False, strip_punct: bool = False) -> str:
    """NFKC -> remove zero-width -> remove whitespace -> optionally strip '*' and punctuation."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH_RE.sub("", s)
    # remove ALL whitespace
    s = "".join(s.split())
    if not keep_star:
        s = s.replace("*", "")
    if strip_punct:
        s = "".join(ch for ch in s if not is_punct(ch))
    return s

# ---------- Distance ----------

def levenshtein_char(a: str, b: str) -> int:
    """Memory-optimized Levenshtein distance at char level."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,     # deletion
                           cur[j - 1] + 1,  # insertion
                           prev[j - 1] + cost))  # substitution
        prev = cur
    return prev[-1]

# ---------- IO ----------

def ingest_key_dir(key_dir: str, debug_headers: bool = False) -> OrderedDict:
    """
    Aggregate ALL *_edit.csv under key_dir and return OrderedDict[id_with_ext -> hanzi_ref].
    Requirements per file: columns '檔名' and '客語漢字'.
    If duplicates occur across files, the first occurrence is kept.
    """
    paths = sorted(glob.glob(os.path.join(key_dir, "*_edit.csv")))
    if not paths:
        print(f"[ERROR] No *_edit.csv found in {key_dir}", file=sys.stderr)
        sys.exit(1)

    refs = OrderedDict()
    per_file_counts = []
    dup_ids = 0
    used_files = 0

    for p in paths:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            fields = rdr.fieldnames or []
            if debug_headers:
                print(f"[DEBUG] {os.path.basename(p)} fields: {fields}")

            if ("檔名" not in fields) or ("客語漢字" not in fields):
                # Skip files that do not contain Hanzi reference column
                continue

            used_files += 1
            count = 0
            for row in rdr:
                fn = (row.get("檔名") or "").strip()
                uid_ext = os.path.basename(fn)  # keep .wav to match submission
                hz = (row.get("客語漢字") or "").strip()
                if not uid_ext:
                    continue
                if uid_ext in refs:
                    dup_ids += 1
                    continue
                refs[uid_ext] = hz
                count += 1
            per_file_counts.append((os.path.basename(p), count))

    if not refs:
        print(f"[ERROR] No Hanzi refs found under {key_dir} (no '客語漢字' column?).", file=sys.stderr)
        sys.exit(1)

    summary = ", ".join([f"{name}:{cnt}" for name, cnt in per_file_counts if cnt > 0]) or "none"
    print(f"[INFO] Loaded {len(refs)} Hanzi refs from {used_files} files: {summary}")
    if dup_ids > 0:
        print(f"[INFO] Ignored {dup_ids} duplicate IDs (first occurrence kept).")
    return refs

def read_pred_csv(pred_csv: str) -> tuple[dict, set]:
    """
    Read predictions CSV with two columns: (filename, hypothesis)
    - Supports optional header (e.g., '錄音檔檔名,辨認結果')
    - Returns:
        preds_map: dict mapping BOTH 'basename.ext' AND 'stem' to the hypothesis
        raw_ids_with_ext: set of the actual IDs (basename.ext) found in the CSV (for 'extra' reporting)
    """
    preds_map = {}
    raw_ids_with_ext = set()

    with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)

    start = 0
    if rows:
        head = "".join(rows[0]).lower()
        if any(k in head for k in ["錄音", "檔名", "file", "filename", "id", "sent"]):
            start = 1

    for r in rows[start:]:
        if not r:
            continue
        first = (r[0] or "").strip()
        uid_ext = os.path.basename(first)   # prefer with extension (.wav)
        uid_stem = os.path.splitext(uid_ext)[0]
        hyp = ",".join(r[1:]).strip() if len(r) > 1 else ""
        if uid_ext:
            raw_ids_with_ext.add(uid_ext)
            preds_map[uid_ext] = hyp
            # also allow stem key for robustness
            if uid_stem not in preds_map:
                preds_map[uid_stem] = hyp

    return preds_map, raw_ids_with_ext

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate Track1 CER (Hanzi, char-level micro average).")
    ap.add_argument("--key_dir", required=True, help="Directory containing *_edit.csv with columns: 檔名, 客語漢字")
    ap.add_argument("--pred_csv", required=True, help="Your prediction CSV (filename,hypothesis)")
    ap.add_argument("--keep_star", action="store_true", help="Keep '*' during scoring (default: remove)")
    ap.add_argument("--strip_punct", action="store_true", help="Remove punctuation during scoring")
    ap.add_argument("--aligned_out", default=None, help="Optional per-utterance CSV path")
    ap.add_argument("--debug_headers", action="store_true", help="Print field names for each key CSV")
    args = ap.parse_args()

    refs = ingest_key_dir(args.key_dir, debug_headers=args.debug_headers)  # uid_ext -> hanzi
    preds, raw_pred_ids_ext = read_pred_csv(args.pred_csv)                 # map (uid_ext+stem) -> hyp, and raw ext IDs

    total_ref_chars = 0
    total_edits = 0
    exact = 0
    aligned_rows = []

    # Missing / extra reporting
    missing_ids = [uid for uid in refs.keys() if uid not in raw_pred_ids_ext and os.path.splitext(uid)[0] not in preds]
    extra_ids = [uid for uid in raw_pred_ids_ext if uid not in refs]

    for uid_ext, ref_raw in refs.items():
        ref = norm_hanzi(ref_raw, keep_star=args.keep_star, strip_punct=args.strip_punct)
        # Try matching by ext first, then by stem
        hyp_raw = preds.get(uid_ext, preds.get(os.path.splitext(uid_ext)[0], ""))
        hyp = norm_hanzi(hyp_raw, keep_star=args.keep_star, strip_punct=args.strip_punct)

        edits = levenshtein_char(ref, hyp)
        total_ref_chars += len(ref)
        total_edits += edits
        if ref == hyp:
            exact += 1

        if args.aligned_out:
            rlen = len(ref)
            cer_utt = (edits / rlen) if rlen > 0 else (0.0 if len(hyp) == 0 else 1.0)
            aligned_rows.append([uid_ext, ref, hyp, rlen, edits, f"{cer_utt:.6f}"])

    cer = (total_edits / total_ref_chars) if total_ref_chars > 0 else 0.0
    em_rate = exact / len(refs) if refs else 0.0

    print("==== Track1 Hanzi CER ====")
    print(f"UTT            = {len(refs)}")
    print(f"REF_CHARS      = {total_ref_chars}")
    print(f"TOTAL_EDITS    = {total_edits}")
    print(f"CER            = {cer:.4f} ({cer*100:.2f}%)")
    print(f"Exact-Match    = {em_rate*100:.2f}%")

    if missing_ids:
        print(f"[WARN] Missing predictions: {len(missing_ids)} (counted as deletions).")
    if extra_ids:
        print(f"[INFO] Extra predictions (ignored): {len(extra_ids)})")

    if args.aligned_out:
        # stable sort by uid for deterministic output
        aligned_rows.sort(key=lambda x: x[0])
        with open(args.aligned_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "ref", "hyp", "ref_len", "edits", "utt_cer"])
            w.writerows(aligned_rows)
        print(f"[INFO] Wrote aligned details to {args.aligned_out}")

if __name__ == "__main__":
    main()
