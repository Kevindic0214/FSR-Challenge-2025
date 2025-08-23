#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, re, glob, argparse
from collections import OrderedDict

# ---------------- helpers ----------------
def tokenize_pinyin(raw: str, drop_star_tokens: bool = True):
    """
    Tokenize Hakka pinyin into syllables.
    - drop tokens starting with '*' (coalesced syllable marker in the key)
    - lowercase
    - keep only [a-z0-9] inside each token
    """
    if not raw:
        return []
    pieces = re.split(r"\s+", raw.strip())
    kept = []
    for p in pieces:
        if not p:
            continue
        if drop_star_tokens and p.startswith("*"):
            continue
        p = p.lower()
        p = re.sub(r"[^a-z0-9]", "", p)
        if p:
            kept.append(p)
    return kept

def edit_distance_tokens(a, b) -> int:
    # classic DP without backtrace (cost=1 for I/D/S)
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1,       # deletion
                        dp[j - 1] + 1,   # insertion
                        prev + cost)     # substitution/match
            prev = tmp
    return dp[m]

def read_key_dir(key_dir: str) -> OrderedDict:
    """
    Read all *_edit.csv under key_dir and return OrderedDict[id -> raw pinyin].
    Requires columns: 「檔名」, 「客語拼音」.
    """
    paths = sorted(glob.glob(os.path.join(key_dir, "*_edit.csv")))
    if not paths:
        raise SystemExit(f"No *_edit.csv found in {key_dir}")
    od = OrderedDict()
    for p in paths:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            if not rdr.fieldnames or ("檔名" not in rdr.fieldnames or "客語拼音" not in rdr.fieldnames):
                raise SystemExit(f"Expected headers 「檔名」 and 「客語拼音」 in {p}")
            for row in rdr:
                fn = (row["檔名"] or "").strip()
                uid = os.path.splitext(os.path.basename(fn))[0]
                py = (row["客語拼音"] or "").strip()
                if uid and uid not in od:
                    od[uid] = py
    return od

def read_pred_csv(pred_csv: str) -> dict:
    """
    Read your Level-Up_拼音.csv: first column filename, second column hypothesis.
    """
    mp = {}
    with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    # skip header heuristically
    start = 0
    if rows:
        head = "".join(rows[0]).lower()
        if any(k in head for k in ["檔名","錄音","file","filename","id","sent"]):
            start = 1
    for r in rows[start:]:
        if not r: continue
        uid = os.path.splitext(os.path.basename(r[0].strip()))[0]
        hyp = ",".join(r[1:]).strip() if len(r) > 1 else ""
        mp[uid] = hyp
    return mp

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Track2 SER (syllable-level WER) from Level-Up_拼音.csv")
    ap.add_argument("--key_dir",  required=True, help="FSR-2025-Hakka-evaluation-key directory (contains *_edit.csv)")
    ap.add_argument("--pred_csv", required=True, help="Level-Up_拼音.csv (filename,hypothesis)")
    ap.add_argument("--drop_star_tokens", action="store_true", default=True,
                    help="Drop tokens starting with '*' from reference (recommended)")
    ap.add_argument("--keep_star_tokens", dest="drop_star_tokens",
                action="store_false",
                help="Keep tokens starting with '*' in reference (not recommended)")
    ap.add_argument("--aligned_out", default="aligned_pinyin.csv", help="Per-utterance details CSV")
    args = ap.parse_args()

    ref_raw = read_key_dir(args.key_dir)
    hyp_raw = read_pred_csv(args.pred_csv)

    total_ref_tokens = 0
    total_edits = 0
    exact = 0
    n = 0

    missing = [uid for uid in ref_raw.keys() if uid not in hyp_raw]
    extra   = [uid for uid in hyp_raw.keys() if uid not in ref_raw]

    with open(args.aligned_out, "w", encoding="utf-8", newline="") as fa:
        w = csv.writer(fa)
        w.writerow(["id","ref","hyp","ref_len","utt_SER"])  # utt_SER = per-utt syllable WER
        for uid, ref_txt in ref_raw.items():
            hyp_txt = hyp_raw.get(uid, "")
            ref_tok = tokenize_pinyin(ref_txt, drop_star_tokens=args.drop_star_tokens)
            hyp_tok = tokenize_pinyin(hyp_txt, drop_star_tokens=False)  # hyp usually has no '*'
            edits = edit_distance_tokens(ref_tok, hyp_tok)
            rl = len(ref_tok)
            total_edits += edits
            total_ref_tokens += rl
            exact += 1 if ref_tok == hyp_tok else 0
            n += 1
            utt_ser = (edits / rl) if rl > 0 else (0.0 if len(hyp_tok) == 0 else 1.0)
            w.writerow([uid, " ".join(ref_tok), " ".join(hyp_tok), rl, f"{utt_ser:.4f}"])

    ser = (total_edits / total_ref_tokens) if total_ref_tokens > 0 else 0.0
    exact_rate = (exact / n) if n > 0 else 0.0

    print("=== Track2 evaluation (SER = syllable-level WER) ===")
    print(f"UTT={n}  REF_TOKENS={total_ref_tokens}  TOTAL_EDITS={total_edits}")
    print(f"SER: {ser:.4f}")
    print(f"Sentence exact-match rate: {exact_rate:.4f}")
    if missing:
        print(f"[WARN] Missing predictions: {len(missing)} (counted as deletions).")
    if extra:
        print(f"[INFO] Extra predictions (ignored): {len(extra)}")
    print(f"Per-utterance details -> {args.aligned_out}")

if __name__ == "__main__":
    main()
