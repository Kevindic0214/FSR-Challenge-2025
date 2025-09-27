#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, re, glob, argparse, unicodedata
from collections import OrderedDict, defaultdict
from pathlib import Path
import math, contextlib, wave

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
# Normalization config aligned with Track 2 prepare/infer
UMLAUTS = {
    "ü": "v", "ǖ": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v",
    "Ü": "v", "Ǖ": "v", "Ǘ": "v", "Ǚ": "v", "Ǜ": "v",
}
STAR_SYL_RE = re.compile(r"\s*(\*[a-z]+[0-9]+|[a-z]+[0-9]+\*)\s*", flags=re.I)

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
    raw = unicodedata.normalize("NFKC", raw)
    raw = ZERO_WIDTH.sub("", raw)
    # unify ü-variants and u:/U: to v BEFORE lowercasing
    tmp = []
    for ch in raw:
        tmp.append("v" if ch in UMLAUTS else ch)
    raw = "".join(tmp)
    raw = raw.replace("u:", "v").replace("U:", "v")
    raw = raw.lower()
    # drop starred syllables like *ki53 or ki53* from reference if requested
    if drop_star_tokens:
        raw = STAR_SYL_RE.sub(" ", raw)
    pieces = re.split(r"\s+", raw.strip())
    kept = []
    for p in pieces:
        if not p:
            continue
        # tokens are already lowercased; keep ascii letters/digits only
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

def first_diff_index_tokens(a, b) -> int:
    """Return first index where token lists differ; -1 if identical length/content.
    If one is a prefix of the other, returns the shorter length.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1

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

def read_key_csv(csv_path: str) -> OrderedDict:
    """
    Read a single key CSV and return OrderedDict[id -> raw pinyin].
    Requires columns: 「檔名」, 「客語拼音」.
    """
    p = csv_path
    od = OrderedDict()
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
    if not od:
        raise SystemExit(f"[ERROR] No refs found in {p}")
    return od

def read_pred_csv(pred_csv: str) -> dict:
    """
    Read prediction CSV: first column is filename, second column is hypothesis.
    - Header row allowed (will be skipped heuristically)
    - Compatible with Track 1 format: "錄音檔檔名,辨認結果"
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
    ap = argparse.ArgumentParser(description="Evaluate Track2 SER (syllable-level WER) with optional bucket/group analysis")
    ap.add_argument("--key_csv",  required=False, help="Official key CSV file (with 檔名,客語拼音)")
    ap.add_argument("--key_dir",  required=False, help="FSR-2025-Hakka-evaluation-key directory (contains *_edit.csv)")
    ap.add_argument("--hyp", "--pred_csv", dest="pred_csv", required=True,
                    help="Prediction CSV (錄音檔檔名,辨認結果)")
    ap.add_argument("--drop_star_tokens", action="store_true", default=True,
                    help="Drop tokens starting with '*' from reference (recommended)")
    ap.add_argument("--keep_star_tokens", dest="drop_star_tokens",
                action="store_false",
                help="Keep tokens starting with '*' in reference (not recommended)")
    ap.add_argument("--aligned_out", default=None, help="Per-utterance details CSV")
    ap.add_argument("--dump_err", default=None, help="Per-utterance mismatch JSONL (like Track 1)")
    # Optional bucket/group analysis
    ap.add_argument("--manifest_meta", default=None,
                    help="Optional manifest JSONL with fields {utt_id,audio,group} for bucket/group analysis")
    ap.add_argument("--audio_root", default=None,
                    help="If manifest 'audio' is relative, join with this root for duration buckets")
    ap.add_argument("--bucket_unit", choices=["tokens","seconds"], default="tokens",
                    help="Bucket by reference syllable length (tokens) or audio duration (seconds)")
    ap.add_argument("--tok_buckets", default="0,10,20,40,80",
                    help="Comma-separated token edges (last open-ended) when --bucket_unit=tokens")
    ap.add_argument("--sec_buckets", default="0,4.8,12.4,20,60",
                    help="Comma-separated second edges when --bucket_unit=seconds")
    ap.add_argument("--dump_bucket_csv", default=None, help="Write per-bucket stats CSV")
    ap.add_argument("--dump_group_csv", default=None, help="Write per-group stats CSV")
    args = ap.parse_args()

    # Load references (prefer key_csv over key_dir if both given)
    if args.key_csv:
        ref_raw = read_key_csv(args.key_csv)
    elif args.key_dir:
        ref_raw = read_key_dir(args.key_dir)
    else:
        raise SystemExit("Provide one of --key_csv or --key_dir")

    hyp_raw = read_pred_csv(args.pred_csv)

    total_ref_tokens = 0
    total_edits = 0
    exact = 0
    n = 0

    missing = [uid for uid in ref_raw.keys() if uid not in hyp_raw]
    extra   = [uid for uid in hyp_raw.keys() if uid not in ref_raw]

    aligned_rows = []
    per_utt = []  # records for bucket/group summary
    err_f = open(args.dump_err, "w", encoding="utf-8") if args.dump_err else None
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
        aligned_rows.append([uid, " ".join(ref_tok), " ".join(hyp_tok), rl, edits, f"{utt_ser:.6f}"])
        per_utt.append({
            "utt_id": uid,
            "ref_len": rl,
            "edits": edits,
            "utt_ser": utt_ser,
        })
        if err_f and ref_tok != hyp_tok:
            err = {
                "utt_id": uid,
                "ref_raw": ref_txt,
                "hyp_raw": hyp_txt,
                "ref_tok": ref_tok,
                "hyp_tok": hyp_tok,
                "ref_len": rl,
                "edits": edits,
                "first_diff_tok": first_diff_index_tokens(ref_tok, hyp_tok),
                "utt_ser": utt_ser,
            }
            import json as _json
            err_f.write(_json.dumps(err, ensure_ascii=False) + "\n")

    if args.aligned_out:
        with open(args.aligned_out, "w", encoding="utf-8", newline="") as fa:
            w = csv.writer(fa)
            # Align header style with Track 1 (field names and order)
            w.writerow(["utt_id","ref","hyp","ref_len","edits","utt_ser"])  # per-utt SER
            # keep original ref order
            for row in aligned_rows:
                w.writerow(row)
    if err_f:
        err_f.close()

    ser = (total_edits / total_ref_tokens) if total_ref_tokens > 0 else 0.0
    exact_rate = (exact / n) if n > 0 else 0.0

    matched = len(ref_raw) - len(missing)
    print(f"[INFO] Coverage: matched {matched}/{len(ref_raw)} refs; missing={len(missing)}; extra={len(extra)} (pred-only)")
    print(f"[AS-IS] SER = {ser:.4f} ({ser*100:.2f}%), EM = {exact_rate*100:.2f}%")
    if args.aligned_out:
        print(f"Aligned output -> {args.aligned_out}")
    if args.dump_err:
        print(f"Errors JSONL -> {args.dump_err}")
    # ---- Optional: bucket/group analysis ----
    def _parse_edges(s: str):
        try:
            arr = [float(x) for x in s.split(',') if x.strip()]
        except Exception:
            raise SystemExit(f"[ERROR] Could not parse bucket edges: {s}")
        arr = sorted(arr)
        if math.isfinite(arr[-1]):
            arr.append(float('inf'))
        return arr

    def _bucket_index(v: float, edges):
        for i in range(len(edges)-1):
            if edges[i] <= v < edges[i+1]:
                return i
        return len(edges)-2

    def _fmt_range(i: int, edges, unit: str):
        a, b = edges[i], edges[i+1]
        if not math.isfinite(b):
            return (f"{int(a)}+" if unit=="tokens" else f"{a:g}+s")
        if unit == 'seconds':
            return f"{a:g}–{b:g}s"
        else:
            return f"{int(a)}–{int(b-1)}"

    def _wav_duration_seconds(path: Path):
        try:
            with contextlib.closing(wave.open(str(path), 'rb')) as wf:
                fr = wf.getframerate(); n = wf.getnframes()
                if fr > 0:
                    return n / float(fr)
        except Exception:
            return None
        return None

    meta = {}
    if args.manifest_meta and os.path.exists(args.manifest_meta):
        meta = {}
        with open(args.manifest_meta, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                import json as _json
                ex = _json.loads(line)
                uid = (ex.get('utt_id') or '').strip()
                uid = uid if uid.endswith('.wav') else (uid + '.wav')
                meta[os.path.splitext(uid)[0]] = {
                    'group': (ex.get('group') or '').strip() or 'UNK',
                    'audio': ex.get('audio') or '',
                }
    # join meta
    if meta:
        for r in per_utt:
            m = meta.get(r['utt_id'], {})
            r['group'] = m.get('group', 'UNK')
            r['audio'] = m.get('audio', '')

    # compute seconds if needed
    if meta and args.bucket_unit == 'seconds':
        aroot = Path(args.audio_root) if args.audio_root else None
        for r in per_utt:
            a = r.get('audio')
            if not a:
                r['duration_sec'] = None
                continue
            p = Path(a)
            if not p.is_absolute() and aroot is not None:
                p = aroot / p
            r['duration_sec'] = _wav_duration_seconds(p)

    if meta and per_utt and (args.dump_bucket_csv or args.dump_group_csv):
        unit = args.bucket_unit
        edges = _parse_edges(args.sec_buckets if unit == 'seconds' else args.tok_buckets)
        # aggregate
        def agg(rows):
            ref_sum = sum(x['ref_len'] for x in rows)
            edit_sum = sum(x['edits'] for x in rows)
            exact_n = sum(1 for x in rows if x['edits'] == 0)
            ser_v = (edit_sum / ref_sum) if ref_sum > 0 else 0.0
            em_v = (exact_n / len(rows)) if rows else 0.0
            return ser_v, em_v, len(rows)

        # buckets
        bucket_rows = defaultdict(list)
        for x in per_utt:
            v = (x.get('duration_sec') if unit == 'seconds' else float(x['ref_len']))
            if unit == 'seconds' and (v is None):
                continue
            idx = _bucket_index(v, edges)
            bucket_rows[idx].append(x)

        bucket_stats = []  # (label, SER, EM, N)
        for i in range(len(edges)-1):
            lab = _fmt_range(i, edges, unit)
            ser_v, em_v, n_v = agg(bucket_rows.get(i, []))
            bucket_stats.append((lab, ser_v, em_v, n_v))

        # groups
        group_rows = defaultdict(list)
        for x in per_utt:
            g = x.get('group', 'UNK')
            group_rows[g].append(x)
        group_stats = []
        for g in sorted(group_rows.keys()):
            ser_v, em_v, n_v = agg(group_rows[g])
            group_stats.append((g, ser_v, em_v, n_v))

        if args.dump_bucket_csv:
            Path(args.dump_bucket_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.dump_bucket_csv, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f); w.writerow(['bucket','ser','em','n'])
                for lab, ser_v, em_v, n_v in bucket_stats:
                    w.writerow([lab, f"{ser_v:.6f}", f"{em_v:.6f}", n_v])
            print(f"[BUCKET] -> {args.dump_bucket_csv}")
        if args.dump_group_csv:
            Path(args.dump_group_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.dump_group_csv, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f); w.writerow(['group','ser','em','n'])
                for g, ser_v, em_v, n_v in group_stats:
                    w.writerow([g, f"{ser_v:.6f}", f"{em_v:.6f}", n_v])
            print(f"[GROUP]  -> {args.dump_group_csv}")

if __name__ == "__main__":
    main()
