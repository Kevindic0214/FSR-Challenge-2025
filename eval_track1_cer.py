#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FSR-2025 Track 1 (Hanzi) Evaluation - R2.1

Modes:
  A) Manifest mode: --ref <manifest.jsonl> + --hyp <pred.csv>
  B) Key mode:      --key_csv <key.csv> or --key_dir <dir> + --hyp <pred.csv>

Normalization (aligned with prepare/infer):
  - Unicode NFKC -> remove zero-width -> remove spaces
  - Default KEEP '*' (use --strip_asterisk to remove)
  - Default keep punctuation (use --strip_punct to remove)

New (diagnostic):
  - --probe_variants : report CER under {AS-IS, unify-to-simplified, unify-to-traditional}
  - --convert_hyp {none,s2t,t2s} : convert hypothesis only (for diagnosis)
  - Character profile of HYP (Han/Latin/Digit/Other ratio)

Outputs:
  - CER, Exact-Match
  - Optional --dump_err JSONL, --aligned_out CSV
"""

import argparse
import csv
import json
import sys
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# ---------- Optional OpenCC ----------
_CC = None
def _try_init_opencc():
    global _CC
    if _CC is not None:
        return
    try:
        from opencc import OpenCC
        _CC = {
            "s2t": OpenCC("s2t"),
            "t2s": OpenCC("t2s"),
        }
    except Exception:
        _CC = {}

def conv_zh(s: str, mode: Optional[str]) -> str:
    """mode in {None,'s2t','t2s'}; returns s if converter missing."""
    if not s or not mode:
        return s
    _try_init_opencc()
    cc = _CC.get(mode)
    if cc is None:
        return s
    try:
        return cc.convert(s)
    except Exception:
        return s

# ---------- Normalization (align with prepare/infer) ----------
_ZW_CHARS_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
_PUNCT_TABLE = str.maketrans({
    "，":"，","。":"。","、":"、","！":"！","？":"！","；":"；","：":"：",
    "（":"（","）":"）","「":"「","」":"」","『":"『","』":"』",
    ",":"，",".":"。","!":"！","?":"？",";":"；",":":"：",
    "(":"（",")":"）","[":"（","]":"）","{":"（","}":"）",
    "—":"－","–":"－","-":"－",
})
def normalize_hanzi(text: str, strip_spaces=True, keep_asterisk=True, strip_punct=False) -> str:
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

# ---------- Levenshtein ----------
def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    curr = [0]*(lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i-1]
        for j in range(1, lb + 1):
            cb = b[j-1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    return prev[lb]

def first_diff_index(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1

# ---------- Loaders ----------
def load_hyp_csv(path: Path) -> Dict[str, str]:
    hyp: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows:
        return hyp
    header = rows[0]
    start = 1 if any(h for h in header) else 0
    def idx(names, default):
        for n in names:
            if n in header:
                return header.index(n)
        return default
    c_fn = idx(["錄音檔檔名","檔名","filename","file","id"], 0)
    c_tx = idx(["辨認結果","結果","hyp","prediction","text"], 1)
    for r in rows[start:]:
        if not r: continue
        fn = (r[c_fn] or "").strip()
        if not fn: continue
        tx = ",".join(r[c_tx:]).strip() if c_tx < len(r) else ""
        hyp[Path(fn).name] = tx
    return hyp

def load_ref_from_manifest(jsonl_path: Path) -> Dict[str, str]:
    ref: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            utt_id = (ex.get("utt_id") or "").strip()
            txt = (ex.get("text") or ex.get("hanzi") or "").strip()
            if not utt_id:
                audio = (ex.get("audio") or "").strip()
                if audio:
                    utt_id = Path(audio).stem
                else:
                    continue
            key = utt_id if utt_id.endswith(".wav") else (utt_id + ".wav")
            ref[key] = txt
    return ref

def _read_csv_rows(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        return list(rdr)

def load_ref_from_key_csv(csv_path: Path) -> Dict[str, str]:
    rows = _read_csv_rows(csv_path)
    ref: Dict[str, str] = {}
    if not rows:
        return ref
    header = rows[0]
    start = 1 if any(h for h in header) else 0
    def idx(names, default):
        for n in names:
            if n in header:
                return header.index(n)
        return default
    c_fn = idx(["錄音檔檔名","檔名","filename","file","id"], 0)
    c_tx = idx(["正解","客語漢字","標註","text"], 1)
    for r in rows[start:]:
        if not r: continue
        ref[Path((r[c_fn] or "").strip()).name] = (r[c_tx] or "").strip()
    return ref

def load_ref_from_key_dir(key_dir: Path) -> Dict[str, str]:
    csvs = sorted(key_dir.glob("*.csv"))
    if len(csvs) == 1:
        return load_ref_from_key_csv(csvs[0])
    refs: Dict[str, str] = {}
    dup, used = 0, 0
    for p in sorted(key_dir.glob("*_edit.csv")):
        rows = _read_csv_rows(p)
        if not rows: continue
        header = rows[0]
        if ("檔名" not in header) or ("客語漢字" not in header):
            continue
        used += 1
        c_fn, c_tx = header.index("檔名"), header.index("客語漢字")
        for r in rows[1:]:
            if not r: continue
            key = Path((r[c_fn] or "").strip()).name
            txt = (r[c_tx] or "").strip()
            if not key: continue
            if key in refs: dup += 1; continue
            refs[key] = txt
    if not refs:
        raise SystemExit(f"[ERROR] No refs found in {key_dir}")
    print(f"[INFO] Loaded {len(refs)} refs from {used} files in {key_dir} (duplicates ignored: {dup})")
    return refs

# ---------- Diagnostics ----------
_HAN_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
_LAT_RE = re.compile(r"[A-Za-z]")
_DIG_RE = re.compile(r"[0-9]")
def profile_text(s: str) -> Tuple[int,int,int,int]:
    han = len(_HAN_RE.findall(s))
    lat = len(_LAT_RE.findall(s))
    dig = len(_DIG_RE.findall(s))
    oth = max(0, len(s) - han - lat - dig)
    return han, lat, dig, oth

def profile_hyp(hyp: Dict[str,str], sample: int = 50):
    keys = list(hyp.keys())[:sample]
    han=lat=dig=oth=0
    for k in keys:
        h = hyp[k]
        a,b,c,d = profile_text(h)
        han+=a; lat+=b; dig+=c; oth+=d
    tot = max(1, han+lat+dig+oth)
    print(f"[DIAG] HYP profile (first {len(keys)} utts): Han={han/tot:.2%}, Latin={lat/tot:.2%}, Digit={dig/tot:.2%}, Other={oth/tot:.2%}")

# ---------- Evaluation ----------
def evaluate(ref: Dict[str,str], hyp: Dict[str,str],
             keep_asterisk: bool, strip_punct: bool,
             dump_err: Optional[Path]=None, aligned_out: Optional[Path]=None):
    total_ref_chars = total_edits = exact = 0
    aligned_rows = []
    err_f = dump_err.open("w", encoding="utf-8") if dump_err else None

    for uid in sorted(ref.keys()):
        r_raw = ref[uid]
        h_raw = hyp.get(uid, "")
        r = normalize_hanzi(r_raw, strip_spaces=True, keep_asterisk=keep_asterisk, strip_punct=strip_punct)
        h = normalize_hanzi(h_raw, strip_spaces=True, keep_asterisk=keep_asterisk, strip_punct=strip_punct)
        edits = levenshtein(r, h)
        total_ref_chars += len(r)
        total_edits += edits
        exact += int(r == h)
        if aligned_out:
            cer_utt = (edits/len(r)) if len(r) > 0 else (0.0 if len(h)==0 else 1.0)
            aligned_rows.append([uid, r, h, len(r), edits, f"{cer_utt:.6f}"])
        if err_f and r != h:
            err_f.write(json.dumps({
                "utt_id": uid, "ref_raw": r_raw, "hyp_raw": h_raw,
                "ref": r, "hyp": h, "ref_len": len(r), "edits": edits,
                "first_diff": first_diff_index(r, h)
            }, ensure_ascii=False) + "\n")

    if err_f: err_f.close()
    cer = (total_edits / total_ref_chars) if total_ref_chars > 0 else 0.0
    em  = (exact / len(ref)) if ref else 0.0
    if aligned_out:
        aligned_out.parent.mkdir(parents=True, exist_ok=True)
        with aligned_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["utt_id","ref","hyp","ref_len","edits","utt_cer"])
            aligned_rows.sort(key=lambda x: x[0]); w.writerows(aligned_rows)
    return cer, em

def print_summary(tag: str, cer: float, em: float):
    print(f"[{tag}] CER = {cer:.4f} ({cer*100:.2f}%), EM = {em*100:.2f}%")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate Track1 CER (Hanzi) with variant probing")
    # Mode A
    ap.add_argument("--ref", type=str, help="Manifest JSONL (fields: utt_id, text/hanzi)")
    # Mode B
    ap.add_argument("--key_csv", type=str, help="Official key CSV file")
    ap.add_argument("--key_dir", type=str, help="Directory containing key CSV or *_edit.csv files")
    # Predictions
    ap.add_argument("--hyp", "--pred_csv", dest="pred_csv", required=True, help="Prediction CSV (錄音檔檔名,辨認結果)")
    # Normalization toggles
    ap.add_argument("--strip_asterisk", action="store_true", help="Strip '*' (default: keep)")
    ap.add_argument("--strip_punct", action="store_true", help="Strip punctuation (default: keep)")
    # Diagnostics
    ap.add_argument("--probe_variants", action="store_true", help="Also report CER after unifying to Simplified/Traditional (requires opencc)")
    ap.add_argument("--convert_hyp", choices=["none","s2t","t2s"], default="none", help="Convert hypothesis only (diagnostic)")
    # Outputs
    ap.add_argument("--dump_err", type=str, default=None, help="Per-utterance mismatch JSONL")
    ap.add_argument("--aligned_out", type=str, default=None, help="Aligned CSV (ref/hyp/cer per utt)")
    args = ap.parse_args()

    # Load reference
    ref_map: Dict[str,str] = {}
    if args.ref:
        ref_map = load_ref_from_manifest(Path(args.ref))
    elif args.key_csv:
        ref_map = load_ref_from_key_csv(Path(args.key_csv))
    elif args.key_dir:
        ref_map = load_ref_from_key_dir(Path(args.key_dir))
    else:
        print("[ERROR] Provide one of: --ref | --key_csv | --key_dir", file=sys.stderr); sys.exit(1)
    if not ref_map: print("[ERROR] No reference loaded.", file=sys.stderr); sys.exit(1)

    # Load hypothesis
    hyp_map = load_hyp_csv(Path(args.pred_csv))
    if not hyp_map: print("[ERROR] No predictions loaded.", file=sys.stderr); sys.exit(1)

    # Character profile (raw hyp)
    profile_hyp(hyp_map, sample=min(50, len(hyp_map)))

    # Optional convert only hyp (diagnostic)
    conv_mode = None if args.convert_hyp == "none" else args.convert_hyp
    if conv_mode:
        _try_init_opencc()
        if not _CC:
            print("[WARN] opencc not installed; skipping --convert_hyp. Try: pip install opencc-python-reimplemented")
        else:
            hyp_map = {k: conv_zh(v, conv_mode) for k,v in hyp_map.items()}
            print(f"[INFO] Applied hyp conversion: {conv_mode}")

    keep_ast = not args.strip_asterisk
    dump_err = Path(args.dump_err) if args.dump_err else None
    aligned_out = Path(args.aligned_out) if args.aligned_out else None

    # AS-IS
    cer, em = evaluate(ref_map, hyp_map, keep_asterisk=keep_ast, strip_punct=args.strip_punct,
                       dump_err=dump_err, aligned_out=aligned_out)
    print_summary("AS-IS", cer, em)

    # Variant probe (unify both sides)
    if args.probe_variants:
        _try_init_opencc()
        if not _CC:
            print("[WARN] opencc not installed; skipping --probe_variants. Try: pip install opencc-python-reimplemented")
        else:
            ref_s = {k: conv_zh(v, "t2s") for k,v in ref_map.items()}
            hyp_s = {k: conv_zh(v, "t2s") for k,v in hyp_map.items()}
            cer_s, em_s = evaluate(ref_s, hyp_s, keep_asterisk=keep_ast, strip_punct=args.strip_punct)
            print_summary("UNIFY→SIMP", cer_s, em_s)

            ref_t = {k: conv_zh(v, "s2t") for k,v in ref_map.items()}
            hyp_t = {k: conv_zh(v, "s2t") for k,v in hyp_map.items()}
            cer_t, em_t = evaluate(ref_t, hyp_t, keep_asterisk=keep_ast, strip_punct=args.strip_punct)
            print_summary("UNIFY→TRAD", cer_t, em_t)

if __name__ == "__main__":
    main()
