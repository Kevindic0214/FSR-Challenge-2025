#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FSR-2025 Track 2 (Pinyin) Inference - R2

- Batch decoding with dtype/device handling (bf16/fp16 on GPU when possible)
- Load processor from LoRA dir (if available) to match training tokenizer
- Robust pinyin normalization aligned with eval (NFKC, remove zero-width, keep a-z0-9, squeeze spaces)
- Auto-detect eval input: directory of wavs or *.jsonl manifest (optional --root for relative paths)
- Optional key filter to keep rows that appear in official key CSV/TXT
- Output CSV header: 錄音檔檔名,辨認出之客語漢字 (content is pinyin)
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple, Set

import torch
import soundfile as sf
import librosa
from tqdm import tqdm

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000

# -------- Audio IO --------
def load_wav(path: Path):
    wav, sr = sf.read(str(path))
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(-1)
    # ensure float32 for feature extractor
    try:
        import numpy as np  # local import to avoid hard dep at top
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
    except Exception:
        pass
    return wav

def list_wavs(root_dir: Path) -> List[Path]:
    return sorted(Path(root_dir).rglob("*.wav"))

# -------- Pinyin normalization (align with eval) --------
_ZW_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
def normalize_pinyin(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZW_RE.sub("", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = " ".join(t.split())
    return t

# -------- Helpers --------
def _lora_base_name(lora_dir: Path) -> Optional[str]:
    try:
        cfg_path = lora_dir / "adapter_config.json"
        if not cfg_path.exists():
            return None
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        base = (
            data.get("base_model_name_or_path")
            or data.get("base_model_name")
            or data.get("base_model")
        )
        if isinstance(base, str):
            base = base.strip()
        return base or None
    except Exception:
        return None

def load_from_jsonl(jsonl_path: Path, root: Optional[Path]) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    root_p = root.resolve() if root else None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            utt_id = ex.get("utt_id") or ex.get("id")
            ap = Path(ex.get("audio") or ex.get("path", ""))
            if not ap.is_absolute():
                if root_p is None:
                    raise SystemExit("[ERROR] audio path is relative; pass --root to join base directory")
                ap = (root_p / ap).resolve()
            if not ap.exists():
                print(f"[WARN] Missing audio: {ap}", file=sys.stderr)
                continue
            if not utt_id:
                utt_id = ap.name
            elif not str(utt_id).endswith(".wav"):
                utt_id = str(utt_id) + ".wav"
            items.append((str(utt_id), ap))
    if not items:
        raise SystemExit(f"[ERROR] No items loaded from manifest: {jsonl_path}")
    return items

def load_key_set(key_path: str) -> Set[str]:
    p = Path(key_path)
    if not p.exists():
        raise SystemExit(f"[ERROR] key file not found: {key_path}")
    keep: Set[str] = set()
    if p.suffix.lower() in {".txt"}:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                keep.add(Path(line).name)
    else:
        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return keep
        header = rows[0]
        start = 1 if any(h for h in header) else 0
        def idx(names, default):
            for n in names:
                if n in header:
                    return header.index(n)
            return default
        col = idx(["錄音檔檔名","檔名","filename","file","id"], 0)
        for r in rows[start:]:
            if not r:
                continue
            val = (r[col] if col < len(r) else "").strip()
            if not val:
                continue
            keep.add(Path(val).name)
    return keep

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, default=Path("FSR-2025-Hakka-evaluation"),
                    help="Directory of wavs OR path to a *.jsonl manifest (auto-detected)")
    ap.add_argument("--outfile", type=Path, default=Path("Level-Up_拼音.csv"),
                    help="Output CSV path (e.g., '單位_隊名_拼音.csv')")
    ap.add_argument("--model", type=str, default="openai/whisper-large-v2",
                    help="Base Whisper model, e.g., openai/whisper-large-v2 or -v3-turbo")
    ap.add_argument("--lora_dir", type=Path, default=Path("exp_track2_whisper_large_lora"),
                    help="LoRA adapter directory (from training)")
    ap.add_argument("--root", type=Path, default=None,
                    help="If eval_root is a JSONL and audio paths are relative, join with this root")
    # decoding
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1, help="Batch size for decoding")
    # utility
    ap.add_argument("--limit", type=int, default=0, help="If >0, decode first N items only")
    ap.add_argument("--log_jsonl", type=Path, default=None, help="Optional per-utt log jsonl")
    ap.add_argument("--key_csv_filter", type=str, default=None,
                    help="Optional key CSV/TXT to filter output rows by filename")
    args = ap.parse_args()

    # Device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    torch_dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else None)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[INFO] Device: {device}, dtype: {torch_dtype}")

    # Safety: check LoRA base compatibility
    if args.lora_dir is not None and Path(args.lora_dir).exists():
        lora_base = _lora_base_name(Path(args.lora_dir))
        if lora_base:
            a = lora_base.lower(); b = args.model.lower()
            if (a not in b) and (b not in a):
                raise SystemExit(
                    f"[ERROR] LoRA adapter base_model_name_or_path='{lora_base}' does not match --model='{args.model}'."
                )

    # Load model and processor
    print(f"[INFO] Loading base model: {args.model}")
    base = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    # Do NOT force zh; let LoRA guide pinyin output
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    if hasattr(base, "generation_config") and base.generation_config is not None:
        base.generation_config.forced_decoder_ids = None
        base.generation_config.suppress_tokens = []

    if args.lora_dir and Path(args.lora_dir).exists():
        print(f"[INFO] Loading LoRA adapter: {args.lora_dir}")
        model = PeftModel.from_pretrained(base, str(args.lora_dir))
        proc_from = str(args.lora_dir)
    else:
        model = base
        proc_from = args.model

    processor = WhisperProcessor.from_pretrained(proc_from)

    model.to(device).eval()
    model_dtype = next(model.parameters()).dtype

    # Build decode list (dir or jsonl)
    eval_path = Path(args.eval_root)
    if eval_path.suffix.lower() == ".jsonl":
        pair_list = load_from_jsonl(eval_path, args.root)
        print(f"[INFO] Loaded {len(pair_list)} items from manifest: {eval_path}")
    else:
        wavs = list_wavs(eval_path)
        if not wavs:
            print("[ERROR] 找不到任何 WAV，請確認路徑！", file=sys.stderr)
            sys.exit(1)
        pair_list = [(p.name, p) for p in wavs]
        print(f"[INFO] Found {len(pair_list)} wavs under: {eval_path}")

    # optional key filter (keep only files listed in key)
    if args.key_csv_filter:
        keep = load_key_set(args.key_csv_filter)
        before = len(pair_list)
        pair_list = [kv for kv in pair_list if kv[0] in keep]
        print(f"[INFO] Key filter: kept {len(pair_list)}/{before} rows by {args.key_csv_filter}")

    if args.limit and args.limit > 0:
        pair_list = pair_list[:args.limit]
        print(f"[INFO] Limiting to first {len(pair_list)} items")

    # ensure output dir exists
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    # decode (batched)
    log_f = args.log_jsonl.open("w", encoding="utf-8") if args.log_jsonl else None
    n_ok = 0
    bs = max(1, int(args.batch))
    gc = model.generation_config
    with args.outfile.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["錄音檔檔名", "辨認出之客語漢字"])  # content is pinyin
        rng = range(0, len(pair_list), bs)
        for i in tqdm(rng, ncols=100, desc="Decoding"):
            chunk = pair_list[i:i+bs]
            utt_ids = [uid for uid, _ in chunk]
            paths = [ap for _, ap in chunk]
            wavs_np = [load_wav(ap) for ap in paths]
            feats_batch = processor.feature_extractor(
                wavs_np, sampling_rate=SR, return_tensors="pt"
            )
            feats = feats_batch.input_features.to(device=device, dtype=model_dtype)
            attn = None
            try:
                attn = feats_batch.data.get("attention_mask", None)
            except Exception:
                attn = None
            if attn is not None:
                attn = attn.to(device=device)
            with torch.inference_mode():
                pred_ids = model.generate(
                    input_features=feats,
                    attention_mask=attn,
                    num_beams=max(1, int(args.beams)),
                    do_sample=(float(args.temperature) > 0),
                    temperature=float(args.temperature),
                    length_penalty=float(args.length_penalty),
                    max_new_tokens=int(args.max_new_tokens),
                    suppress_tokens=None,
                    no_repeat_ngram_size=3,
                    forced_decoder_ids=None,
                )
            raw_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
            for uid, apath, raw_text in zip(utt_ids, paths, raw_texts):
                text = normalize_pinyin(raw_text)
                w.writerow([uid, text])
                n_ok += 1
                if log_f:
                    log_f.write(json.dumps({
                        "utt_id": uid,
                        "path": str(apath),
                        "raw_text": raw_text,
                        "text": text,
                    }, ensure_ascii=False) + "\n")
    if log_f:
        log_f.close()

    print(f"[INFO] Wrote {n_ok} rows → {args.outfile}")
    print("[HINT] Evaluate with: python eval_track2_ser.py --key_dir FSR-2025-Hakka-evaluation-key --pred_csv", args.outfile)

if __name__ == "__main__":
    main()
