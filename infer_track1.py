#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FSR-2025 Track 1 (Hanzi) Inference - R2

- Auto-detect input mode:
  * If --eval_root points to a directory: recursively decode *.wav
  * If --eval_root points to a *.jsonl: read items (utt_id, audio) from manifest
- Force Chinese transcription via decoder prompt (avoid language drift)
- Robust text normalization aligned with prepare/eval (NFKC, remove zero-width, remove spaces; optional '*' / punct)
- Optional key filter to keep rows that appear in an official key CSV/TXT
- Output CSV header: 錄音檔檔名,辨認結果
"""

import argparse, csv, glob, json, os, sys, unicodedata, re
from pathlib import Path
from typing import List, Tuple, Optional, Set

import torch, torchaudio
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000

# -------- Text normalization (align with prepare/eval) --------
_ZW_CHARS_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
_PUNCT_TABLE = str.maketrans({
    "，":"，","。":"。","、":"、","！":"！","？":"？","；":"；","：":"：",
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

# -------- I/O helpers --------
def list_wavs(root_dir: str) -> List[str]:
    # Use pathlib for better cross-platform support
    root_path = Path(root_dir)
    wavs = sorted([str(p) for p in root_path.rglob("*.wav")])
    if not wavs:
        print(f"[ERROR] No wav files found under directory: {root_dir}", file=sys.stderr)
        sys.exit(1)
    return wavs

def load_from_jsonl(jsonl_path: str, root: Optional[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    root_p = Path(root).resolve() if root else None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            utt_id = ex.get("utt_id")
            apath = Path(ex["audio"])
            if not apath.is_absolute():
                if root_p is None:
                    raise SystemExit("[ERROR] audio path is relative; pass --root to join base directory")
                apath = (root_p / apath).resolve()
            if not apath.exists():
                print(f"[WARN] Missing audio: {apath}", file=sys.stderr)
                continue
            # Align with key schema: filename includes .wav
            if not utt_id:
                utt_id = apath.name
            elif not utt_id.endswith(".wav"):
                utt_id = utt_id + ".wav"
            items.append((utt_id, str(apath)))
    if not items:
        raise SystemExit(f"[ERROR] No items loaded from manifest: {jsonl_path}")
    return items

def load_audio(path: str, target_sr: int = SR):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)

def load_key_set(key_path: str) -> Set[str]:
    """
    Accept CSV/TXT. If CSV has header including '錄音檔檔名', use that column.
    Else use first column. For TXT, one filename per line.
    """
    p = Path(key_path)
    if not p.exists():
        raise SystemExit(f"[ERROR] key file not found: {key_path}")
    exts = {".csv", ".txt", ".tsv"}
    if p.suffix.lower() not in exts:
        raise SystemExit(f"[ERROR] unsupported key file type: {p.suffix}")
    keep: Set[str] = set()
    if p.suffix.lower() == ".txt":
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                keep.add(line)
    else:
        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return keep
        header = rows[0]
        # assume first row is header if any cell is non-empty; otherwise start from 0
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
    ap.add_argument("--eval_root", required=True,
                    help="Directory of wavs OR path to a *.jsonl manifest (auto-detected)")
    ap.add_argument("--outfile", required=True, help="Output CSV path")
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo",
                    help="Base model: e.g., openai/whisper-large-v3-turbo or openai/whisper-large-v2")
    ap.add_argument("--lora_dir", default=None,
                    help="Optional LoRA adapter directory (loaded on top of --model)")
    ap.add_argument("--root", type=str, default=None,
                    help="If eval_root is a JSONL and audio paths are relative, join with this root")
    # decoding
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    # postprocess
    ap.add_argument("--strip_asterisk", action="store_true", help="Remove '*' (default: keep)")
    ap.add_argument("--strip_punct", action="store_true", help="Strip punctuation (default: keep)")
    # utility
    ap.add_argument("--limit", type=int, default=0, help="If >0, decode first N items only")
    ap.add_argument("--log_jsonl", type=str, default=None, help="Optional per-utt log jsonl")
    ap.add_argument("--key_csv_filter", type=str, default=None,
                    help="Optional key CSV/TXT to filter output rows by filename")
    ap.add_argument("--batch_size_decode", type=int, default=4,
                    help="Batch size for decoding/generation (increase to improve GPU utilization)")
    args = ap.parse_args()

    # Safety check: ensure LoRA adapter matches the chosen base model
    def _lora_base_name(lora_dir: str) -> Optional[str]:
        try:
            cfg_path = Path(lora_dir) / "adapter_config.json"
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

    if args.lora_dir:
        lora_base = _lora_base_name(args.lora_dir)
        if lora_base:
            a = lora_base.lower()
            b = args.model.lower()
            if (a not in b) and (b not in a):
                raise SystemExit(
                    f"[ERROR] LoRA adapter base_model_name_or_path='{lora_base}' does not match --model='{args.model}'."
                    " Please use a matching base model, or change --model accordingly."
                )
        else:
            print(
                f"[WARN] Cannot determine adapter base model in {args.lora_dir} (missing/invalid adapter_config.json).",
                "Proceeding without strict check...",
            )

    # device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    torch_dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else None)
    print(f"[INFO] Device: {device}, dtype: {torch_dtype}")

    # model & processor
    print(f"[INFO] Loading base model: {args.model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if args.lora_dir:
        print(f"[INFO] Loading LoRA adapter: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
    proc_from = args.lora_dir if args.lora_dir else args.model
    processor = WhisperProcessor.from_pretrained(proc_from)

    model.to(device).eval()
    model_dtype = next(model.parameters()).dtype

    # Force Chinese transcription prompt (avoid language drift)
    forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_ids
    gc = model.generation_config
    gc.num_beams = args.beams
    gc.do_sample = args.temperature > 0
    gc.temperature = args.temperature
    gc.length_penalty = args.length_penalty
    gc.max_new_tokens = args.max_new_tokens

    # build decode list (auto-detect)
    eval_path = Path(args.eval_root)
    if eval_path.suffix.lower() == ".jsonl":
        pair_list = load_from_jsonl(str(eval_path), args.root)
        print(f"[INFO] Loaded {len(pair_list)} items from manifest: {eval_path}")
    else:
        wavs = list_wavs(str(eval_path))
        pair_list = [(Path(p).name, p) for p in wavs]
        print(f"[INFO] Found {len(pair_list)} wavs under: {eval_path}")

    # optional key filter (for official warm-up) – apply before limit
    if args.key_csv_filter:
        keep = load_key_set(args.key_csv_filter)
        before = len(pair_list)
        pair_list = [kv for kv in pair_list if kv[0] in keep]
        print(f"[INFO] Key filter: kept {len(pair_list)}/{before} rows by {args.key_csv_filter}")

    if args.limit and args.limit > 0:
        pair_list = pair_list[:args.limit]
        print(f"[INFO] Limiting to first {len(pair_list)} items")

    # ensure output dir exists
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    # decode (batched)
    log_f = open(args.log_jsonl, "w", encoding="utf-8") if args.log_jsonl else None
    n_ok = 0
    bs = max(1, int(args.batch_size_decode))
    with open(args.outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["錄音檔檔名", "辨認結果"])
        rng = range(0, len(pair_list), bs)
        for i in tqdm(rng, ncols=100, desc="Decoding"):
            chunk = pair_list[i:i+bs]
            utt_ids = [uid for uid, _ in chunk]
            paths = [ap for _, ap in chunk]
            wavs = [load_audio(p, target_sr=SR) for p in paths]
            feats_batch = processor.feature_extractor(
                [w.numpy() for w in wavs], sampling_rate=SR, return_tensors="pt"
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
                    num_beams=gc.num_beams,
                    do_sample=gc.do_sample,
                    temperature=gc.temperature,
                    length_penalty=gc.length_penalty,
                    max_new_tokens=gc.max_new_tokens,
                    no_repeat_ngram_size=3,
                    forced_decoder_ids=forced_ids,
                )
            raw_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
            for uid, apath, raw_text in zip(utt_ids, paths, raw_texts):
                text = normalize_hanzi(
                    raw_text,
                    strip_spaces=True,
                    keep_asterisk=(not args.strip_asterisk),
                    strip_punct=args.strip_punct,
                )
                w.writerow([uid, text])
                n_ok += 1
                if log_f:
                    log_f.write(json.dumps({
                        "utt_id": uid,
                        "path": apath,
                        "raw_text": raw_text,
                        "text": text,
                    }, ensure_ascii=False) + "\n")
    if log_f:
        log_f.close()

    print(f"[INFO] Wrote {n_ok} rows → {args.outfile}")
    print("[HINT] Next: evaluate on dev (ref=manifest) or warm-up key.")
    print("  dev:   python eval_track1_cer.py --ref HAT-Vol2/manifests_track1/dev.jsonl --hyp", args.outfile)
    print("  warmup:python eval_track1_cer.py --key_dir FSR-2025-Hakka-evaluation-key --pred_csv", args.outfile)

if __name__ == "__main__":
    main()
