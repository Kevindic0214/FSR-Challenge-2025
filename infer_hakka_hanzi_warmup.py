#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Warm-up / General inference for Track 1 (Hanzi)

Features:
- Works for zero-shot (e.g., openai/whisper-large-v3-turbo) or LoRA on v2.
- Forces Chinese transcription prompt to avoid language drift.
- Uses robust text normalization aligned with prepare/eval.
- Outputs CSV with header: 錄音檔檔名,辨認結果
"""

import argparse
import os
import glob
import csv
import sys
import json
import torch
import torchaudio
import unicodedata
from tqdm import tqdm
from typing import Optional
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000

# ---------- Text normalization (align with prepare/eval) ----------
import re
_ZW_CHARS_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
_PUNCT_TABLE = str.maketrans({
    "，": "，", "。": "。", "、": "、", "！": "！", "？": "？", "；": "；", "：": "：",
    "（": "（", "）": "）", "「": "「", "」": "」", "『": "『", "』": "』",
    ",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：",
    "(": "（", ")": "）", "[": "（", "]": "）", "{": "（", "}": "）",
    "—": "－", "–": "－", "-": "－",
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

# ---------- I/O ----------
def list_wavs(eval_root: str):
    wavs = sorted(glob.glob(os.path.join(eval_root, "**", "*.wav"), recursive=True))
    if not wavs:
        print(f"[ERROR] No wav files found under: {eval_root}", file=sys.stderr)
        sys.exit(1)
    return wavs

def load_audio(path: str, target_sr: int = SR):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True, help="Directory containing wavs (recursively scanned)")
    ap.add_argument("--outfile", required=True, help="Output CSV path")
    # Model / LoRA
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo",
                    help="Base model, e.g., openai/whisper-large-v3-turbo or openai/whisper-large-v2")
    ap.add_argument("--lora_dir", default=None,
                    help="Optional LoRA adapter directory; if set, will be loaded on top of --model")
    # Decoding
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    # Postprocess
    ap.add_argument("--strip_asterisk", action="store_true", help="Remove '*' before writing CSV (default keep)")
    ap.add_argument("--strip_punct", action="store_true", help="Strip punctuation (default keep)")
    # Misc
    ap.add_argument("--limit", type=int, default=0, help="If >0, only decode first N wavs (for sanity check)")
    ap.add_argument("--log_jsonl", type=str, default=None, help="Optional per-utt log jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] Device: {device}, dtype: {torch_dtype}")

    print(f"[INFO] Loading base model: {args.model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if args.lora_dir:
        print(f"[INFO] Loading LoRA adapter from: {args.lora_dir}")
        model = PeftModel.from_pretrained(model, args.lora_dir)
    model.to(device).eval()

    # Processor: prefer LoRA dir if given (to avoid tokenizer drift), else base model
    proc_from = args.lora_dir if args.lora_dir else args.model
    processor = WhisperProcessor.from_pretrained(proc_from)

    # ----- Force Chinese transcription prompt (avoid language drift) -----
    # Use decoder prompt ids from processor to set language/task explicitly.
    forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_ids
    # Keep default suppress tokens; do NOT zero them out to avoid weird tokens
    # model.generation_config.suppress_tokens = None  # leave defaults
    gen_cfg = model.generation_config
    gen_cfg.num_beams = args.beams
    gen_cfg.do_sample = (args.temperature is not None and args.temperature > 0)
    gen_cfg.temperature = args.temperature
    gen_cfg.length_penalty = args.length_penalty
    gen_cfg.max_new_tokens = args.max_new_tokens

    wav_paths = list_wavs(args.eval_root)
    if args.limit and args.limit > 0:
        wav_paths = wav_paths[:args.limit]
    print(f"[INFO] Found {len(wav_paths)} wavs under {args.eval_root}")

    writer = None
    log_f = open(args.log_jsonl, "w", encoding="utf-8") if args.log_jsonl else None
    n_ok = 0

    with open(args.outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["錄音檔檔名", "辨認結果"])

        for wp in tqdm(wav_paths, ncols=100, desc="Decoding"):
            utt_id = os.path.basename(wp)  # include .wav (matches key schema)
            audio = load_audio(wp, target_sr=SR)

            inputs = processor.feature_extractor(audio.numpy(), sampling_rate=SR, return_tensors="pt")
            feats = inputs.input_features.to(device)

            with torch.no_grad():
                pred_ids = model.generate(input_features=feats)

            raw_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            text = normalize_hanzi(
                raw_text,
                strip_spaces=True,
                keep_asterisk=(not args.strip_asterisk),
                strip_punct=args.strip_punct,
            )
            w.writerow([utt_id, text])
            n_ok += 1

            if log_f:
                log_f.write(json.dumps({
                    "utt_id": utt_id,
                    "path": wp,
                    "raw_text": raw_text,
                    "text": text,
                }, ensure_ascii=False) + "\n")

    if log_f:
        log_f.close()

    print(f"[INFO] Wrote {n_ok} rows to {args.outfile}")
    print("[HINT] Next:")
    print(f"  python make_keyonly_track2.py --key_dir FSR-2025-Hakka-evaluation-key "
          f"--pred_csv {args.outfile} --out Level-Up_hanzi_keyonly.csv")
    print("  python eval_track1_cer.py --key_dir FSR-2025-Hakka-evaluation-key "
          " --pred_csv Level-Up_hanzi_keyonly.csv --aligned_out aligned_hanzi.csv")

if __name__ == "__main__":
    main()
