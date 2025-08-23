#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Warm-up inference for Track 1 (Hanzi)
- Decode all wavs under --eval_root
- Load Whisper-large-v2 + LoRA adapter from --lora_dir
- Output CSV with header: 錄音檔檔名,辨認結果
- Postprocess: remove spaces; optionally remove '*' via --strip_asterisk
"""

import argparse
import os
import glob
import csv
import sys
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000

def list_wavs(eval_root: str):
    # Recursively collect all wavs; keep a stable order
    wavs = sorted(glob.glob(os.path.join(eval_root, "**", "*.wav"), recursive=True))
    if not wavs:
        print(f"[ERROR] No wav files found under: {eval_root}", file=sys.stderr)
        sys.exit(1)
    return wavs

def load_audio(path: str, target_sr: int = SR):
    wav, sr = torchaudio.load(path)
    # mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True, help="Warm-up evaluation root dir")
    ap.add_argument("--lora_dir", required=True, help="LoRA adapter directory (with processor/tokenizer)")
    ap.add_argument("--outfile", required=True, help="Output CSV path")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=225)
    ap.add_argument("--strip_asterisk", action="store_true", help="Remove '*' before writing CSV")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    base = "openai/whisper-large-v2"
    print(f"[INFO] Loading base model: {base}")
    model = WhisperForConditionalGeneration.from_pretrained(base)
    print(f"[INFO] Loading LoRA adapter from: {args.lora_dir}")
    model = PeftModel.from_pretrained(model, args.lora_dir)

    # Gen config
    gc = model.generation_config
    gc.num_beams = args.beams
    gc.do_sample = False
    gc.length_penalty = args.length_penalty
    gc.max_new_tokens = args.max_new_tokens
    # Disable forced tokens / translation
    gc.forced_decoder_ids = None
    gc.suppress_tokens = []
    try:
        gc.task = "transcribe"
        gc.language = None
    except Exception:
        pass

    # Inference-friendly settings
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.eval().to(device)

    # Use processor saved with the adapter to avoid tokenizer drift
    processor = WhisperProcessor.from_pretrained(args.lora_dir)

    wav_paths = list_wavs(args.eval_root)
    print(f"[INFO] Found {len(wav_paths)} wavs under {args.eval_root}")

    results = []
    for wp in tqdm(wav_paths, ncols=100, desc="Decoding"):
        utt_id = os.path.basename(wp)
        audio = load_audio(wp, target_sr=SR)

        inputs = processor.feature_extractor(audio.numpy(), sampling_rate=SR, return_tensors="pt")
        feats = inputs.input_features.to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                input_features=feats,
                num_beams=gc.num_beams,
                do_sample=gc.do_sample,
                length_penalty=gc.length_penalty,
                max_new_tokens=gc.max_new_tokens,
                forced_decoder_ids=None,
                suppress_tokens=[],
            )

        text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        # Hanzi postprocess: remove spaces; optionally remove '*'
        text = "".join(text.split())
        if args.strip_asterisk:
            text = text.replace("*", "")

        results.append((utt_id, text))

    # Write CSV with header
    with open(args.outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["錄音檔檔名", "辨認結果"])
        for k, v in results:
            w.writerow([k, v])

    print(f"[INFO] Wrote {len(results)} rows to {args.outfile}")
    print("[HINT] Run key-only filter and CER eval, e.g.:")
    print(f"  python make_keyonly_track2.py --key_dir FSR-2025-Hakka-evaluation-key "
          f"--pred_csv {args.outfile} --out Level-Up_hanzi_keyonly.csv")
    print("  python eval_track1_cer.py --key_dir FSR-2025-Hakka-evaluation-key "
          " --pred_csv Level-Up_hanzi_keyonly.csv --aligned_out aligned_hanzi.csv")

if __name__ == "__main__":
    main()
