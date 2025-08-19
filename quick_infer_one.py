#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch, soundfile as sf, librosa
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000
MODEL_NAME = "openai/whisper-large-v2"

def load_wav(path):
    wav, sr = sf.read(path)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    return wav

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <lora_dir> <audio_path>")
        sys.exit(1)

    lora_dir = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # 關掉 forced decoder / suppress token，直接轉寫
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    if hasattr(base, "generation_config") and base.generation_config is not None:
        base.generation_config.forced_decoder_ids = None
        base.generation_config.suppress_tokens = []
        base.generation_config.task = "transcribe"
        base.generation_config.language = None

    # 套上 LoRA 權重
    model = PeftModel.from_pretrained(base, lora_dir)
    model.to(device).eval()

    wav = load_wav(audio_path)
    feats = processor.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device)

    gen_kwargs = dict(
        do_sample=False,
        num_beams=1,   # greedy decoding
        return_dict_in_generate=False
    )

    with torch.inference_mode():
        pred_ids = model.generate(feats, **gen_kwargs)
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    print(f"\n[Audio] {audio_path}")
    print(f"[Output] {text}")

if __name__ == "__main__":
    main()
