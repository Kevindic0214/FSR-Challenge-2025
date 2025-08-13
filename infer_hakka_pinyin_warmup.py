#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import sys
import json
from typing import List

import torch
import soundfile as sf
import librosa
from tqdm import tqdm

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000
BASE_MODEL = "openai/whisper-large-v2"

def load_wav(path: Path):
    wav, sr = sf.read(str(path))
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    return wav

def norm_pinyin(text: str) -> str:
    # 只保留 a-z0-9 和空白，全部小寫、壓成單空白
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = " ".join(text.split())
    return text

def find_all_wavs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.wav"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", type=Path, default=Path("FSR-2025-Hakka-evaluation"),
                   help="熱身賽語料根目錄")
    p.add_argument("--lora_dir", type=Path, default=Path("exp_track2_whisper_large_lora"),
                   help="你訓練好的 LoRA 輸出資料夾")
    p.add_argument("--outfile", type=Path, default=Path("Level-Up_拼音.csv"),
                   help="輸出 CSV 檔名（可改成 '單位_隊名_拼音.csv'）")
    p.add_argument("--beams", type=int, default=1, help="beam size（1=greedy，較快）")
    p.add_argument("--batch", type=int, default=1, help="一次處理幾個檔案（建議 1）")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[INFO] Scanning wavs under: {args.eval_root}")
    wavs = find_all_wavs(args.eval_root)
    if not wavs:
        print("找不到任何 WAV，請確認路徑！", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(wavs)} wav files.")

    # 載入 Processor + Base + LoRA
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    # 關閉 Whisper 預設 forced/suppress，並指定做「轉寫」
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    if hasattr(base, "generation_config") and base.generation_config is not None:
        base.generation_config.forced_decoder_ids = None
        base.generation_config.suppress_tokens = []
        base.generation_config.task = "transcribe"
        base.generation_config.language = None

    model = PeftModel.from_pretrained(base, args.lora_dir)
    # 再保險一次
    model.config.forced_decoder_ids = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        model.generation_config.task = "transcribe"
        model.generation_config.language = None

    model.to(device).eval()

    # 以速度為主：greedy（num_beams=1）、不回傳 scores
    gen_kwargs = dict(
        do_sample=False,
        num_beams=max(1, args.beams),
        length_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=False,   # 關掉以加速
    )

    # 輸出 CSV（兩欄）
    # 欄名：錄音檔檔名、辨認出之客語漢字（比賽規格雖寫漢字，這裡實際填「拼音」）
    with args.outfile.open("w", encoding="utf-8") as fout, torch.inference_mode():
        fout.write("錄音檔檔名,辨認出之客語漢字\n")
        for w in tqdm(wavs, desc="Decoding"):
            # 取得檔名（不含副檔名）
            utt_id = w.stem

            # 前處理
            wav = load_wav(w)
            feats = processor.feature_extractor(
                wav, sampling_rate=SR, return_tensors="pt"
            ).input_features.to(device)

            # 轉寫
            out = model.generate(input_features=feats, **gen_kwargs)
            seq = out.sequences[0]
            text = processor.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0]
            text = norm_pinyin(text)

            # 寫出一行
            # 注意：例子沒附副檔名，所以只寫 stem
            print(f"{utt_id},{text}", file=fout)

    print(f"[DONE] CSV saved to: {args.outfile.resolve()}")

if __name__ == "__main__":
    main()
