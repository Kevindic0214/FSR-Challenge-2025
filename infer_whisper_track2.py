#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path
import torch, soundfile as sf, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

SR = 16000
MODEL_NAME = "openai/whisper-large-v2"
OUT_DIR = Path("exp_track2_whisper_large_lora")     # 你的 LoRA 目錄
JSONL = Path("HAT-Vol2/manifests/dev.jsonl")        # 要解碼的 jsonl（交作業時換成 pilot jsonl）
OUT_TSV = Path("decode_track2.tsv")                 # 輸出 (utt_id, text, avg_logprob)

def load_wav(path):
    wav, sr = sf.read(path)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if wav.ndim > 1:
        wav = wav.mean(-1)
    return wav

def safe_avg_logprob(gen_out, seq):
    # 1) 先用 sequences_scores（beam search 會有；greedy 也支援）
    if getattr(gen_out, "sequences_scores", None) is not None:
        try:
            return float(gen_out.sequences_scores[0].item())
        except Exception:
            pass
    # 2) 後備：用 token-level scores 粗估（對齊 offset = len(seq) - len(scores)）
    scores = getattr(gen_out, "scores", None)
    if not scores:
        return float("nan")
    T = len(scores)
    offset = max(0, seq.shape[-1] - T)   # 把 decoder prompt 長度吃掉
    toks = seq[offset:offset+T]
    total = 0.0
    for t, tok in enumerate(toks):
        step_logits = scores[t]
        # 不跟 beam_indices 硬對齊，直接取 row=0 當近似（已經是最佳序列）
        row = 0
        logp = torch.log_softmax(step_logits[row], dim=-1)[int(tok)].item()
        total += logp
    return total / max(1, T)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    # 關掉預設 forced/suppress、只做轉寫（不翻譯）
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    if hasattr(base, "generation_config") and base.generation_config is not None:
        base.generation_config.forced_decoder_ids = None
        base.generation_config.suppress_tokens = []
        base.generation_config.task = "transcribe"
        base.generation_config.language = None

    # 套上 LoRA
    model = PeftModel.from_pretrained(base, OUT_DIR)
    # 再保險一次
    model.config.forced_decoder_ids = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        model.generation_config.task = "transcribe"
        model.generation_config.language = None

    model.to(device).eval()

    gen_kwargs = dict(
        do_sample=False,
        num_beams=5,
        length_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=True,        # 需要這個才會有 sequences_scores / scores
    )

    items = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines()]
    with OUT_TSV.open("w", encoding="utf-8") as fout, torch.inference_mode():
        fout.write("utt_id\ttext\tavg_logprob\n")
        for ex in items:
            utt_id = ex.get("utt_id", Path(ex["audio"]).stem)
            wav = load_wav(ex["audio"])
            feats = processor.feature_extractor(wav, sampling_rate=SR, return_tensors="pt").input_features.to(device)

            out = model.generate(input_features=feats, **gen_kwargs)
            seq = out.sequences[0]

            text = processor.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0]
            text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
            text = ' '.join(text.split())

            score = safe_avg_logprob(out, seq)
            fout.write(f"{utt_id}\t{text}\t{score:.6f}\n")

    print(f"Done. Wrote {len(items)} lines to {OUT_TSV}")

if __name__ == "__main__":
    main()
