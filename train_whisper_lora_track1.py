#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper-large-v2 + LoRA for Track 1 (Hakka Hanzi) — R3.1

- Read JSONL manifest (each line contains {"utt_id","audio","text"} or {"utt_id","audio","hanzi"})
- Consistent normalization with prepare/infer/eval (default keep '*', keep punctuation)
- Evaluation metric: CER (beam=5, forced Chinese transcription)
- 24GB GPU friendly: LoRA + gradient checkpointing + TF32 + bf16 (auto-detection)
"""
import os
import json
import argparse
import unicodedata
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torchaudio

from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, EarlyStoppingCallback,
    TrainingArguments, Trainer, set_seed
)
from peft import LoraConfig, get_peft_model

# ---- More stable memory management / 4090 friendly ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SR = 16000

# ---------- Normalization (aligned with prepare/infer/eval) ----------
_PUNCT_TABLE = str.maketrans({
    "，":"，","。":"。","、":"、","！":"！","？":"？","；":"；","：":"：",
    "（":"（","）":"）","「":"「","」":"」","『":"『","』":"』",
    ",":"，",".":"。","!":"！","?":"？",";":"；",":":"：",
    "(":"（",")":"）","[":"（","]":"）","{":"（","}":"）",
    "—":"－","–":"－","-":"－",
})
_ZW_CHARS_RE = _re.compile(r"[\u200B-\u200F\uFEFF]")
def normalize_hanzi(text: str, strip_spaces=True, keep_asterisk=True, strip_punct=False) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _ZW_CHARS_RE.sub("", t)
    t = t.translate(_PUNCT_TABLE)
    if strip_spaces:
        t = _re.sub(r"\s+", "", t)
    if not keep_asterisk:
        t = t.replace("*", "")
    if strip_punct:
        t = _re.sub(r"[，。、！？」；：「『』（）－,\.!\?:;\[\]\{\}\(\)\"']", "", t)
    return t

# ---------- Dataset ----------
class JsonlASRDataset(torch.utils.data.Dataset):
    """
    JSONL fields:
      - "audio": audio file path (can be relative to --root)
      - "hanzi" or "text": Hakka Hanzi annotation
      - "utt_id": optional
    """
    def __init__(self, jsonl_path: Path, processor: WhisperProcessor, root: Optional[Path],
                 keep_asterisk: bool, strip_punct: bool):
        self.lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        self.processor = processor
        self.root = root
        self.keep_asterisk = keep_asterisk
        self.strip_punct = strip_punct

    def __len__(self): return len(self.lines)

    def _resolve_audio(self, p: str) -> str:
        P = Path(p)
        if not P.is_absolute():
            if self.root is None:
                raise RuntimeError(f"Relative audio path but --root not provided: {p}")
            P = (self.root / P).resolve()
        return str(P)

    def __getitem__(self, i: int):
        ex = json.loads(self.lines[i])
        apath = self._resolve_audio(ex["audio"])

        wav, sr = torchaudio.load(apath)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.squeeze(0)

        feats = self.processor.feature_extractor(wav.numpy(), sampling_rate=SR, return_tensors="pt")

        txt_raw = ex.get("hanzi", ex.get("text", ""))
        txt_norm = normalize_hanzi(
            txt_raw, strip_spaces=True,
            keep_asterisk=self.keep_asterisk,
            strip_punct=self.strip_punct
        )

        ids = self.processor.tokenizer(txt_norm, add_special_tokens=True).input_ids

        return {
            "input_features": feats.input_features[0],
            "labels": torch.tensor(ids, dtype=torch.long),
            "utt_id": ex.get("utt_id", Path(apath).stem),
        }

# ---------- Collator ----------
@dataclass
class DataCollator:
    def __call__(self, features: List[Dict[str, Any]]):
        feats = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        return {
            "input_features": torch.stack(feats),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            "utt_id": [f["utt_id"] for f in features],
        }

# ---------- CER ----------
def cer_metric(preds: List[str], refs: List[str]) -> float:
    def ed(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0: return lb  # noqa: E701
        if lb == 0: return la  # noqa: E701
        prev = list(range(lb + 1))
        curr = [0]*(lb + 1)
        for i in range(1, la + 1):
            curr[0] = i
            ca = a[i-1]
            for j in range(1, lb + 1):
                cb = b[j-1]
                cost = 0 if ca == cb else 1
                curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
            prev, curr = curr, prev
        return prev[lb]
    total_e, total_ref = 0, 0
    for h, r in zip(preds, refs):
        total_e += ed(h, r)
        total_ref += len(r)
    return 100.0 * total_e / max(1, total_ref)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=Path, default=Path("HAT-Vol2/manifests_track1/train.jsonl"))
    ap.add_argument("--dev_jsonl",   type=Path, default=Path("HAT-Vol2/manifests_track1/dev.jsonl"))
    ap.add_argument("--root",        type=Path, default=Path("HAT-Vol2"), help="Root directory for joining relative audio paths")
    ap.add_argument("--out_dir",     type=Path, default=Path("runs/track1/lora_v2_r16_e3"))
    ap.add_argument("--base_model",  type=str, default="openai/whisper-large-v2")
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_strategy", choices=["epoch","steps"], default="epoch",
                    help="When to run evaluation: per epoch or per steps")
    ap.add_argument("--eval_steps", type=int, default=500,
                    help="Run evaluation every N steps when --eval_strategy=steps")
    # Normalization switches
    ap.add_argument("--strip_asterisk", action="store_true", help="Whether to remove '*' from training text (default: keep)")
    ap.add_argument("--strip_punct", action="store_true", help="Whether to remove punctuation from training text (default: keep)")
    args = ap.parse_args()

    set_seed(1337)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # dtype selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    use_bf16 = bf16_ok
    print(f"[INFO] Device={device}, bf16={use_bf16}")

    # Processor & tokenizer
    processor = WhisperProcessor.from_pretrained(args.base_model)
    tokenizer: WhisperTokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=(torch.bfloat16 if use_bf16 else None),
        low_cpu_mem_usage=True,
    )
    # Common Whisper training settings
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    # Keep model defaults (incl. suppress_tokens); don't forcibly clear them
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # LoRA (always set up for training)
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
        bias="none", task_type="SPEECH_SEQ_2_SEQ",
    )
    model = get_peft_model(model, lcfg)

    # Dataset
    keep_ast = not args.strip_asterisk
    train_ds = JsonlASRDataset(args.train_jsonl, processor, args.root, keep_ast, args.strip_punct)
    dev_ds   = JsonlASRDataset(args.dev_jsonl,   processor, args.root, keep_ast, args.strip_punct)
    collator = DataCollator()

    # Generation settings (for eval)
    forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    gen_kwargs = dict(
        do_sample=False,
        num_beams=5,
        temperature=0.0,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        max_new_tokens=256,
        forced_decoder_ids=forced_ids,
        # suppress_tokens=[], # Keep model default
        return_dict_in_generate=False,
        output_scores=False,
    )

    # TrainingArguments
    args_hf = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=500,
        bf16=use_bf16,
        fp16=(not use_bf16 and device=="cuda"),
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        logging_steps=25,
        evaluation_strategy=args.eval_strategy,
        eval_steps=(args.eval_steps if args.eval_strategy == "steps" else None),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    class CERTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(input_features=inputs["input_features"], labels=inputs["labels"])
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            self.model.eval()
            preds, refs = [], []
            total_loss, total_items = 0.0, 0
            loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=2,
                pin_memory=True,
            )
            with torch.no_grad():
                for batch in loader:
                    feats = batch["input_features"].to(self.model.device, non_blocking=True)
                    # Feature dtype must align with model dtype
                    model_dtype = next(self.model.parameters()).dtype
                    feats = feats.to(dtype=model_dtype)
                    # Eval loss (teacher-forced)
                    labels = batch["labels"].to(self.model.device, non_blocking=True)
                    out = self.model(input_features=feats, labels=labels)
                    loss_val = float(out.loss.detach().cpu().item())
                    bs = labels.size(0)
                    total_loss += loss_val * bs
                    total_items += bs
                    # Generation for CER
                    ids = self.model.generate(input_features=feats, **gen_kwargs)
                    hyp = processor.batch_decode(ids, skip_special_tokens=True)
                    hyp = [normalize_hanzi(t, strip_spaces=True, keep_asterisk=keep_ast, strip_punct=args.strip_punct)
                           for t in hyp]
                    # Reference (for CER)
                    lab = batch["labels"].cpu().numpy()
                    lab[lab == -100] = tokenizer.pad_token_id
                    ref = processor.batch_decode(lab, skip_special_tokens=True)
                    ref = [normalize_hanzi(t, strip_spaces=True, keep_asterisk=keep_ast, strip_punct=args.strip_punct)
                           for t in ref]
                    preds.extend(hyp)
                    refs.extend(ref)

            cer = cer_metric(preds, refs)
            eval_loss = (total_loss / max(1, total_items)) if total_items > 0 else 0.0
            print(f"[EVAL] CER={cer:.2f}%  |  loss={eval_loss:.4f}")
            return {f"{metric_key_prefix}_cer": cer, f"{metric_key_prefix}_loss": eval_loss}

    trainer = CERTrainer(
        model=model,
        args=args_hf,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor,  # Let Trainer save processor (contains tokenizer + feature_extractor)
        compute_metrics=None,
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    trainer.train()
    model.save_pretrained(args.out_dir)      # Save LoRA adapter
    processor.save_pretrained(args.out_dir)
    print("[DONE] Track1 training finished. Adapter saved at", args.out_dir)

if __name__ == "__main__":
    main()
