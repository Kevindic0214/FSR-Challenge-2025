#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper-large-v2 + LoRA for Track 1 (Hakka Hanzi) — R3.3

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
import time
import math
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
try:
    from peft import TaskType
except ImportError:
    TaskType = None  # noqa: F811

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
        duration_sec = float(wav.numel()) / float(SR)

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
            "audio_path": apath,
            "group": ex.get("group", "XX"),
            "duration_sec": duration_sec,
            "has_ast": ("*" in txt_raw),
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
            "audio_path": [f.get("audio_path", "") for f in features],
            "group": [f.get("group", "XX") for f in features],
            "duration_sec": [float(f.get("duration_sec", 0.0)) for f in features],
            "has_ast": [bool(f.get("has_ast", False)) for f in features],
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
    # Logging / diagnostics
    ap.add_argument("--grad_log_interval", type=int, default=100,
                    help="Log grad_norm every N global steps")
    ap.add_argument("--hardest_k", type=int, default=50,
                    help="Top-K hardest samples to dump per evaluation")
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
    # 更穩定的梯度檢查點（避免 reentrant 帶來的問題）
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    # 嘗試開啟 Flash-Attention 2（若不可用就忽略）
    try:
        model.set_attn_implementation("flash_attention_2")
    except Exception:
        pass

    # LoRA (always set up for training)
    # Force speech-specific task type to avoid PEFT passing text-only kwargs (e.g., input_ids) to Whisper.
    # Using string keeps compatibility across PEFT versions.
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
        bias="none",
        task_type="SPEECH_SEQ_2_SEQ",
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

    # 讓 save 與 eval 的節奏一致，避免 best model 在 step 發生但只在 epoch 儲存
    _save_strategy = args.eval_strategy
    _save_steps = (args.eval_steps if args.eval_strategy == "steps" else None)
    
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
        save_strategy=_save_strategy,
        save_steps=_save_steps,
        eval_accumulation_steps=32,
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
            feats = inputs["input_features"].to(model.device, non_blocking=True)
            model_dtype = next(model.parameters()).dtype
            feats = feats.to(dtype=model_dtype)
            labels = inputs["labels"].to(model.device, non_blocking=True)
            outputs = model(input_features=feats, labels=labels)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            self.model.eval()
            preds, refs = [], []
            utt_ids_all, audio_paths_all, groups_all, durs_all, has_ast_all = [], [], [], [], []
            total_loss, total_items = 0.0, 0
            total_audio_sec = 0.0
            t0 = time.time()
            # reset peak memory to measure eval peak
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
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
                    # side info aggregation
                    utt_ids_all.extend(batch.get("utt_id", []))
                    audio_paths_all.extend(batch.get("audio_path", []))
                    grp_list = batch.get("group", ["XX"]) 
                    groups_all.extend(grp_list)
                    durs = [float(x) for x in batch.get("duration_sec", [0.0]*bs)]
                    durs_all.extend(durs)
                    total_audio_sec += float(sum(durs))
                    has_ast_all.extend([bool(x) for x in batch.get("has_ast", [False]*bs)])
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

            # Aggregate metrics
            wall = max(1e-6, time.time() - t0)
            cer = cer_metric(preds, refs)
            eval_loss = (total_loss / max(1, total_items)) if total_items > 0 else 0.0

            # Per-sample CER and ratio
            def _ed(a: str, b: str) -> int:
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
                        curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
                    prev, curr = curr, prev
                return prev[lb]

            per_cer = []  # percentage
            len_ratios = []
            hyp_lens, ref_lens = [] , []
            for h, r in zip(preds, refs):
                e = _ed(h, r)
                rlen = len(r)
                hlen = len(h)
                ref_len_safe = max(1, rlen)
                per_cer.append(100.0 * e / ref_len_safe)
                len_ratios.append((hlen / ref_len_safe) if ref_len_safe > 0 else float('nan'))
                hyp_lens.append(hlen)
                ref_lens.append(rlen)

            # Group CERs
            group_cer = {}
            for g in ["DF","DM","ZF","ZM","XX"]:
                idx = [i for i, gg in enumerate(groups_all) if gg == g]
                if idx:
                    gp = [preds[i] for i in idx]
                    gr = [refs[i] for i in idx]
                    group_cer[g] = cer_metric(gp, gr)

            # Length bucket CERs
            buckets = [
                ("[0-6.5s]", 0.0, 6.5),
                ("[6.5-9.6]", 6.5, 9.6),
                ("[9.6-12.4]", 9.6, 12.4),
                ("[12.4-20]", 12.4, 20.0),
            ]
            bucket_cer = {}
            for name, lo, hi in buckets:
                idx = [i for i, d in enumerate(durs_all) if (d >= lo and d < hi)]
                if idx:
                    bp = [preds[i] for i in idx]
                    br = [refs[i] for i in idx]
                    bucket_cer[name] = cer_metric(bp, br)

            # Asterisk sensitivity
            idx_has = [i for i, f in enumerate(has_ast_all) if f]
            idx_no = [i for i, f in enumerate(has_ast_all) if not f]
            cer_has = cer_metric([preds[i] for i in idx_has], [refs[i] for i in idx_has]) if idx_has else float('nan')
            cer_no = cer_metric([preds[i] for i in idx_no], [refs[i] for i in idx_no]) if idx_no else float('nan')

            # Degeneration: repeated character trigram in hypothesis + average duplicate count
            def trigram_dup_count(s: str) -> int:
                counts = {}
                for i in range(len(s) - 2):
                    tri = s[i:i+3]
                    counts[tri] = counts.get(tri, 0) + 1
                return sum((c - 1) for c in counts.values() if c >= 2)
            dup_counts = [trigram_dup_count(h) for h in preds]
            repeat_rate = sum(1 for c in dup_counts if c > 0) / max(1, len(dup_counts))
            dup_mean = (sum(dup_counts) / max(1, len(dup_counts)))

            # Length ratio stats
            lr_vals = [x for x in len_ratios if math.isfinite(x)]
            if lr_vals:
                lr_mean = float(sum(lr_vals) / len(lr_vals))
                srt = sorted(lr_vals)
                mid = len(srt) // 2
                lr_median = float(srt[mid]) if (len(srt) % 2 == 1) else float(0.5 * (srt[mid-1] + srt[mid]))
            else:
                lr_mean = float('nan'); lr_median = float('nan')

            # Throughput and memory
            audio_hours = total_audio_sec / 3600.0
            sps = (total_audio_sec / wall) if wall > 0 else float('nan')
            max_mem_gb = float('nan')
            if torch.cuda.is_available():
                try:
                    max_mem_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
                except Exception:
                    pass

            # Dump Top-K hardest samples
            try:
                # length bucket assignment for each sample
                def bucket_name(d: float) -> str:
                    if 0.0 <= d < 6.5: return "[0-6.5s]"
                    if 6.5 <= d < 9.6: return "[6.5-9.6]"
                    if 9.6 <= d < 12.4: return "[9.6-12.4]"
                    if 12.4 <= d < 20.0: return "[12.4-20]"
                    return ">=20"
                per_items = list(zip(utt_ids_all, audio_paths_all, groups_all, durs_all, has_ast_all, refs, preds, per_cer))
                per_items.sort(key=lambda x: (-x[7], -x[3]))
                step = getattr(self.state, 'global_step', None)
                epoch = getattr(self.state, 'epoch', None)
                stamp = f"step{step}" if step is not None else (f"epoch{int(epoch)}" if epoch is not None else "eval")
                out_tsv = Path(self.args.output_dir) / f"eval_topk_hard_{stamp}.tsv"
                k = max(1, int(args.hardest_k))
                with out_tsv.open('w', encoding='utf-8') as f:
                    # Order: utt_id, ref, hyp, cer, audio_path, group, len_bucket, has_ast
                    f.write("utt_id\tref\thyp\tcer\taudio_path\tgroup\tlen_bucket\thas_ast\n")
                    for (uid, ap, grp, dur, ha, r, h, cer_i) in per_items[:k]:
                        f.write(
                            f"{uid}\t{r}\t{h}\t{cer_i:.2f}\t{ap}\t{grp}\t{bucket_name(float(dur))}\t{int(ha)}\n"
                        )
            except Exception:
                pass

            hyp_len_mean = float(sum(hyp_lens) / len(hyp_lens)) if hyp_lens else float('nan')
            ref_len_mean = float(sum(ref_lens) / len(ref_lens)) if ref_lens else float('nan')

            metrics = {
                f"{metric_key_prefix}_cer": cer,
                f"{metric_key_prefix}_loss": eval_loss,
                "eval_seconds_per_second": sps,
                "eval_audio_hours": audio_hours,
                "eval_max_mem_gb": max_mem_gb,
                "eval_repeat_rate_3gram": repeat_rate,
                "eval_repeat_3gram_dup_count_mean": dup_mean,
                "eval_len_ratio_mean": lr_mean,
                "eval_len_ratio_median": lr_median,
                "eval_cer_has_ast": cer_has,
                "eval_cer_no_ast": cer_no,
                "eval_hyp_len_mean": hyp_len_mean,
                "eval_ref_len_mean": ref_len_mean,
            }
            for g, v in group_cer.items():
                metrics[f"{metric_key_prefix}_cer_{g}"] = v
            for name, v in bucket_cer.items():
                metrics[f"{metric_key_prefix}_cer_len_{name}"] = v

            # Filter out non-finite values to avoid logger/JSON errors
            safe_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    safe_metrics[k] = float(v)

            # Log so TensorBoard/checkpointing see the metrics, and trigger callbacks (EarlyStopping, best model)
            self.log(safe_metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, safe_metrics)
            print(f"[EVAL] CER={cer:.2f}%  |  loss={eval_loss:.4f}  |  sps={sps:.2f}  |  hrs={audio_hours:.3f}")
            return safe_metrics

    trainer = CERTrainer(
        model=model,
        args=args_hf,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor,  # Let Trainer save processor (contains tokenizer + feature_extractor)
        compute_metrics=None,
    )

    class ExtraStatsCallback(EarlyStoppingCallback):
        def __init__(self, early_stopping_patience=2, grad_log_interval=100):
            super().__init__(early_stopping_patience=early_stopping_patience)
            self.grad_log_interval = grad_log_interval

        def on_train_begin(self, args, state, control, **kwargs):
            trainer = getattr(self, 'trainer', None)
            model = trainer.model if trainer is not None else None
            if model is not None and trainer is not None:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                trainer.log({"train/trainable_params": float(trainable)})

        def on_step_end(self, args, state, control, **kwargs):
            trainer = getattr(self, 'trainer', None)
            if trainer is None:
                return
            step = state.global_step or 0
            if step > 0 and (step % max(1, self.grad_log_interval) == 0):
                try:
                    total_sq = 0.0
                    for p in trainer.model.parameters():
                        if p.grad is not None:
                            val = p.grad.data.norm(2).item()
                            if math.isfinite(val):
                                total_sq += val * val
                    trainer.log({"train/grad_norm": math.sqrt(total_sq)})
                except Exception:
                    pass

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Mirror HF 'loss' to 'train/loss' to keep tag naming consistent.
            Avoid recursion by only logging when 'loss' exists and we are not already logging the mirrored key.
            """
            trainer = getattr(self, 'trainer', None)
            if trainer is None or not isinstance(logs, dict):
                return
            if 'loss' in logs and 'train/loss' not in logs:
                try:
                    val = float(logs['loss'])
                    if math.isfinite(val):
                        trainer.log({"train/loss": val})
                except Exception:
                    pass

    trainer.add_callback(ExtraStatsCallback(early_stopping_patience=2, grad_log_interval=int(args.grad_log_interval)))
    trainer.train()
    model.save_pretrained(args.out_dir)      # Save LoRA adapter
    processor.save_pretrained(args.out_dir)
    print("[DONE] Track1 training finished. Adapter saved at", args.out_dir)

if __name__ == "__main__":
    main()
