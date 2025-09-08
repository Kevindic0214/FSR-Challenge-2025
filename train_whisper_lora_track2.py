#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import soundfile as sf
import librosa

from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer,
    TrainingArguments, Trainer, set_seed
)
from peft import LoraConfig, get_peft_model

# ---- 建議：減少碎片化（可選，但有助穩定）----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- 讓 4090 跑更穩 ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- 預設 I/O 路徑（可由 CLI 覆寫） ----------
DEF_MANIFEST_DIR = Path("HAT-Vol2/manifests_track2")
DEF_TRAIN_JSONL = DEF_MANIFEST_DIR / "train.jsonl"
DEF_DEV_JSONL   = DEF_MANIFEST_DIR / "dev.jsonl"
DEF_OUT_DIR     = Path("exp_track2_whisper_large_lora")

DEF_MODEL_NAME  = "openai/whisper-large-v2"
SR = 16000

# ---------- 資料集 ----------
class JsonlASRDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, processor: WhisperProcessor, audio_root: Optional[Path] = None):
        self.items = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines()]
        self.processor = processor
        self.audio_root = audio_root.resolve() if audio_root is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ex = self.items[i]
        apath = Path(ex.get("audio") or ex.get("path"))  # 相對或絕對
        if not apath.is_absolute() and self.audio_root is not None:
            apath = (self.audio_root / apath).resolve()
        wav, _sr = sf.read(str(apath))
        if _sr != SR:
            wav = librosa.resample(wav, orig_sr=_sr, target_sr=SR)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        duration_sec = float(len(wav)) / float(SR)

        inputs = self.processor.feature_extractor(
            wav, sampling_rate=SR, return_tensors="pt"
        )
        # Track2: 直接用空白分隔的 a-z0-9 拼音序列
        labels = self.processor.tokenizer(
            ex["text"], add_special_tokens=True
        ).input_ids

        return {
            "input_features": inputs.input_features[0],
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": ex["text"],
            "utt_id": ex.get("utt_id", ex.get("id", Path(str(apath)).stem)),
            "audio_path": str(apath),
            "group": ex.get("group", "XX"),
            "duration_sec": duration_sec,
        }

# ---------- collator ----------
@dataclass
class DataCollator:
    processor: WhisperProcessor
    def __call__(self, features):
        input_feats = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        return {
            "input_features": torch.stack(input_feats),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            "utt_id": [f.get("utt_id", "") for f in features],
            "audio_path": [f.get("audio_path", "") for f in features],
            "group": [f.get("group", "XX") for f in features],
            "duration_sec": [float(f.get("duration_sec", 0.0)) for f in features],
        }

# ---------- SER ----------
def ser_metric(preds: List[str], refs: List[str]) -> float:
    def ed(a, b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1): dp[i][0] = i
        for j in range(len(b)+1): dp[0][j] = j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                c = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
        return dp[-1][-1]

    total_edit, total_ref = 0, 0
    for hyp, ref in zip(preds, refs):
        h = hyp.split()
        r = ref.split()
        total_edit += ed(h, r)
        total_ref  += len(r)
    return 100.0 * total_edit / max(1, total_ref)

# ---------- 主程式 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=Path, default=DEF_TRAIN_JSONL)
    ap.add_argument("--dev_jsonl",   type=Path, default=DEF_DEV_JSONL)
    ap.add_argument("--audio_root",  type=Path, default=None, help="If manifest uses relative audio paths, join with this root")
    ap.add_argument("--out_dir",     type=Path, default=DEF_OUT_DIR)
    ap.add_argument("--base_model",  type=str, default=DEF_MODEL_NAME)
    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--report_to", type=str, default="none", choices=["none","tensorboard"])
    ap.add_argument("--seed", type=int, default=1337)
    # Diagnostics
    ap.add_argument("--grad_log_interval", type=int, default=100, help="log grad_norm every N steps")
    args = ap.parse_args()

    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.base_model)
    tokenizer: WhisperTokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 裝置與 dtype 偵測
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    use_bf16 = bf16_ok

    # 模型 & 記憶體優化
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=(torch.bfloat16 if use_bf16 else None),
        low_cpu_mem_usage=True,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # LoRA 訓練時避免 cache
    # 更穩定的梯度檢查點
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    try:
        model.set_attn_implementation("flash_attention_2")
    except Exception:
        pass

    # LoRA 設定（注意 target_modules 名稱對齊 Whisper）
    lcfg = LoraConfig(
        r=int(args.lora_r), lora_alpha=int(args.lora_alpha), lora_dropout=float(args.lora_dropout),
        target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
        bias="none", task_type="SPEECH_SEQ_2_SEQ",
    )
    model = get_peft_model(model, lcfg)

    # 再保險：同步 generation_config
    try:
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.suppress_tokens = []
    except Exception:
        pass

    train_ds = JsonlASRDataset(args.train_jsonl, processor, audio_root=args.audio_root)
    dev_ds   = JsonlASRDataset(args.dev_jsonl, processor, audio_root=args.audio_root)
    collator = DataCollator(processor)

    # 解碼設定
    gen_kwargs = dict(
        do_sample=False,
        num_beams=5,              # pilot-test 可先 5；若 OOM 可臨時改 1
        length_penalty=1.0,
        suppress_tokens=None,
        output_scores=False,
        return_dict_in_generate=False,
    )

    # 訓練參數（為 24GB 卡調的值）
    hf_args = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        num_train_epochs=int(args.epochs),
        warmup_steps=500,
        bf16=use_bf16,
        fp16=(not use_bf16 and device=="cuda"),
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        logging_steps=25,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_ser",
        greater_is_better=False,
        report_to=([] if args.report_to=="none" else [args.report_to]),
        remove_unused_columns=False,
        save_total_limit=2,
        eval_accumulation_steps=32,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # 自定義 Trainer：保證把 input_features 傳進去（避免傳成 input_ids）
    class SERTrainer(Trainer):
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
            utt_ids_all, audio_paths_all, groups_all, durs_all = [], [], [], []
            total_audio_sec = 0.0
            # loader
            loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=2,
                pin_memory=True,
            )
            import time, math
            t0 = time.time()
            # reset GPU peak mem
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            # decode loop
            with torch.no_grad():
                for batch in loader:
                    feats = batch["input_features"].to(self.model.device, non_blocking=True)
                    model_dtype = next(self.model.parameters()).dtype
                    feats = feats.to(dtype=model_dtype)
                    ids = self.model.generate(input_features=feats, **gen_kwargs)
                    hyp = processor.batch_decode(ids, skip_special_tokens=True)
                    hyp = [re.sub(r'[^a-z0-9\s]', ' ', t.lower()) for t in hyp]
                    hyp = [' '.join(t.split()) for t in hyp]
                    preds.extend(hyp)

                    lab = batch["labels"].cpu().numpy()
                    lab[lab == -100] = tokenizer.pad_token_id
                    ref = processor.batch_decode(lab, skip_special_tokens=True)
                    ref = [re.sub(r'[^a-z0-9\s]', ' ', t.lower()) for t in ref]
                    ref = [' '.join(t.split()) for t in ref]
                    refs.extend(ref)

                    # meta
                    bs = len(hyp)
                    utt_ids_all.extend(batch.get("utt_id", [""]*bs))
                    audio_paths_all.extend(batch.get("audio_path", [""]*bs))
                    groups_all.extend(batch.get("group", ["XX"]*bs))
                    durs = [float(x) for x in batch.get("duration_sec", [0.0]*bs)]
                    durs_all.extend(durs)
                    total_audio_sec += float(sum(durs))

            # Overall SER
            ser = ser_metric(preds, refs)
            # Group SER
            def tokens(s: str):
                return s.split()
            def ed_tok(a: List[str], b: List[str]) -> int:
                n, m = len(a), len(b)
                if n == 0: return m
                if m == 0: return n
                dp = list(range(m + 1))
                for i in range(1, n + 1):
                    prev = dp[0]
                    dp[0] = i
                    ai = a[i-1]
                    for j in range(1, m + 1):
                        tmp = dp[j]
                        cost = 0 if ai == b[j-1] else 1
                        dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
                        prev = tmp
                return dp[m]
            # per-utt SER for hardest list
            per_ser = []
            for h, r in zip(preds, refs):
                ht, rt = tokens(h), tokens(r)
                E = ed_tok(ht, rt)
                per_ser.append(100.0 * E / max(1, len(rt)))

            # Group SERs
            group_ser: Dict[str, float] = {}
            for g in ["DF","DM","ZF","ZM","XX"]:
                idx = [i for i, gg in enumerate(groups_all) if gg == g]
                if idx:
                    gp = [preds[i] for i in idx]
                    gr = [refs[i] for i in idx]
                    group_ser[g] = ser_metric(gp, gr)

            # Length bucket SERs (same as Track 1 buckets)
            buckets = [
                ("[0-6.5s]", 0.0, 6.5),
                ("[6.5-9.6]", 6.5, 9.6),
                ("[9.6-12.4]", 9.6, 12.4),
                ("[12.4-20]", 12.4, 20.0),
            ]
            bucket_ser: Dict[str, float] = {}
            for name, lo, hi in buckets:
                idx = [i for i, d in enumerate(durs_all) if (d >= lo and d < hi)]
                if idx:
                    gp = [preds[i] for i in idx]
                    gr = [refs[i] for i in idx]
                    bucket_ser[name] = ser_metric(gp, gr)

            # Throughput and memory
            wall = max(1e-6, time.time() - t0)
            sps = (total_audio_sec / wall) if wall > 0 else float('nan')
            max_mem_gb = float('nan')
            if torch.cuda.is_available():
                try:
                    max_mem_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
                except Exception:
                    pass

            # Dump Top-K hardest samples
            try:
                hardest_k = 50
                per = list(zip(utt_ids_all, audio_paths_all, groups_all, durs_all, refs, preds, per_ser))
                per.sort(key=lambda x: (-x[6], -x[3]))  # sort by SER desc, then duration desc
                step = getattr(self.state, 'global_step', None)
                epoch = getattr(self.state, 'epoch', None)
                stamp = f"step{step}" if step is not None else (f"epoch{int(epoch)}" if epoch is not None else "eval")
                out_tsv = Path(self.args.output_dir) / f"eval_topk_hard_{stamp}.tsv"
                with out_tsv.open('w', encoding='utf-8') as f:
                    f.write("utt_id\tref\thyp\tser\taudio_path\tgroup\tlen_bucket\n")
                    def bucket_name(d: float) -> str:
                        if 0.0 <= d < 6.5: return "[0-6.5s]"
                        if 6.5 <= d < 9.6: return "[6.5-9.6]"
                        if 9.6 <= d < 12.4: return "[9.6-12.4]"
                        if 12.4 <= d < 20.0: return "[12.4-20]"
                        return ">=20"
                    for (uid, ap, grp, dur, r, h, ser_i) in per[:hardest_k]:
                        f.write(f"{uid}\t{r}\t{h}\t{ser_i:.2f}\t{ap}\t{grp}\t{bucket_name(float(dur))}\n")
            except Exception:
                pass

            metrics: Dict[str, Any] = {f"{metric_key_prefix}_ser": ser,
                                        "eval_seconds_per_second": sps,
                                        "eval_audio_hours": (total_audio_sec / 3600.0),
                                        "eval_max_mem_gb": max_mem_gb}
            for g, v in group_ser.items():
                metrics[f"{metric_key_prefix}_ser_{g}"] = v
            for name, v in bucket_ser.items():
                metrics[f"{metric_key_prefix}_ser_len_{name}"] = v
            print(f"[EVAL] SER={ser:.2f}% | sps={sps:.2f} | hrs={total_audio_sec/3600.0:.3f}")
            return metrics

    trainer = SERTrainer(
        model=model,
        args=hf_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor,  # 保存 processor（含 tokenizer + feature_extractor）
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
            step = getattr(state, 'global_step', 0) or 0
            if trainer is None or step <= 0 or (step % max(1, self.grad_log_interval) != 0):
                return
            try:
                total_sq = 0.0
                for p in trainer.model.parameters():
                    if p.grad is not None:
                        val = p.grad.data.norm(2).item()
                        if not (val != val):  # NaN check
                            total_sq += val * val
                trainer.log({"train/grad_norm": (total_sq ** 0.5)})
            except Exception:
                pass

        def on_log(self, args, state, control, logs=None, **kwargs):
            trainer = getattr(self, 'trainer', None)
            if trainer is None or not isinstance(logs, dict):
                return
            if 'loss' in logs and 'train/loss' not in logs:
                try:
                    trainer.log({"train/loss": float(logs['loss'])})
                except Exception:
                    pass

    trainer.add_callback(ExtraStatsCallback(early_stopping_patience=2, grad_log_interval=int(args.grad_log_interval)))

    trainer.train()
    # 只存 LoRA 參數（體積小、部署方便）
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print("[DONE] Training finished. Best adapter saved at", args.out_dir)

if __name__ == "__main__":
    main()
