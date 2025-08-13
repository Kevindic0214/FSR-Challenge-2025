#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

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

# ---------- I/O 路徑 ----------
MANI_DIR = Path("HAT-Vol2/manifests_track2")
TRAIN_JSONL = MANI_DIR / "train.jsonl"
DEV_JSONL   = MANI_DIR / "dev.jsonl"
OUT_DIR     = Path("exp_track2_whisper_large_lora")

MODEL_NAME  = "openai/whisper-large-v2"
SR = 16000

# ---------- 資料集 ----------
class JsonlASRDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, processor: WhisperProcessor):
        self.items = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines()]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ex = self.items[i]
        wav, _sr = sf.read(ex["audio"])
        if _sr != SR:
            wav = librosa.resample(wav, orig_sr=_sr, target_sr=SR)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

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
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
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
    set_seed(1337)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    tokenizer: WhisperTokenizer = processor.tokenizer

    # 模型 & 記憶體優化
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # LoRA 訓練時避免 cache
    model.gradient_checkpointing_enable()  # 省顯存，務必開

    # LoRA 設定（注意 target_modules 名稱對齊 Whisper）
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
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

    train_ds = JsonlASRDataset(TRAIN_JSONL, processor)
    dev_ds   = JsonlASRDataset(DEV_JSONL, processor)
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
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=2,     # 若仍吃緊 -> 改成 1
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,    # 保持較大的有效 batch
        learning_rate=5e-4,
        num_train_epochs=3,                # 先跑 3epoch 對付 8/11；之後可加長
        warmup_steps=500,
        fp16=True,
        logging_steps=25,
        evaluation_strategy="epoch",       # 舊版 transformers 相容
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_ser",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # 自定義 Trainer：保證把 input_features 傳進去（避免傳成 input_ids）
    class SERTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(input_features=inputs["input_features"], labels=inputs["labels"])
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            self.model.eval()
            preds, refs = [], []
            loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=collator,
                num_workers=2,
                pin_memory=True,
            )
            for batch in loader:
                with torch.no_grad():
                    input_features = batch["input_features"].to(self.model.device, non_blocking=True)
                    gen_ids = self.model.generate(
                        input_features=input_features,  # 這裡用 input_features（Whisper 專用）
                        **gen_kwargs,
                    )
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                texts = [re.sub(r'[^a-z0-9\s]', ' ', t.lower()) for t in texts]
                texts = [' '.join(t.split()) for t in texts]
                preds.extend(texts)

                lab = batch["labels"].cpu().numpy()
                lab[lab == -100] = tokenizer.pad_token_id
                refs.extend(processor.batch_decode(lab, skip_special_tokens=True))

            ser = ser_metric(preds, refs)
            metrics = {f"{metric_key_prefix}_ser": ser}
            print(f"[EVAL] SER={ser:.2f}%")
            return metrics

    trainer = SERTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor.feature_extractor,  # 讓 Trainer 知道如何 pad
        compute_metrics=None,
    )

    trainer.train()
    # 只存 LoRA 參數（體積小、部署方便）
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print("[DONE] Training finished. Best adapter saved at", OUT_DIR)

if __name__ == "__main__":
    main()
