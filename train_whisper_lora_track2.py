#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
import torch
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import List

from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer,
    TrainingArguments, Trainer, set_seed
)
from peft import LoraConfig, get_peft_model

# ---------- I/O 路徑 ----------
MANI_DIR = Path("HAT-Vol2/manifests")  # <- 依需要修改
TRAIN_JSONL = MANI_DIR/"train.jsonl"
DEV_JSONL   = MANI_DIR/"dev.jsonl"
OUT_DIR     = Path("exp_track2_whisper_medium_lora")

MODEL_NAME  = "openai/whisper-large-v2"  # 可改 large-v2（算力足再換）

SR = 16000

# ---------- 資料集 ----------
class JsonlASRDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, processor: WhisperProcessor):
        self.items = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
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
        # Whisper tokenizer 可直接吃我們的 a-z0-9 序列；不需要 as_target_tokenizer
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
        batch = {
            "input_features": torch.stack(input_feats),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        }
        return batch

# ---------- SER（音節級） ----------
def ser_metric(preds: List[str], refs: List[str]) -> float:
    # 以空白分詞為音節，做 Levenshtein
    def ed(a, b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1):
            dp[i][0] = i
        for j in range(len(b)+1):
            dp[0][j] = j
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                c = 0 if a[i-1]==b[j-1] else 1
                dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
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

    # 關閉語言/時間戳等預設特殊行為（我們是單語拼音序列）
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    # 關閉 Whisper 預設的 forced tokens / 抑制清單，避免加標點/時間戳
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # 關閉 cache，因為我們會用 LoRA 訓練，cache 會有問題
    model.gradient_checkpointing_enable()
    # 也同步到 generation_config（有些版本會讀這裡）
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.suppress_tokens = []

    # LoRA 設定
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","out_proj", "fc1", "fc2"],
        bias="none", task_type="SPEECH_SEQ_2_SEQ"
    )
    model = get_peft_model(model, lcfg)
    # 再次保險：PEFT 包裝後也同步設定生成相關參數
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

    # 生成時的一些設定：不要自動加標點、不要翻譯
    gen_kwargs = dict(
        do_sample=False,
        num_beams=5,
        length_penalty=1.0,
        suppress_tokens=None,
        output_scores=False,
        return_dict_in_generate=False,
    )

    def compute_metrics(eval_pred):
        # 用 greedy/beam 生成再算 SER
        logits, labels = eval_pred  # Trainer 預設給 raw logits/labels，不好直接轉文字；改用自定評估
        # 我們改用 on_evaluate hook：於 eval_step 內用 model.generate，較繁雜；這裡採簡化版：
        return {}

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        num_train_epochs=10,
        warmup_steps=500,
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_ser",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2
    )

    # 自定義 eval：在每個 eval epoch 用 generate 計算 SER
    class SERTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            input_features = inputs["input_features"]
            labels = inputs["labels"]
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            self.model.eval()
            preds, refs = [], []
            for batch in torch.utils.data.DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=collator):
                with torch.no_grad():
                    input_features = batch["input_features"].to(self.model.device)
                    gen_ids = self.model.generate(
                        inputs=input_features,
                        **gen_kwargs,
                    )
                # 轉文字
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                # 正規化（跟訓練一致）
                texts = [re.sub(r'[^a-z0-9\s]', ' ', t.lower()) for t in texts]
                texts = [' '.join(t.split()) for t in texts]
                preds.extend(texts)
                # 取參考
                lab = batch["labels"].cpu().numpy()
                # 將 -100 還原成 pad，解碼
                lab[lab == -100] = tokenizer.pad_token_id
                refs.extend(processor.batch_decode(lab, skip_special_tokens=True))
            ser = ser_metric(preds, refs)
            metrics = {f"{metric_key_prefix}_ser": ser}
            self.control.should_save = True
            print(f"[EVAL] SER={ser:.2f}%")
            return metrics

    trainer = SERTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor.feature_extractor,  # 只為了讓 Trainer 知道 pad 方法
        compute_metrics=None
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print("[DONE] Training finished. Best model saved at", OUT_DIR)

if __name__ == "__main__":
    main()
