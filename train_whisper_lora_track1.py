#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper-large-v2 + LoRA for Track 1 (客語漢字)
- 讀取 JSONL manifest（每行至少含 {"utt_id","audio","text"} 或 {"utt_id","audio","hanzi"}）
- labels 以「漢字序列」訓練
- 驗證指標：CER（character error rate）
- 24GB 4090 友善（LoRA + gradient checkpointing + TF32）
"""

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

# ---- 減少 CUDA 記憶體碎片（顯著穩定）----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- 讓 4090 更穩（開啟 TF32）----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- I/O 路徑 ----------
MANI_DIR    = Path("HAT-Vol2/manifests_track1")
TRAIN_JSONL = MANI_DIR / "train.jsonl"
DEV_JSONL   = MANI_DIR / "dev.jsonl"
OUT_DIR     = Path("exp_track1_whisper_large_lora")

MODEL_NAME  = "openai/whisper-large-v2"
SR = 16000

# ---------- 資料集 ----------
class JsonlASRDataset(torch.utils.data.Dataset):
    """
    期待 jsonl 欄位：
      - 必要： "audio" (wav 路徑)
      - 文字： 優先用 "hanzi"，否則用 "text"（請確保為漢字）
      - 可選： "utt_id"
    """
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

        inputs = self.processor.feature_extractor(wav, sampling_rate=SR, return_tensors="pt")
        # --- 取漢字標籤 ---
        text_hz = ex.get("hanzi", ex.get("text", ""))
        text_hz = re.sub(r"\s+", "", str(text_hz))  # 去多餘空白

        labels = self.processor.tokenizer(text_hz, add_special_tokens=True).input_ids

        return {
            "input_features": inputs.input_features[0],
            "labels": torch.tensor(labels, dtype=torch.long),
            "utt_id": ex.get("utt_id", Path(ex["audio"]).stem),
        }

# ---------- collator ----------
@dataclass
class DataCollator:
    processor: WhisperProcessor
    def __call__(self, features):
        feats = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        return {
            "input_features": torch.stack(feats),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
            "utt_id": [f["utt_id"] for f in features],
        }

# ---------- CER ----------
def cer_metric(preds: List[str], refs: List[str]) -> float:
    """Character Error Rate（去掉空白後逐字比較）。"""
    def ed(a, b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1): dp[i][0] = i
        for j in range(len(b)+1): dp[0][j] = j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                c = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
        return dp[-1][-1]

    total_e, total_ref = 0, 0
    for h, r in zip(preds, refs):
        H = list(re.sub(r"[\s*]+", "", h))
        R = list(re.sub(r"[\s*]+", "", r))
        total_e += ed(H, R)
        total_ref += len(R)
    return 100.0 * total_e / max(1, total_ref)

# ---------- 主程式 ----------
def main():
    set_seed(1337)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    tokenizer: WhisperTokenizer = processor.tokenizer

    # （建議）pad_token 與 eos 對齊，避免 padding 警告
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 嘗試設定中文轉寫前綴（不同 transformers 版本 API 可能不同，故 try/except）
    try:
        # 只用於訓練時的 tokenizer 行為，不強制 generate 的 forced tokens
        processor.tokenizer.set_prefix_tokens(language="zh", task="transcribe")
    except Exception:
        pass

    # 模型：清空 forced/suppress，避免翻譯/時間戳/符號干擾
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # 省顯存
    # 某些情況下搭配 LoRA + ckpt 需要
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # LoRA 設定
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
        bias="none", task_type="SPEECH_SEQ_2_SEQ",
    )
    model = get_peft_model(model, lcfg)

    # 同步 generation 設定（避免被默認翻譯）
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = []
            model.generation_config.task = "transcribe"
            model.generation_config.language = None
    except Exception:
        pass

    train_ds = JsonlASRDataset(TRAIN_JSONL, processor)
    dev_ds   = JsonlASRDataset(DEV_JSONL, processor)
    collator = DataCollator(processor)

    # 解碼設定（eval 用 beam=5 較穩；最終可再試 1/8/10）
    gen_kwargs = dict(
        do_sample=False,
        num_beams=5,
        length_penalty=1.0,
        max_new_tokens=225,
        return_dict_in_generate=False,
        output_scores=False,
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=2,      # 24GB 卡建議起手式
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,     # 等效全域 batch 較大
        learning_rate=5e-4,
        num_train_epochs=3,                 # 先 3 個 epoch 快速落地
        warmup_steps=500,
        fp16=True,
        logging_steps=25,
        evaluation_strategy="epoch",
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
        # 僅傳 Whisper 需要的鍵，避免傳成 input_ids
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(input_features=inputs["input_features"], labels=inputs["labels"])
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # 每個 epoch 用 generate 計算 CER
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            self.model.eval()
            preds, refs = [], []
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
                    gen_ids = self.model.generate(
                        input_features=feats, 
                        forced_decoder_ids=None,
                        suppress_tokens=[],
                        **gen_kwargs
                    )
                    # 解碼為漢字
                    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                    texts = [re.sub(r"\s+", "", t) for t in texts]  # 去多餘空白
                    preds.extend(texts)

                    # 參考
                    lab = batch["labels"].cpu().numpy()
                    lab[lab == -100] = tokenizer.pad_token_id
                    ref_txts = processor.batch_decode(lab, skip_special_tokens=True)
                    ref_txts = [re.sub(r"\s+", "", t) for t in ref_txts]
                    refs.extend(ref_txts)

            cer = cer_metric(preds, refs)
            metrics = {f"{metric_key_prefix}_cer": cer}
            print(f"[EVAL] CER={cer:.2f}%")
            return metrics

    trainer = CERTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor,  # 用 processor（可保存 + 具備特徵與tokenizer）
        compute_metrics=None,
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)      # 只存 LoRA adapter（小很多）
    processor.save_pretrained(OUT_DIR)
    print("[DONE] Track1 training finished. Adapter saved at", OUT_DIR)

if __name__ == "__main__":
    main()
