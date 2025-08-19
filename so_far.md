# FSR-2025 Hakka ASR（Level-Up）技術規格 v0.1

> Author: Kevin（NYCU）
> GPU: RTX 4090 24 GB（CUDA 12.6）
> 目標：**先完成 Pilot / 熱身賽提交**，同時建立可擴充到 Final 的可復現流程

---

## 1) 競賽概要

* **主辦**：NYCU（Formosa Speech Recognition Challenge 2025 – Hakka ASR II）
* **任務**：建立能辨識客語（**朗讀/自發**）的 ASR
* **Track**

  * **Track 1**：客語漢字（CER）
  * **Track 2**：客語拼音（SER, 本調）
* **資料**

  * 訓練使用：主辦提供 **FSR-2025-Hakka-train**（約 60 h，六堆/詔安，單聲道 16 kHz）
  * 評測：**HAT-Vol2**（約 80 h，內含 eval/test；test 不可人工聽取）
  * 熱身賽（Pilot）：**FSR-2025-Hakka-evaluation**（錄製 \~8 h + 媒體 \~2 h）
* **提交**

  * 兩個 CSV：`客語漢字.csv`、`客語拼音.csv`
    欄位：`錄音檔檔名, 辨認結果`
  * 檔名格式：`單位_隊名_漢字.csv`、`單位_隊名_拼音.csv`（本隊：**Level-Up**）

---

## 2) 參考資料／論文（精選）

* FSR 2023 / ROCLING 2023 競賽論文（共 10 篇，檔名 `2023.rocling-1.46.pdf` \~ `1.56.pdf`）

  * 以 **46、47** 兩篇作為優先參考（方法描述較完整，便於復現）
* 官方/社群資源

  * Whisper Hakka baseline（主辦 GitHub）
  * ESPnet baseline（主辦 GitHub）
  * FSR-2023 scoring script（SER/CER 計算）

> 註：本版先落地 **Whisper + LoRA** 流程；若需，我們再彙整每篇論文的具體技巧（增強、LM rescoring、多任務等）並做消融研究。

---

## 3) 目前實作（可復現）

### 3.1 專案結構（重點檔案）

```
FSR-Challenge-2025/
├── prepare_hakka_track2.py          # 生成 Track2 (拼音) train/dev manifests
├── train_whisper_lora_track2.py     # Whisper-large-v2 + LoRA 訓練 (拼音)
├── infer_whisper_track2.py          # 批量解碼；含平均 logprob 估計
├── infer_hakka_pinyin_warmup.py     # 產生熱身賽「Level-Up_拼音.csv」
├── quick_infer_one.py               # 快速單檔驗證（語音→拼音）
│
├── prepare_hakka_track1.py          # 生成 Track1 (漢字) train/dev manifests
├── train_whisper_lora_track1.py     # Whisper-large-v2 + LoRA 訓練 (漢字)
#（建議新增）infer_hakka_hanzi_warmup.py  # 產生熱身賽「Level-Up_漢字.csv」
│
├── HAT-Vol2/                        # 原始資料 + manifests*/  (train/dev jsonl)
├── FSR-2025-Hakka-evaluation/       # 熱身賽資料（錄製+媒體）
├── exp_track2_whisper_large_lora/   # Track2 LoRA/processor 輸出
└── exp_track1_whisper_large_lora/   # Track1 LoRA/processor 輸出
```

### 3.2 資料前處理

**Track 2（拼音）** – `prepare_hakka_track2.py`

* 讀 `*_edit.csv`，欄位：`檔名, 客語拼音, 備註`
* 規則化：

  * 小寫化、保留 `[a-z0-9]` 與空白（音節以空白分隔）
  * 將合音 `來*去` 這類記號展開為拼音序列（保留正確發音音節）
* 資料清理：

  * 可選擇剔除備註含「**正確讀音**」的樣本（避免訓練時學到錯誤發音）
* 切分：

  * **dev 12 位說話人**（DF/DM/ZF/ZM 平衡），其餘做 train
* 產生 `HAT-Vol2/manifests/{train,dev}.jsonl`

**Track 1（漢字）** – `prepare_hakka_track1.py`

* 讀 `*_edit.csv`，欄位：`檔名, 客語漢字, 備註`
* 規則化：

  * 移除多餘空白；合音星號 `*` 可保留或移除（預設保留；提交前會移除）
* 清理與切分方式與 Track2 一致
* 產生 `HAT-Vol2/manifests_track1/{train,dev}.jsonl`

> 數量（目前一次執行結果）
> Track2：`kept=26757, dropped_mispronounced=592, train_utt=24038, dev_utt=2719`
> Track1：`kept=26757, dropped_mispronounced=592, train_utt=24061, dev_utt=2696`

### 3.3 模型與訓練（共同設定）

* **Base**：`openai/whisper-large-v2`
* **微調**：**PEFT–LoRA**

  * `r=8, alpha=16, dropout=0.1`
  * `target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]`
* **顯存優化**：`gradient_checkpointing_enable()`、`use_cache=False`、TF32 on
* **輸入**：log-Mel（WhisperProcessor.feature\_extractor，SR=16 k）
* **輸出**：

  * Track2：空白分隔之音節序列（a-z0-9）
  * Track1：客語漢字序列（不含空白）
* **生成設定（驗證）**：`num_beams=5`（greedy/beam 可切）
* **訓練參數**（24 GB 卡）

  * batch（train/eval）= (4/8) 或 (2/4)；`grad_accum=8~16`
  * LR=5e-4；Warmup=500；Epoch=3（先驗證能力，再延長）
  * `remove_unused_columns=False`（避免 Trainer 把 feature 丟掉）
  * 監控指標：Track2→**SER**；Track1→**CER**（自訂 evaluate）

### 3.4 目前成果

**Track 2（拼音）**

* 訓練：3 epoch 完成
* 開發集指標：**SER ≈ 9.06–9.07%**（`quick_ser_check.py` 與訓練日誌一致）
* 推論測試：`quick_infer_one.py` 能正確吐出音節序列
* 已可產生 **熱身賽 `Level-Up_拼音.csv`**（錄製/媒體一起跑）

**Track 1（漢字）**

* 訓練：3 epoch 完成；**training loss 收斂到 \~0.028**
* 目前 `trainer_state.json` 未記 eval（CER）——訓練流程 OK，待補評估/推論
* 檔案夾 `exp_track1_whisper_large_lora/` 內含：

  * `adapter_model.safetensors`（LoRA 權重）
  * tokenizer / preprocessor 設定
  * checkpoints（1503 / 2253）

---

## 4) 推論與提交

### 4.1 熱身賽資料

```
FSR-2025-Hakka-evaluation/
├── 熱身賽_錄製語料_大埔腔_4H/  .../<spk>/<utt>.wav
├── 熱身賽_錄製語料_詔安腔_4H/  .../<spk>/<utt>.wav
├── 熱身賽_媒體語料_大埔腔_1H/  .../<rand>.wav
└── 熱身賽_媒體語料_詔安腔_1H/  .../<rand>.wav
```

### 4.2 產生 CSV（命名：**Level-Up**）

* **拼音**（已就緒）

  ```bash
  (fsr2025) python3 infer_hakka_pinyin_warmup.py \
    --eval_root FSR-2025-Hakka-evaluation \
    --lora_dir  exp_track2_whisper_large_lora \
    --outfile   Level-Up_拼音.csv \
    --beams 1
  ```

* **漢字**（建議新增 `infer_hakka_hanzi_warmup.py`，邏輯與拼音版相同）

  ```bash
  (fsr2025) python3 infer_hakka_hanzi_warmup.py \
    --eval_root FSR-2025-Hakka-evaluation \
    --lora_dir  exp_track1_whisper_large_lora \
    --outfile   Level-Up_漢字.csv \
    --beams 1 \
    --strip_asterisk   # 交卷前移除「*」
  ```

> 兩個 CSV 欄位都為：
> `錄音檔檔名, 辨認結果`（無表頭時，保留純兩欄亦可；依主辦範例）

---

## 5) 訓練／評估監控

* `plot_loss.py`：從 `trainer_state.json` 匯出 `metrics.csv`、繪製 `training_loss.png`
* 建議啟用：

  * **TensorBoard**：`report_to="tensorboard"`, `logging_dir="tb_logs/track{1,2}"`
  * 或 **wandb**（offline/私有）：`report_to="wandb"`

---

## 6) 已知議題與處理

* **Whisper 預設 forced tokens / 翻譯**
  → 全部關閉（`forced_decoder_ids=None`、`suppress_tokens=[]`、`task="transcribe"`）。
* **PEFT / Transformers 相容性**
  → 版本已可正常運行（先前 `EncoderDecoderCache` 錯誤已排除）。
* **顯存**
  → LoRA + ckpt + TF32；若仍 OOM：降 batch / `num_beams=1`（只影響 eval）。
* **分數（SER/CER）**
  → 自行實作 Levenshtein；Track1/2 的正規化規則與訓練一致（避免偏差）。
* **合音 `*`**
  → Track1 訓練預設保留；**提交前移除 `*`**，確保格式與主辦一致。

---

## 7) 後續計畫（優先順序）

1. **Track1 推論腳本**（`infer_hakka_hanzi_warmup.py`）→ 產生 `Level-Up_漢字.csv`
2. **驗證 CER（dev）**：補上 `evaluation_strategy="epoch"` 並記錄 `eval_cer`
3. **解碼調參**：beam=1/5/8/10、length penalty、logprob 篩選
4. **資料擴增**（朗讀→媒體域適配）：speed perturb、SpecAugment、RIR/noise
5. **(可選) 語言模型 / Re-scoring**：N-gram/小型 Transformer LM（Track2 先做）
6. **(可選) 多腔調適配**：腔調 tag（DF/DM/ZF/ZM）作為條件化提示詞
7. **(可選) 多任務**：Track1/2 共享編碼器、分頭解碼器（parameter-efficient）

---

## 8) 一鍵重現（目前可直接執行）

```bash
# 1) 產生 manifests
python3 prepare_hakka_track2.py --root HAT-Vol2 --drop_mispronounce
python3 prepare_hakka_track1.py --root HAT-Vol2 --drop_mispronounce

# 2) 訓練
python3 train_whisper_lora_track2.py    # 得到 exp_track2_whisper_large_lora/
python3 train_whisper_lora_track1.py    # 得到 exp_track1_whisper_large_lora/

# 3) 驗證/監控
python3 quick_ser_check.py               # Track2 dev SER
python3 plot_loss.py --exp_dir exp_track1_whisper_large_lora

# 4) 熱身賽輸出
python3 infer_hakka_pinyin_warmup.py \
  --eval_root FSR-2025-Hakka-evaluation \
  --lora_dir  exp_track2_whisper_large_lora \
  --outfile   Level-Up_拼音.csv

# （Track1）建議新增 infer_hakka_hanzi_warmup.py 後執行：
# python3 infer_hakka_hanzi_warmup.py --eval_root FSR-2025-Hakka-evaluation \
#   --lora_dir exp_track1_whisper_large_lora --outfile Level-Up_漢字.csv --strip_asterisk
```