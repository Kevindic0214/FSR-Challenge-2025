# FSR-2025 Hakka ASR Competition Project

## Project Goal

Participating in the **Formosa Speech Recognition Challenge 2025 - Hakka ASR II** to develop an automatic speech recognition (ASR) system for Taiwanese Hakka.

## About the Challenge

* **Organizer**: National Yang Ming Chiao Tung University (NYCU)
* **Theme**: Taiwanese Hakka Automatic Speech Recognition
* **Background**: Taiwanese Hakka is spoken natively by about 1.5% of Taiwan's population. Through technology, we aim to help preserve this precious language.
* **Participation**: Student participant (competing for Student Awards)

## Project Status

### Current Progress (as of 2025-08-24)

* **Track 1 (Hanzi)**

  * Model: Whisper-large-v2 + LoRA
  * Warm-up Evaluation: **CER = 26.49%**, Exact-Match = 49.7% (4299 utts)
  * Reference implementation of CER evaluator completed (`eval_track1_cer.py`).

* **Track 2 (Pinyin)**

  * Model: Whisper-large-v2 + LoRA
  * Warm-up Evaluation: **SER = 25.69%**, Exact-Match = 18.4% (4299 utts)
  * Tokenization and scoring policies locked (drop `*token` in reference).

* **Baseline Comparison**

  * Official baseline (B): **Whisper-large-v3-turbo**
  * Reported scores: Track1 CER ≈ **10.42%**, Track2 SER ≈ **23.40%**.
  * Plan: Run zero-shot decoding with `whisper-large-v3-turbo` for A/B comparison.

### Repository Structure

```
FSR-Challenge-2025/
├── training/                # LoRA training scripts
├── inference/               # Decoding scripts (Track1/2)
├── evaluation/              # SER/CER evaluators
├── experiments/             # v2+LoRA outputs (archived)
├── experiments_v3t/         # new branch for v3-turbo
├── submissions/             # CSV submissions (Pilot/Final)
└── docs/                    # Notes, snapshots, results
```

### Next Steps

1. **Run zero-shot v3-turbo decoding** for both tracks (beams = 1/5/8, `max_new_tokens >= 256`).
2. Compare results with v2+LoRA; update `RESULTS.md`.
3. Optional: retrain LoRA on top of v3-turbo if needed.
4. Explore domain-specific augmentation (speed perturb, noise, RIR) for robustness.
5. Prepare Pilot submission (`NYCU_Level-Up_pinyin.csv`, `NYCU_Level-Up_hanzi.csv`).

## Important Links

* [Official Website](https://sites.google.com/nycu.edu.tw/fsw/home/challenge-2025?authuser=0)
* [Whisper Hakka Baseline Repo](https://github.com/Speech-AI-Research-Center/whisper-hakka)

---

*Working towards preserving Taiwanese Hakka culture* 💪
