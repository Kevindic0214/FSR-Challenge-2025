# FSR-2025 Hakka ASR Challenge - Complete Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Project Goal**: Participating in the **Formosa Speech Recognition Challenge 2025 - Hakka ASR II** to develop an automatic speech recognition system for Taiwanese Hakka with complete implementations for both tracks.

## 🎯 Challenge Overview

* **Organizer**: National Yang Ming Chiao Tung University (NYCU)
* **Theme**: Taiwanese Hakka Automatic Speech Recognition
* **Background**: Taiwanese Hakka is spoken natively by about 1.5% of Taiwan's population. Through technology, we aim to help preserve this precious language.
* **Participation**: Student participant (competing for Student Awards)

### Track Details

* **Track 1 (Hanzi)**: Character Error Rate (CER) evaluation - Chinese character recognition
* **Track 2 (Pinyin)**: Syllable Error Rate (SER) evaluation - Romanized phonetic transcription

Both tracks are fully implemented in this repository with dedicated training, inference, and evaluation pipelines.

### Data

* **Training**: HAT-Vol2 dataset (~60H, Dapu/Zhao'an dialects, mono 16kHz)
* **Evaluation**: FSR-2025-Hakka-evaluation (warm-up: ~10H total)
* **Key Files**: FSR-2025-Hakka-evaluation-key (ground truth for evaluation)

## 🏗️ Project Structure

```
FSR-Challenge-2025/
├── 📊 Data Preparation
│   ├── prepare_hakka_track1.py          # Generate train/dev manifests for hanzi
│   └── prepare_hakka_track2.py          # Generate train/dev manifests for pinyin
├── 🎓 Training
│   ├── train_whisper_lora_track1.py     # Whisper-large-v2 + LoRA training (hanzi)
│   └── train_whisper_lora_track2.py     # Whisper-large-v2 + LoRA training (pinyin)
├── 🔮 Inference
│   ├── infer_hakka_hanzi_warmup.py      # Batch inference for hanzi evaluation
│   ├── infer_hakka_pinyin_warmup.py     # Batch inference for pinyin evaluation
│   ├── infer_whisper_track2.py          # Alternative inference for track 2
│   └── quick_infer_one.py               # Quick single-file inference
├── 📈 Evaluation
│   ├── eval_track1_cer.py               # CER evaluation with diagnostics
│   ├── eval_track2_ser.py               # SER evaluation for pinyin
│   └── quick_ser_check.py               # Quick SER validation
├── �️ Utilities
│   ├── make_keyonly_track2.py           # Generate key-only files for track 2
│   └── plot_loss.py                     # Training loss visualization
├── �🚀 Pipeline Management
│   ├── run_track1_pipeline.py           # Complete track 1 pipeline
│   └── run_track1_pipeline.sh           # Complete track 1 pipeline (bash)
├── 📁 Data Directories
│   ├── HAT-Vol2/                        # Training dataset (~60H)
│   │   ├── manifests_track1/            # Track 1 train/dev manifests
│   │   ├── manifests_track2/            # Track 2 train/dev manifests
│   │   ├── 訓練_大埔腔30H/              # Dapu dialect training audio
│   │   ├── 訓練_詔安腔30H/              # Zhao'an dialect training audio
│   │   └── *.csv                        # Training metadata files
│   ├── FSR-2025-Hakka-evaluation/       # Evaluation dataset (~10H)
│   │   ├── 熱身賽_媒體語料_大埔腔_1H/   # Dapu media corpus
│   │   ├── 熱身賽_媒體語料_詔安腔_1H/   # Zhao'an media corpus
│   │   ├── 熱身賽_錄製語料_大埔腔_4H/   # Dapu recorded corpus
│   │   └── 熱身賽_錄製語料_詔安腔_4H/   # Zhao'an recorded corpus
│   └── FSR-2025-Hakka-evaluation-key/   # Ground truth labels
├── 📊 Experiments & Results
│   ├── experiments_v2/                  # Version 2 experiment results
│   ├── experiments_v3t/                 # Version 3t experiment results
│   └── archive/                         # Previous experiments and submissions
└── 📝 Documentation
    ├── README.md                        # This file
    └── *.md                            # Additional documentation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository and navigate to it
git clone <your-repo-url>
cd FSR-Challenge-2025

# Install dependencies
pip install -r requirements-minimal.txt

# Or install full dependencies (with optional features)
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your data structure looks like this:
```
FSR-Challenge-2025/
├── HAT-Vol2/
│   ├── 訓練_DF_大埔腔_女_edit.csv
│   ├── 訓練_DM_大埔腔_男_edit.csv  
│   ├── 訓練_ZF_詔安腔_女_edit.csv
│   ├── 訓練_ZM_詔安腔_男_edit.csv
│   ├── manifests_track1/            # Generated by prepare_hakka_track1.py
│   ├── manifests_track2/            # Generated by prepare_hakka_track2.py
│   ├── 訓練_大埔腔30H/
│   └── 訓練_詔安腔30H/
├── FSR-2025-Hakka-evaluation/
│   ├── 熱身賽_媒體語料_大埔腔_1H/
│   ├── 熱身賽_媒體語料_詔安腔_1H/
│   ├── 熱身賽_錄製語料_大埔腔_4H/
│   └── 熱身賽_錄製語料_詔安腔_4H/
└── FSR-2025-Hakka-evaluation-key/
    ├── 熱身賽_媒體語料_大埔腔_edit.csv
    ├── 熱身賽_媒體語料_詔安腔_edit.csv
    ├── 熱身賽_錄製語料_DF_大埔腔_女_edit.csv
    ├── 熱身賽_錄製語料_DM_大埔腔_男_edit.csv
    ├── 熱身賽_錄製語料_ZF_詔安腔_女_edit.csv
    └── 熱身賽_錄製語料_ZM_詔安腔_男_edit.csv
```

### 3. Run Complete Pipeline

**Track 1 (Hanzi Recognition)**

*Option A: Using Python Pipeline (Recommended)*
```bash
# Run complete track 1 pipeline (all stages)
python run_track1_pipeline.py

# Run specific stages only
python run_track1_pipeline.py --stage_start 1 --stage_end 2  # Data prep + training only

# Dry run to see what would be executed
python run_track1_pipeline.py --dry_run
```

*Option B: Using Bash Pipeline*
```bash
chmod +x run_track1_pipeline.sh
./run_track1_pipeline.sh  # Run all stages
./run_track1_pipeline.sh 1 2  # Run stages 1-2 only
```

*Option C: Manual Step-by-Step (Track 1)*
```bash
# Stage 1: Data Preparation
python prepare_hakka_track1.py --root HAT-Vol2 --drop_mispronounce

# Stage 2: Training  
python train_whisper_lora_track1.py --train_jsonl HAT-Vol2/manifests_track1/train.jsonl

# Stage 3: Inference
python infer_hakka_hanzi_warmup.py --eval_root FSR-2025-Hakka-evaluation \
    --model openai/whisper-large-v2 --lora_dir runs/track1/lora_v2_r16_e3 \
    --outfile predictions.csv

# Stage 4: Evaluation
python eval_track1_cer.py --key_dir FSR-2025-Hakka-evaluation-key \
    --hyp predictions.csv
```

**Track 2 (Pinyin Recognition)**

*Manual Step-by-Step (Track 2)*
```bash
# Stage 1: Data Preparation
python prepare_hakka_track2.py --root HAT-Vol2 --drop_mispronounce

# Stage 2: Training  
python train_whisper_lora_track2.py --train_jsonl HAT-Vol2/manifests_track2/train.jsonl

# Stage 3: Inference
python infer_hakka_pinyin_warmup.py --eval_root FSR-2025-Hakka-evaluation \
    --model openai/whisper-large-v2 --lora_dir runs/track2/lora_v2_r16_e3 \
    --outfile predictions_pinyin.csv

# Alternative inference method
python infer_whisper_track2.py --eval_root FSR-2025-Hakka-evaluation \
    --lora_dir runs/track2/lora_v2_r16_e3 --outfile predictions_pinyin.csv

# Stage 4: Evaluation
python eval_track2_ser.py --key_dir FSR-2025-Hakka-evaluation-key \
    --hyp predictions_pinyin.csv

# Quick SER check
python quick_ser_check.py --hyp predictions_pinyin.csv --ref reference.csv
```

**Utility Scripts**
```bash
# Generate key-only files for track 2
python make_keyonly_track2.py --key_dir FSR-2025-Hakka-evaluation-key

# Quick inference for a single file
python quick_infer_one.py --audio_file path/to/audio.wav --model_dir runs/track1/model

# Plot training loss
python plot_loss.py --log_dir runs/track1/lora_v2_r16_e3
```

## 📋 Detailed Component Description

### 1. Data Preparation

**Track 1: `prepare_hakka_track1.py`** - Hanzi Recognition Data
**Track 2: `prepare_hakka_track2.py`** - Pinyin Recognition Data

**Features:**
- Scans and merges all `*_edit.csv` files from HAT-Vol2
- Robust text normalization (NFKC, zero-width removal, punctuation handling)
- Balanced dev speaker selection across dialect groups (DF/DM/ZF/ZM)
- Handles mispronunciation filtering and audio path validation
- Outputs reproducible train/dev splits in JSONL format
- Track-specific text processing (hanzi vs pinyin)

**Key Parameters:**
```bash
--root HAT-Vol2                 # Dataset root directory  
--dev_speakers 12               # Number of dev speakers (balanced)
--drop_mispronounce            # Filter mispronounced samples
--strip_asterisk               # Remove co-articulation markers
--relative_audio_path          # Use relative paths (recommended)
```

**Output Format:**
```json
# Track 1 (Hanzi)
{"utt_id": "DF101K2001_001", "audio": "path/to/file.wav", "hanzi": "客語漢字", "text": "客語漢字", "group": "DF"}

# Track 2 (Pinyin)  
{"utt_id": "DF101K2001_001", "audio": "path/to/file.wav", "pinyin": "hag5 ngi3", "text": "hag5 ngi3", "group": "DF"}
```

### 2. Training

**Track 1: `train_whisper_lora_track1.py`** - Hanzi Recognition Training
**Track 2: `train_whisper_lora_track2.py`** - Pinyin Recognition Training

**Model Architecture:**
- Base model: `openai/whisper-large-v2`
- Fine-tuning: LoRA (Low-Rank Adaptation)
- Memory optimization: gradient checkpointing, bfloat16, TF32

**Training Configuration:**
```python
# LoRA settings
lora_r = 16
lora_alpha = 32  
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

# Training hyperparameters
batch_size = 2          # Per device
grad_accum = 16         # Gradient accumulation steps  
learning_rate = 1e-4
epochs = 3
warmup_steps = 500
```

**Features:**
- Automatic dtype detection (bfloat16 > float16 > float32)
- Chinese transcription enforcement (prevents language drift)
- CER-based evaluation during training (Track 1) / SER-based evaluation (Track 2)
- Compatible with 24GB GPU (RTX 4090)
- Track-specific text processing and evaluation metrics

### 3. Inference

**Track 1: `infer_hakka_hanzi_warmup.py`** - Hanzi Recognition Inference
**Track 2: `infer_hakka_pinyin_warmup.py`** - Pinyin Recognition Inference
**Alternative: `infer_whisper_track2.py`** - Alternative Track 2 Inference
**Utility: `quick_infer_one.py`** - Single File Quick Inference

**Input Modes:**
- Directory scanning: Recursively finds `*.wav` files
- JSONL manifest: Reads structured evaluation data
- Auto-detection based on input path

**Decoding Options:**
```bash
--beams 1                    # Beam search size (1=greedy)
--temperature 0.0            # Sampling temperature  
--max_new_tokens 256         # Maximum decode length
--strip_asterisk             # Remove * markers in output
--key_csv_filter             # Filter by official key file
```

**Features:**
- Forced Chinese decoding (language="zh", task="transcribe") for Track 1
- Language-appropriate decoding for Track 2 (pinyin output)
- Robust audio loading and resampling
- Batch processing with progress tracking
- Optional per-utterance logging
- Track-specific output formatting

### 4. Evaluation

**Track 1: `eval_track1_cer.py`** - Character Error Rate Evaluation
**Track 2: `eval_track2_ser.py`** - Syllable Error Rate Evaluation  
**Utility: `quick_ser_check.py`** - Quick SER Validation

**Evaluation Modes:**
- **Manifest mode**: Compare against JSONL reference
- **Key mode**: Compare against official CSV keys

**Metrics:**
- **CER** (Track 1): Character Error Rate using Levenshtein distance
- **SER** (Track 2): Syllable Error Rate for pinyin transcription
- **Exact Match**: Sentence-level exact match percentage for both tracks

**Advanced Features:**
```bash
# Track 1 specific
--probe_variants           # Test simplified/traditional variants
--convert_hyp s2t          # Convert hypothesis for diagnosis

# Both tracks
--dump_err errors.jsonl    # Export error analysis
--aligned_out aligned.csv  # Per-utterance results
```

**Diagnostic Output:**
- Character/syllable distribution analysis
- Per-utterance error breakdown
- Simplified/Traditional Chinese variant testing (Track 1)
- Pinyin syllable accuracy analysis (Track 2)

### 5. Utilities

**`make_keyonly_track2.py`** - Generate key-only files for Track 2 submissions
**`plot_loss.py`** - Visualize training loss curves from tensorboard logs
**`quick_infer_one.py`** - Quick inference for single audio files
**`quick_ser_check.py`** - Quick SER validation for Track 2

## 🔧 Configuration & Customization

### Text Normalization

All scripts use consistent normalization functions:
- `normalize_hanzi()` for Track 1: Unicode NFKC normalization, zero-width character removal, punctuation standardization
- `normalize_pinyin()` for Track 2: Pinyin-specific normalization, tone marker handling
- Optional space/asterisk/punctuation removal for both tracks

### Model Configuration

Default paths and settings can be customized:
```python
# Track 1 Training
python train_whisper_lora_track1.py \
    --base_model "openai/whisper-large-v2" \
    --out_dir "runs/track1/lora_v2_r16_e3" \
    --lora_r 16 --lora_alpha 32

# Track 2 Training  
python train_whisper_lora_track2.py \
    --base_model "openai/whisper-large-v2" \
    --out_dir "runs/track2/lora_v2_r16_e3" \
    --lora_r 16 --lora_alpha 32
```

### Pipeline Configuration

Create custom config for `run_track1_pipeline.py`:
```json
{
  "root_dir": "HAT-Vol2",
  "model_dir": "runs/track1/custom_model", 
  "training": {
    "epochs": 5,
    "batch_size": 4,
    "lr": 5e-5
  },
  "inference": {
    "beams": 5,
    "temperature": 0.1
  }
}
```

## 📊 Performance & Results

### Current Implementation Status
- **Track 1 (Hanzi)**: ✅ Complete pipeline with CER evaluation
- **Track 2 (Pinyin)**: ✅ Complete pipeline with SER evaluation
- **Model**: Whisper-large-v2 + LoRA (r=16, α=32) for both tracks
- **Training**: 3 epochs on HAT-Vol2 (~27K utterances)
- **Hardware**: RTX 4090 24GB
- **Evaluation**: FSR-2025-Hakka-evaluation warm-up set

### Experiment Versions
- **experiments_v2/**: Version 2 results with beam search comparisons
- **experiments_v3t/**: Version 3t results with temperature tuning
- **archive/v2_lora/**: Archived submissions and detailed evaluation results

### Hardware Requirements
- **Minimum**: 8GB GPU memory (reduce batch_size to 1)
- **Recommended**: 16GB+ GPU memory  
- **Optimal**: 24GB+ GPU memory (RTX 4090/A6000/H100)

### Training Time Estimates
- **Data preparation**: ~2 minutes (both tracks)
- **Training (3 epochs)**: ~2-4 hours per track (RTX 4090)
- **Inference**: ~10 minutes (4K utterances)
- **Evaluation**: ~30 seconds

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size and increase gradient accumulation
python train_whisper_lora_track1.py --batch_size 1 --grad_accum 32
python train_whisper_lora_track2.py --batch_size 1 --grad_accum 32
```

**2. Audio File Not Found**
```bash
# Use absolute paths or ensure relative path resolution
python prepare_hakka_track1.py --relative_audio_path
python prepare_hakka_track2.py --relative_audio_path
```

**3. Text Normalization Inconsistency**
```bash
# All scripts use same normalization - check parameters match between stages
--strip_asterisk    # Should be consistent in train/infer/eval for both tracks
```

**4. Track 2 SER Evaluation Issues**
```bash
# Use quick SER check for debugging
python quick_ser_check.py --hyp predictions.csv --ref reference.csv

# Generate key-only files if needed
python make_keyonly_track2.py --key_dir FSR-2025-Hakka-evaluation-key
```

### Debug Mode

Enable detailed logging:
```bash
# Add debug logging to any script
python <script>.py --verbose

# Monitor training with tensorboard (both tracks)
tensorboard --logdir runs/track1/lora_v2_r16_e3
tensorboard --logdir runs/track2/lora_v2_r16_e3

# Plot training loss
python plot_loss.py --log_dir runs/track1/lora_v2_r16_e3
```

## 🔍 Error Analysis & Diagnostics

### Detailed Error Analysis
```bash
# Track 1: Generate comprehensive CER error breakdown
python eval_track1_cer.py \
    --key_dir FSR-2025-Hakka-evaluation-key \
    --hyp predictions.csv \
    --dump_err errors.jsonl \
    --aligned_out aligned.csv \
    --probe_variants

# Track 2: Generate comprehensive SER error breakdown
python eval_track2_ser.py \
    --key_dir FSR-2025-Hakka-evaluation-key \
    --hyp predictions_pinyin.csv \
    --dump_err errors_ser.jsonl \
    --aligned_out aligned_ser.csv
```

### Character-level Analysis (Track 1)
The evaluation script provides:
- Character distribution in predictions (Han/Latin/Digit/Other ratios)
- First character difference position for each error
- Simplified vs Traditional Chinese variant impact

### Syllable-level Analysis (Track 2)  
The evaluation script provides:
- Syllable boundary detection accuracy
- Tone marker prediction accuracy
- Phoneme-level error patterns

### Per-utterance Results
Check `aligned.csv` (Track 1) or `aligned_ser.csv` (Track 2) for:
- Individual utterance CER/SER scores
- Reference vs hypothesis alignment
- Length statistics and error counts
- Track-specific metrics and analysis

## 🚀 Advanced Usage

### Custom Model Training
```bash
# Train Track 1 with different base model
python train_whisper_lora_track1.py \
    --base_model "openai/whisper-large-v3-turbo" \
    --lora_r 32 --lora_alpha 64 \
    --epochs 5 --lr 5e-5

# Train Track 2 with custom settings
python train_whisper_lora_track2.py \
    --base_model "openai/whisper-large-v3-turbo" \
    --lora_r 32 --lora_alpha 64 \
    --epochs 5 --lr 5e-5

# Resume from checkpoint
python train_whisper_lora_track1.py \
    --resume_from_checkpoint runs/track1/lora_v2_r16_e3/checkpoint-1500
```

### Batch Inference with Custom Settings  
```bash
# Track 1: Use different decoding strategy
python infer_hakka_hanzi_warmup.py \
    --eval_root FSR-2025-Hakka-evaluation \
    --lora_dir runs/track1/lora_v2_r16_e3 \
    --outfile predictions.csv \
    --beams 5 --temperature 0.1 --length_penalty 1.2

# Track 2: Use alternative inference method
python infer_whisper_track2.py \
    --eval_root FSR-2025-Hakka-evaluation \
    --lora_dir runs/track2/lora_v2_r16_e3 \
    --outfile predictions_pinyin.csv \
    --beams 3 --temperature 0.0
```

### Multi-GPU Training
```bash
# Use accelerate for multi-GPU training (both tracks)
accelerate config  # Configure multi-GPU setup
accelerate launch train_whisper_lora_track1.py <args>
accelerate launch train_whisper_lora_track2.py <args>
```

## 📚 Dependencies

### Core Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+ (with CUDA support recommended)
- **Transformers**: 4.30+
- **PEFT**: 0.4+ (for LoRA)
- **torchaudio**: 2.0+

### Optional Dependencies  
- **opencc-python-reimplemented**: Traditional/Simplified Chinese conversion
- **tensorboard**: Training visualization
- **accelerate**: Multi-GPU training support

## 🤝 Contributing

This is a competition project, but contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NYCU Speech Processing Lab** for organizing the FSR-2025 challenge
- **OpenAI** for the Whisper models
- **Hugging Face** for Transformers and PEFT libraries
- **HAT-Vol2** dataset creators for providing Hakka speech data

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing issues in the repository
3. Create a new issue with detailed information

---

*Working towards preserving Taiwanese Hakka culture through technology* 💪

**Competition Status**: 
- ✅ Track 1: Complete hanzi recognition pipeline
- ✅ Track 2: Complete pinyin recognition pipeline  
- ✅ Data preparation for both tracks
- ✅ Training infrastructure for both tracks
- ✅ Inference systems for both tracks
- ✅ Evaluation frameworks (CER & SER)
- ✅ Error analysis and diagnostics
- 🚧 Performance optimization ongoing
- 📊 Multiple experiment versions available