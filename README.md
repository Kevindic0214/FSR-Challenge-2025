# FSR-2025 Hakka ASR Challenge - Track 1 Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Project Goal**: Participating in the **Formosa Speech Recognition Challenge 2025 - Hakka ASR II** to develop an automatic speech recognition system for Taiwanese Hakka (Track 1: Hanzi Recognition).

## 🎯 Challenge Overview

* **Organizer**: National Yang Ming Chiao Tung University (NYCU)
* **Theme**: Taiwanese Hakka Automatic Speech Recognition
* **Background**: Taiwanese Hakka is spoken natively by about 1.5% of Taiwan's population. Through technology, we aim to help preserve this precious language.
* **Participation**: Student participant (competing for Student Awards)

### Track Details

* **Track 1 (Hanzi)**: Character Error Rate (CER) evaluation
* **Track 2 (Pinyin)**: Syllable Error Rate (SER) evaluation *(separate implementation)*

### Data

* **Training**: HAT-Vol2 dataset (~60H, Dapu/Zhao'an dialects, mono 16kHz)
* **Evaluation**: FSR-2025-Hakka-evaluation (warm-up: ~10H total)
* **Key Files**: FSR-2025-Hakka-evaluation-key (ground truth for evaluation)

## 🏗️ Project Structure

```
FSR-Challenge-2025/
├── 📊 Data Preparation
│   └── prepare_hakka_track1.py          # Generate train/dev manifests
├── 🎓 Training
│   └── train_whisper_lora_track1.py     # Whisper-large-v2 + LoRA training
├── 🔮 Inference
│   └── infer_hakka_hanzi_warmup.py      # Batch inference for evaluation
├── 📈 Evaluation
│   └── eval_track1_cer.py               # CER evaluation with diagnostics
├── 🚀 Pipeline Management
│   ├── run_track1_pipeline.py           # Complete pipeline (Python)
│   ├── run_track1_pipeline.sh           # Complete pipeline (Bash)
│   ├── requirements.txt                 # Full dependencies
│   └── requirements-minimal.txt         # Minimal dependencies
└── 📁 Data Directories
    ├── HAT-Vol2/                        # Training dataset
    ├── FSR-2025-Hakka-evaluation/       # Evaluation dataset
    └── FSR-2025-Hakka-evaluation-key/   # Ground truth labels
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
│   ├── 訓練_大埔腔30H/
│   └── 訓練_詔安腔30H/
├── FSR-2025-Hakka-evaluation/
└── FSR-2025-Hakka-evaluation-key/
```

### 3. Run Complete Pipeline

**Option A: Using Python Pipeline (Recommended)**
```bash
# Run complete pipeline (all stages)
python run_track1_pipeline.py

# Run specific stages only
python run_track1_pipeline.py --stage_start 1 --stage_end 2  # Data prep + training only

# Dry run to see what would be executed
python run_track1_pipeline.py --dry_run
```

**Option B: Using Bash Pipeline**
```bash
chmod +x run_track1_pipeline.sh
./run_track1_pipeline.sh  # Run all stages
./run_track1_pipeline.sh 1 2  # Run stages 1-2 only
```

**Option C: Manual Step-by-Step**
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

## 📋 Detailed Component Description

### 1. Data Preparation (`prepare_hakka_track1.py`)

**Features:**
- Scans and merges all `*_edit.csv` files from HAT-Vol2
- Robust text normalization (NFKC, zero-width removal, punctuation handling)
- Balanced dev speaker selection across dialect groups (DF/DM/ZF/ZM)
- Handles mispronunciation filtering and audio path validation
- Outputs reproducible train/dev splits in JSONL format

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
{"utt_id": "DF101K2001_001", "audio": "path/to/file.wav", "hanzi": "客語漢字", "text": "客語漢字", "group": "DF"}
```

### 2. Training (`train_whisper_lora_track1.py`)

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
- CER-based evaluation during training
- Compatible with 24GB GPU (RTX 4090)

### 3. Inference (`infer_hakka_hanzi_warmup.py`)

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
- Forced Chinese decoding (language="zh", task="transcribe")
- Robust audio loading and resampling
- Batch processing with progress tracking
- Optional per-utterance logging

### 4. Evaluation (`eval_track1_cer.py`)

**Evaluation Modes:**
- **Manifest mode**: Compare against JSONL reference
- **Key mode**: Compare against official CSV keys

**Metrics:**
- **CER**: Character Error Rate using Levenshtein distance
- **Exact Match**: Sentence-level exact match percentage

**Advanced Features:**
```bash
--probe_variants           # Test simplified/traditional variants
--dump_err errors.jsonl    # Export error analysis
--aligned_out aligned.csv  # Per-utterance results
--convert_hyp s2t          # Convert hypothesis for diagnosis
```

**Diagnostic Output:**
- Character distribution analysis (Han/Latin/Digit/Other)
- Per-utterance error breakdown
- Simplified/Traditional Chinese variant testing

## 🔧 Configuration & Customization

### Text Normalization

All scripts use consistent `normalize_hanzi()` function:
- Unicode NFKC normalization
- Zero-width character removal
- Punctuation standardization  
- Optional space/asterisk/punctuation removal

### Model Configuration

Default paths and settings can be customized:
```python
# In train_whisper_lora_track1.py
--base_model "openai/whisper-large-v2"    # Can use whisper-large-v3-turbo
--out_dir "runs/track1/lora_v2_r16_e3"    # Model output directory
--lora_r 16 --lora_alpha 32               # LoRA hyperparameters
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

### Current Baseline Performance
- **Model**: Whisper-large-v2 + LoRA (r=16, α=32)
- **Training**: 3 epochs on HAT-Vol2 (~27K utterances)
- **Hardware**: RTX 4090 24GB
- **Evaluation**: FSR-2025-Hakka-evaluation warm-up set

### Hardware Requirements
- **Minimum**: 8GB GPU memory (reduce batch_size to 1)
- **Recommended**: 16GB+ GPU memory  
- **Optimal**: 24GB+ GPU memory (RTX 4090/A6000/H100)

### Training Time Estimates
- **Data preparation**: ~2 minutes
- **Training (3 epochs)**: ~2-4 hours (RTX 4090)
- **Inference**: ~10 minutes (4K utterances)
- **Evaluation**: ~30 seconds

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size and increase gradient accumulation
python train_whisper_lora_track1.py --batch_size 1 --grad_accum 32
```

**2. Audio File Not Found**
```bash
# Use absolute paths or ensure relative path resolution
python prepare_hakka_track1.py --relative_audio_path
```

**3. Text Normalization Inconsistency**
```bash
# All scripts use same normalization - check parameters match between stages
--strip_asterisk    # Should be consistent in train/infer/eval
```

### Debug Mode

Enable detailed logging:
```bash
# Add debug logging to any script
python <script>.py --verbose

# Monitor training with tensorboard  
tensorboard --logdir runs/track1/lora_v2_r16_e3
```

## 🔍 Error Analysis & Diagnostics

### Detailed Error Analysis
```bash
# Generate comprehensive error breakdown
python eval_track1_cer.py \
    --key_dir FSR-2025-Hakka-evaluation-key \
    --hyp predictions.csv \
    --dump_err errors.jsonl \
    --aligned_out aligned.csv \
    --probe_variants
```

### Character-level Analysis
The evaluation script provides:
- Character distribution in predictions (Han/Latin/Digit/Other ratios)
- First character difference position for each error
- Simplified vs Traditional Chinese variant impact

### Per-utterance Results
Check `aligned.csv` for:
- Individual utterance CER scores
- Reference vs hypothesis alignment
- Length statistics and error counts

## 🚀 Advanced Usage

### Custom Model Training
```bash
# Train with different base model
python train_whisper_lora_track1.py \
    --base_model "openai/whisper-large-v3-turbo" \
    --lora_r 32 --lora_alpha 64 \
    --epochs 5 --lr 5e-5

# Resume from checkpoint
python train_whisper_lora_track1.py \
    --resume_from_checkpoint runs/track1/lora_v2_r16_e3/checkpoint-1500
```

### Batch Inference with Custom Settings  
```bash
# Use different decoding strategy
python infer_hakka_hanzi_warmup.py \
    --eval_root FSR-2025-Hakka-evaluation \
    --lora_dir runs/track1/lora_v2_r16_e3 \
    --outfile predictions.csv \
    --beams 5 --temperature 0.1 --length_penalty 1.2
```

### Multi-GPU Training
```bash
# Use accelerate for multi-GPU training
accelerate config  # Configure multi-GPU setup
accelerate launch train_whisper_lora_track1.py <args>
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
- ✅ Data preparation pipeline
- ✅ Training infrastructure  
- ✅ Inference system
- ✅ Evaluation framework
- 🚧 Performance optimization ongoing