# 🎵 EZ_RVC Project Assessment & Restructuring Plan

## 📋 Table of Contents
- [Current Project Analysis](#current-project-analysis)
- [Restructuring Candidates](#restructuring-candidates)
- [Proposed /app Architecture](#proposed-app-architecture)
- [README Development Plan](#readme-development-plan)
- [Dockerization Strategy](#dockerization-strategy)
- [Image-to-Video Integration](#image-to-video-integration)
- [Medium Article Integration](#medium-article-integration)
- [Implementation Roadmap](#implementation-roadmap)

---

## 🔍 Current Project Analysis

### **Core Components Identified:**

1. **Voice Encoders** (`/vencoder/`)
   - HuBERT-based content encoders
   - ContentVec variants (256L12, 768L12, etc.)
   - WhisperPPG for multilingual support
   - WavLM and DPHubert implementations

'!!to do later!!'
2. **Diffusion Engine** (`/diffusion/`)
   - Gaussian diffusion with multiple samplers (DDIM, PLMS, DPM-Solver, UniPC)
   - Shallow diffusion support for hybrid approach
   - Fast sampling algorithms for real-time inference

3. **Preprocessing Pipeline** (`/preprocess/`)
   - Audio slicing with intelligent silence detection
   - Speaker diarization using pyannote.audio
   - F0 extraction with multiple predictors (CREPE, PM, Harvest)
   - Volume normalization and resampling

4. **Inference Engine** (`/inference/`)
   - Complete voice conversion pipeline
   - Real-time processing capabilities
   - Multiple model support (SoVITS, diffusion-only, hybrid)
   - Speaker mixing and enhancement features

5. **Training Infrastructure** (`train.py`, `/model_dir/`)
   - Multi-GPU distributed training
   - GAN-based architecture with discriminator
   - Model checkpointing and resumption
   - TensorBoard integration

6. **Text Encoding** (`/tencoder/`)
   - Tacotron2-based text-to-speech pipeline
   - Direct text input support for voice synthesis

### **Current Strengths:**
✅ **Multi-platform support** (M1 Mac + Google Colab)
✅ **Advanced diffusion models** with fast sampling
✅ **Comprehensive preprocessing** with speaker diarization
✅ **Real-time inference** capabilities
✅ **Multiple speech encoders** for different use cases
✅ **Existing trained models** (SZA, Ted Cruz, etc.)

### **Current Issues:**
❌ **Scattered project structure** - No clear organization
❌ **Hard-coded paths** throughout codebase
❌ **Missing documentation** - No clear setup instructions
❌ **No containerization** - Difficult deployment
❌ **Image-to-video limitation** - Only mouth movement
❌ **No dependency management** - Single massive requirements.txt

---
  The encoders directory contains legitimate audio feature extraction models:
    - HuBERT variants (HubertSoft, CNHubertLarge, DPHubert)
    - ContentVec models (256L9, 256L12, 768L12 variants)
    - Whisper PPG (WhisperPPG, WhisperPPGLarge)
    - WavLM (WavLMBasePlus)
    - Supporting infrastructure (hubert/, dphubert/, wavlm/, whisper/ subdirectories),
   and contentVec ect similar model logic together?



## 🏗️ Restructuring Candidates

### **ACTUAL IMPLEMENTED STRUCTURE:**
```
/app/
├── models/
│   ├── encoders/          # Voice encoders (HuBERT, ContentVec, etc.) ✅ COMPLETED
│   ├── diffusion/         # Diffusion models 🔄 FUTURE PHASE (in /to_do/ for now)
│   ├── vocoders/          # Audio generation models ✅ COMPLETED
│   └── model_dir/         # Model architectures and modules ✅ COMPLETED
├── preprocessing/         # Audio preprocessing pipeline ✅ COMPLETED
├── inference/            # Voice conversion engine ✅ COMPLETED
├── training/             # Training orchestrator ✅ COMPLETED
├── ui/                   # Combined web and CLI interfaces ✅ FOLDER CREATED
├── utils/                # Audio utilities and helpers ✅ COMPLETED
└── configs/              # Configuration files ✅ COMPLETED
```

### **CURRENT FOCUS: SoVITS-Only Implementation**
**Phase 1 Priority**: Get SoVITS generator working perfectly before adding diffusion complexity

### **Project Root Structure:**
```
EZ_RVC/
├── app/                  # Main application code ✅ IMPLEMENTED
├── data/                 # Training data, models, results
├── config/              # External configuration files
├── scripts/             # Automation scripts
├── tests/               # Test suite
├── docs/                # Documentation
├── docker/              # Docker configurations
├── requirements/        # Dependency management
├── README.md            # Main documentation
├── pyproject.toml       # Modern Python packaging
└── .env.example         # Environment variables template
```

---

## 🏢 ACTUAL /app Architecture (IMPLEMENTED)

### **Current Implementation:**
```
EZ_RVC/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders/              # Voice encoder models ✅
│   │   │   ├── HubertSoft.py      # HuBERT speech encoder
│   │   │   ├── ContentVec*.py     # ContentVec variants
│   │   │   ├── WhisperPPG.py      # Whisper PPG encoder
│   │   │   ├── WavLMBasePlus.py   # WavLM encoder
│   │   │   ├── hubert/            # HuBERT implementation
│   │   │   ├── dphubert/          # Differentially private HuBERT
│   │   │   ├── wavlm/             # WavLM components
│   │   │   ├── whisper/           # Whisper components
│   │   │   └── todo_Text_encoder/ # TTS pipeline (LOW PRIORITY)
│   │   ├── diffusion/             # Diffusion model components 🔄 FUTURE PHASE (in /to_do/ for now)
│   │   ├── vocoders/              # Audio generation models ✅
│   │   │   ├── hifigan/           # HiFiGAN vocoder
│   │   │   ├── hifiganwithsnake/  # HiFiGAN with Snake activation
│   │   │   └── nsf_hifigan/       # NSF-HiFiGAN vocoder
│   │   ├── model_dir/             # Model architectures ✅
│   │   │   ├── modules/           # Core model modules
│   │   │   ├── nsf_hifigan/       # NSF-HiFiGAN configs
│   │   │   ├── pretrain/          # Pretrained model handling
│   │   │   └── sovits-pretrain-*/ # SoVITS pretrained models
│   │   └── models.py              # Model definitions
│   ├── preprocessing/             # Audio preprocessing ✅
│   │   ├── preprocess.py          # Main preprocessing script
│   │   ├── slicer.py              # Audio segmentation
│   │   ├── preprocess_hubert_f0.py # Feature extraction
│   │   ├── preprocess_flist_config.py # File list generation
│   │   └── filelists/             # Training/validation lists
│   ├── inference/                 # Voice conversion engine ✅
│   │   ├── inference_main.py      # Main conversion script
│   │   └── spkmix.py             # Speaker mixing
│   ├── training/                  # Training orchestrator ✅
│   │   ├── train.py              # Main training script
│   │   └── train_m1.py           # M1 Mac optimized training
│   ├── ui/                       # User interfaces ✅ FOLDER CREATED
│   │   ├── streamlit_app.py      # Web interface (TO IMPLEMENT)
│   │   ├── cli/                  # Command-line interfaces (TO IMPLEMENT)
│   │   └── components/           # Reusable UI components (TO IMPLEMENT)
│   ├── utils/                    # Utilities and helpers ✅
│   │   ├── data_utils.py         # Data loading utilities
│   │   └── utils.py              # General utilities
│   ├── configs/                  # Configuration files ✅
│   │   ├── 44k/                  # 44kHz model configs
│   │   └── configs_template/     # Template configurations
│   └── images/                   # Static assets ✅
└── (external directories as planned above)
```

---

## 📚 README Development Plan

### **Structure Based on ABSA-Drift & ResidentRAG:**

```markdown
# 🎵 EZ_RVC — Real-time Voice Conversion with Adversarial AI

[Hero Image: Voice conversion workflow diagram]

Real-time voice conversion using **So-VITS-SVC** with **HuBERT content encoding** and **Gaussian diffusion models** for high-quality singing voice synthesis and speech conversion.

## 📋 Table of Contents
- [Problem Description](#problem-description)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start-docker--recommended)
- [Voice Conversion Pipeline](#voice-conversion-pipeline)
- [Model Training](#model-training-workflow)
- [Monitoring](#monitoring--evaluation)
- [Technologies](#core-technologies)

## ❓ Problem Description

Traditional voice conversion systems struggle with **naturalness**, **identity preservation**, and **real-time performance**. EZ_RVC solves this by:

- **Content-independent conversion** using HuBERT speech representations
- **High-quality synthesis** with Gaussian diffusion models
- **Real-time processing** with optimized inference pipeline
- **Multi-speaker support** with speaker mixing capabilities

> *"This system implements a state-of-the-art voice conversion pipeline using So-VITS-SVC architecture. The system separates content from speaker identity using HuBERT encoders, applies speaker-specific conditioning, and generates high-quality audio using Gaussian diffusion models. The pipeline supports both offline training and real-time inference, enabling applications from entertainment content creation to accessibility tools."*

## 🏗️ System Architecture

[Architecture diagram showing: Audio Input → Content Encoder → Diffusion Model → Vocoder → Output]

## 🚀 Quick Start (Docker — recommended)

```bash
# 1) Clone and setup
git clone https://github.com/dvrk-dvys/EZ_RVC
cd EZ_RVC

# 2) Start the complete stack
docker-compose up -d

# 3) Access the web interface
open http://localhost:8501
```

## 🎤 Voice Conversion Pipeline

### Input: Source Audio + Target Speaker
### Output: Converted Audio with Target Voice Characteristics

[Flow diagram showing the 8-step process described in your Medium article]

## 🧠 Model Training Workflow

### Dataset Preparation
```bash
# 1. Audio preprocessing and slicing (modify paths in script)
python app/preprocess/preprocess.py

# 2. Generate training/validation file lists
python app/preprocess/preprocess_flist_config.py \
    --train_list "./dataset/filelists/train_colab_speaker.txt" \
    --val_list "./dataset/filelists/val_colab_speaker.txt" \
    --source_dir "dataset/" \
    --speech_encoder "hubertsoft"

# 3. Extract features (HuBERT, F0, spectrograms)
python app/preprocess/preprocess_hubert_f0.py \
    --f0_predictor "crepe" \
    --use_diff
```

### Training Pipeline
```bash
# Train the conversion model using JSON config
python app/training/train.py \
    -c "app/configs/44k/config_colab_speaker.json" \
    -m 44k \
    -md "dataset/44k/speaker_name"
```

### Inference
```bash
# Convert voice using trained model
python app/inference/inference_main.py \
    --input_audio "path/to/source.wav" \
    --output_audio "path/to/converted.wav" \
    --model_path "logs/44k/G_latest.pth" \
    --config_path "app/configs/44k/config_colab_speaker.json"
```

## 📊 Monitoring & Evaluation

- **Real-time Performance Metrics**: Latency, throughput, memory usage
- **Audio Quality Metrics**: MOS scores, speaker similarity, naturalness
- **Training Monitoring**: Loss curves, validation metrics, model convergence

## 🛠️ Core Technologies

### Voice Conversion & AI
- [**So-VITS-SVC**](https://github.com/svc-develop-team/so-vits-svc) - Soft voice conversion architecture
- [**HuBERT**](https://huggingface.co/facebook/hubert-large-ls960-ft) - Content encoding for speaker independence
- [**Gaussian Diffusion**](https://arxiv.org/abs/2006.11239) - High-quality audio generation
- [**PyTorch**](https://pytorch.org/) - Deep learning framework

### Audio Processing
- [**librosa**](https://librosa.org/) - Audio analysis and processing
- [**pyannote.audio**](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [**CREPE**](https://github.com/marl/crepe) - Pitch estimation
- [**soundfile**](https://python-soundfile.readthedocs.io/) - Audio I/O

### Deployment & Infrastructure
- [**Docker**](https://www.docker.com/) - Containerization
- [**Streamlit**](https://streamlit.io/) - Web interface
- [**FastAPI**](https://fastapi.tiangolo.com/) - REST API
- [**Weights & Biases**](https://wandb.ai/) - Experiment tracking
```

---

## 🐳 Dockerization Strategy

### **Multi-Stage Dockerfile:**
```dockerfile
# Base image with CUDA support for GPU acceleration
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

# Development stage
FROM base AS development
COPY requirements/development.txt .
RUN pip install --no-cache-dir -r development.txt

# Production stage
FROM base AS production
COPY app/ /app/
COPY config/ /config/
WORKDIR /app
EXPOSE 8501 8000
CMD ["streamlit", "run", "web/streamlit_app.py"]
```

### **Docker Compose Services:**
```yaml
version: '3.8'

services:
  voice-conversion:
    build:
      context: .
      target: production
    ports:
      - "8501:8501"  # Streamlit web interface
      - "8000:8000"  # FastAPI backend
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_CACHE_DIR=/app/models

  preprocessing:
    build:
      context: .
      target: development
    volumes:
      - ./data:/app/data
    command: ["python", "cli/preprocess.py"]

  training:
    build:
      context: .
      target: development
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["python", "cli/train.py"]

  model-registry:
    image: mlflow/mlflow:2.0
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: ["mlflow", "server", "--host", "0.0.0.0"]
```

---

## 🎬 Image-to-Video Integration

### **Current Issue Analysis:**
- **Problem**: Only mouth movement sync, no facial expression/head movement
- **Root Cause**: Limited facial landmark detection and animation
- **Impact**: Unnatural-looking video output for presentation/interview use

### **Modern Solutions Research:**

1. **Wav2Lip Enhanced** (Open Source)
   ```python
   # Enhanced Wav2Lip with full facial animation
   from app.integrations.video_sync import EnhancedWav2Lip

   lip_sync_engine = EnhancedWav2Lip(
       model_path="models/wav2lip_enhanced.pth",
       face_detection_model="RetinaFace",
       landmark_model="MediaPipe"
   )
   ```

2. **Real-Time Face Reenactment** (Research Implementation)
   - **First Order Motion Model** for head movement
   - **FaceSwapper** for identity preservation
   - **DualGAN** for expression transfer

3. **Commercial API Integration** (Fallback)
   - **RunwayML** face animation API
   - **DeepBrain AI** talking head synthesis
   - **D-ID** real-time face animation

### **Implementation Plan:**
```python
# app/integrations/video_sync/enhanced_lip_sync.py
class EnhancedVideoSync:
    def __init__(self):
        self.face_detector = self.load_face_detector()
        self.landmark_detector = self.load_landmark_detector()
        self.motion_model = self.load_motion_model()

    def sync_audio_to_video(self, audio_path, image_path, output_path):
        # 1. Extract audio features (mel-spectrogram, F0, energy)
        audio_features = self.extract_audio_features(audio_path)

        # 2. Detect face and landmarks in source image
        face_data = self.detect_face_and_landmarks(image_path)

        # 3. Generate facial motion sequence from audio
        motion_sequence = self.audio_to_motion(audio_features)

        # 4. Apply motion to face with natural head movement
        animated_frames = self.apply_motion_to_face(
            face_data, motion_sequence, include_head_movement=True
        )

        # 5. Render final video with synchronized audio
        self.render_video(animated_frames, audio_path, output_path)
```

---

## 📖 Medium Article Integration

### **Connecting Code to Article: "Decoding the Sound of Virality"**

Your Medium article provides excellent theoretical foundation. Here's how to integrate it:

1. **README Introduction Section:**
   ```markdown
   ## 🧠 The Science Behind Voice Conversion

   This implementation brings to life the concepts explored in
   ["Decoding the Sound of Virality: A Deep Dive into Adversarial AI for Voice Conversion"](https://medium.com/@ja.harr91/decoding-the-sound-of-virality-a-deep-dive-into-adversarial-ai-for-voice-conversion-tasks-on-m1-d60d32cfb2d4).

   The 8 key aspects of voice learned by our AI system:
   1. **Content (c)** - HuBERT soft embeddings of speech content
   2. **Fundamental Frequency (f0)** - Pitch patterns and voiced/unvoiced segments
   3. **Spectrogram (spec)** - Time-frequency representation
   4. **Raw Audio (y)** - Original waveform data
   5. **Speaker Identity (spk)** - Target voice characteristics
   6. **Segment Lengths** - Temporal dynamics of speech
   7. **Voiced/Unvoiced (uv)** - Speech phoneme classifications
   8. **Volume Dynamics** - Loudness patterns and expression
   ```

2. **Technical Documentation References:**
   ```markdown
   ## 🔬 Deep Dive: Adversarial Training Process

   Our implementation follows the GAN-based approach detailed in the Medium article:

   **Generator Network**: Learns to transform source voice characteristics
   **Discriminator Network**: Distinguishes real vs. generated audio
   **Training Convergence**: Achieved when generated samples are indistinguishable

   See `app/training/trainer.py` for the complete adversarial training loop.
   ```

3. **Code Comments Linking to Article:**
   ```python
   # app/core/voice_conversion.py
   class VoiceConverter:
       """
       Implements the So-VITS-SVC architecture described in:
       "Decoding the Sound of Virality" - https://medium.com/@ja.harr91/...

       This converter learns the 8 key voice aspects:
       - Content representation (HuBERT embeddings)
       - F0 pitch patterns
       - Spectral characteristics
       - Temporal dynamics
       - Speaker identity features
       - Volume and energy patterns
       - Voiced/unvoiced classifications
       - Raw audio relationships
       """
   ```

---

## 🚀 Implementation Roadmap

## ✅ **ALREADY COMPLETED**
- [x] 📊 **Project structure analysis** - identified core components
- [x] 🏗️ **Created /app base structure** - modern folder organization
- [x] 🗺️ **Component mapping** - vencoder → encoders, vdecoder → vocoders
- [x] 📋 **Planning document** - comprehensive roadmap created
- [x] 🔍 **Environment detection analysis** - identified 17 files needing centralization

## 🚀 **NEXT PHASE: Modularization & Dockerization**

#### 🔧 **Current Work (Preprocessing Pipeline):**
- [ ] 📦 **Modularize preprocessing** - clean imports and routing
- [ ] 🐳 **Dockerize preprocessing** - containerized local pipeline
- [ ] 🤝 **Hybrid data detection** - auto-detect raw vs processed
- [ ] 🌐 **HuggingFace Hub integration** - replace Google Drive

#### 🎯 **Phase 1 Goals:**
```
Local M2 Mac + Docker + Streamlit base app
├── 🔄 Preprocessing (local or Colab)
├── 🚀 Training (Colab with HF Hub)
├── 🎵 Inference (local M2 optimized)
└── 🎚️ Post-processing (Audacity integration)
```

---

### **M1 Mac Local Optimization Assessment - RESEARCH ONLY**

#### **Identified M1 Mac Performance Issues:**
1. **CUDA Dependencies** in requirements.txt (incompatible with M1)
2. **Memory Management** issues with large model loading
3. **MPS Backend** not fully utilized for all operations
4. **Compilation Issues** with some audio processing libraries

#### **Proposed Local-Only Solutions:**
1. **Metal Performance Shaders (MPS) Optimization**
   - Replace CUDA calls with MPS equivalents
   - Optimize tensor operations for Apple Silicon
   - Use `torch.backends.mps.is_available()` detection

2. **Docker with Apple Silicon Support**
   - Multi-arch containers (linux/arm64)
   - Optimized base images for M1 Macs
   - Unified environment across local/cloud

3. **Dependency Split Strategy**
   ```
   requirements/
   ├── base.txt           # Core dependencies
   ├── m1_mac.txt         # Apple Silicon optimized
   ├── cuda.txt          # NVIDIA GPU support
   └── colab.txt         # Google Colab specific
   ```

4. **Hybrid Execution Strategy**
   - Preprocessing: Local M1 (fast, efficient)
   - Training: Google Colab (GPU intensive)
   - Inference: Local M1 (real-time, private)

#### **Google Colab Integration Plan:**
- Maintain existing Colab notebooks as training option
- Add seamless model transfer between local/cloud
- Cloud training → Local inference workflow

---

### **Phase-by-Phase Implementation (PLANNING ONLY)**

### **Phase 1: M1 Mac Optimization & Structure (Week 1-2)**
- [ ] **Assess and fix M1 Mac compatibility issues**
- [ ] **Create optimized dependency management**
- [ ] **Set up /app directory structure**
- [ ] **Implement MPS backend optimization**
- [ ] **Create local-vs-cloud execution strategies**

---

## 🚨 Environment Detection Problems

### **Critical Issues Identified Across 17 Files:**

#### **1. Inconsistent MPS vs CUDA Detection**
- `inference/infer_tool.py:135`: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` (missing MPS)
- `app/models/encoders/HubertSoft.py:189`: `torch.device("mps" if torch.cuda.is_available() else "cpu")` (incorrectly uses cuda check for MPS)
- **Problem**: Mixed device detection logic causes M1 Mac performance issues

#### **2. Scattered Colab Detection (17 Files)**
Files with colab references that need centralization:
```
/Users/jordanharris/Code/EZ_RVC/app/inference/inference_main.py
/Users/jordanharris/Code/EZ_RVC/app/models/encoders/HubertSoft.py
/Users/jordanharris/Code/EZ_RVC/app/utils/utils.py
/Users/jordanharris/Code/EZ_RVC/app/preprocess/preprocess_hubert_f0.py
/Users/jordanharris/Code/EZ_RVC/inference/infer_tool.py
/Users/jordanharris/Code/EZ_RVC/colab_notebooks/EZ_RVC_FINAL.ipynb
/Users/jordanharris/Code/EZ_RVC/tests/test.py
/Users/jordanharris/Code/EZ_RVC/app/preprocess/preprocess_flist_config.py
/Users/jordanharris/Code/EZ_RVC/app/configs/44k/diffusion.yaml
/Users/jordanharris/Code/EZ_RVC/app/configs/44k/config_colab_sza.json
/Users/jordanharris/Code/EZ_RVC/app/configs/44k/config_colab_ow.json
/Users/jordanharris/Code/EZ_RVC/app/configs/44k/config_colab_cruz.json
/Users/jordanharris/Code/EZ_RVC/app/configs/44k/config_colab_adams.json
/Users/jordanharris/Code/EZ_RVC/app/configs/44k/config_colab.json
/Users/jordanharris/Code/EZ_RVC/colab_notebooks/sovits4_for_colab.ipynb
/Users/jordanharris/Code/EZ_RVC/colab_notebooks/EZ_Diairization.ipynb
```

#### **3. Mixed Device Assignment Patterns**
- Some files use `mps_device = torch.device("mps")` directly
- Others check availability first: `torch.backends.mps.is_available()`
- **Problem**: No unified device detection strategy

#### **4. No Centralized Environment Configuration**
- Each module handles device detection independently
- Hardcoded paths throughout codebase
- Google Drive mounting (outdated practice)

### **Proposed Centralized Solution:**

#### **Environment Detection Module (`app/core/environment.py`)**
```python
# Modern environment-aware configuration
class EnvironmentManager:
    def detect_environment(self):
        # Auto-detect: local M1, local CUDA, Colab, Docker
    def get_optimal_device(self):
        # Return best available: MPS > CUDA > CPU
    def get_storage_paths(self):
        # Environment-appropriate storage (no more Drive mounting)
    def get_model_registry(self):
        # Programmatic model downloading/caching
```

#### **🚀 Modern Storage: HuggingFace Hub + Ephemeral**

**🔄 New Colab Flow (No More Drive!):**
```python
# 1️⃣ Pull models from HF Hub
g_path = hf_hub_download("dvrk-dvys/EZ_RVC", "logs/44k/G_0.pth")

# 2️⃣ Auto-detect preprocessing need
!python app/cli/preprocess.py --input {RAW_DIR} --output {PROC_DIR}

# 3️⃣ Train on fast ephemeral storage (/content)
!python train.py -c config/colab.yaml -md {PROC_DIR}

# 4️⃣ Push final models back to HF
api.upload_file("logs/44k/G_10000.pth", repo_id="dvrk-dvys/EZ_RVC")
```

**📁 Data Contract:**
```
data/processed/<exp_name>/
├── filelists/ {train.txt, val.txt}
├── features/<speaker>/ {*.soft.pt, *.f0.npy, *.spec.pt}
├── spk_map.json
└── config/ {preprocess.yaml, train.yaml}
```

**🎯 Benefits:**
- ✅ **Fast I/O** (no Drive mounting)
- ✅ **Auto-cleanup** (ephemeral storage)
- ✅ **Reproducible** (HF Hub versioning)
- ✅ **Hybrid preprocessing** (local or Colab)

---

## 📋 **FUTURE ROADMAP**

### **🔧 Phase 2: Advanced Features**
- [ ] 📝 **Comprehensive README** (ABSA-Drift style)
- [ ] 🎬 **Video lip-sync integration**
- [ ] 🌐 **FastAPI + Streamlit web interface**

### **🚀 Phase 3: Production Ready**
- [ ] 📖 **Medium article integration**
- [ ] 📊 **Model registry + experiment tracking**
- [ ] ☁️ **Cloud deployment guides**

---

## 🎵 EZ_RVC Voice Conversion Pipeline Diagram

```
📁 RAW AUDIO INPUT
│
├── 🎤 Target Speaker Training Data (SZA songs, Ted Cruz speeches, etc.)
│   └── `/dataset_raw/[speaker_name]/` *.wav, *.m4a files
│
└── 🎯 Source Audio (Your script reading or song performance)
    └── Input file for conversion

                    ⬇️ PREPROCESSING PHASE ⬇️

🔧 STEP 1: Audio Preprocessing (`preprocess/preprocess.py`)
┌─────────────────────────────────────────────────────────────┐
│ Input: Raw audio files from `/dataset_raw/`                │
│ • Resample to 44kHz (music) or 16kHz (speech)             │
│ • Volume normalization and silence trimming                │
│ • Audio format conversion (m4a → wav)                      │
│ Output: Clean audio in `/dataset/44k/raw/`                 │
└─────────────────────────────────────────────────────────────┘
                              ⬇️

🎭 STEP 2: Speaker Diarization (`preprocess/preprocess.py`)
┌─────────────────────────────────────────────────────────────┐
│ Uses: pyannote.audio for "who spoke when" detection        │
│ • Identifies speaker segments in multi-speaker audio       │
│ • Extracts main speaker segments                           │
│ • Filters out background voices/music                      │
│ Output: Speaker-isolated segments                          │
└─────────────────────────────────────────────────────────────┘
                              ⬇️

✂️ STEP 3: Intelligent Audio Slicing (`preprocess/slicer.py`)
┌─────────────────────────────────────────────────────────────┐
│ • Slice audio into 3-20 second training segments           │
│ • Silence-based segmentation (configurable dB threshold)   │
│ • Remove segments that are too short/long                  │
│ • Smart overlap handling for natural speech flow           │
│ Output: Training-ready audio chunks                        │
└─────────────────────────────────────────────────────────────┘
                              ⬇️

📋 STEP 4: Training File List Generation (`preprocess/preprocess_flist_config.py`)
┌─────────────────────────────────────────────────────────────┐
│ **Create training/validation file lists for model:**       │
│ • Scan all processed audio chunks in `/dataset/22k/sza/`  │
│ • Generate `train.txt` with file paths for training       │
│ • Generate `val.txt` with file paths for validation       │
│ • Split data (e.g., 90% train, 10% validation)           │
│ Output: `filelists/train.txt`, `filelists/val.txt`        │
└─────────────────────────────────────────────────────────────┘
                              ⬇️

🧠 STEP 5: Feature Extraction (`preprocess/preprocess_hubert_f0.py`)
┌─────────────────────────────────────────────────────────────┐
│ **For each audio chunk in file lists, extract features:**  │
│ 1. Content (c): HuBERT soft embeddings → `.soft.pt`       │
│ 2. F0 (pitch): Fundamental frequency → `.f0.npy`          │
│ 3. Spectrogram: Time-frequency data → `.spec.pt`          │
│ 4. Raw audio (y): Original waveform                       │
│ 5. Speaker ID (spk): Target voice identity                 │
│ 6. Lengths: Segment duration patterns                     │
│ 7. UV flags: Voiced/unvoiced classification               │
│ 8. Volume: Loudness envelope → `.vol.npy`                │
│ Output: Feature files in `/dataset/44k/[speaker_name]/`    │
└─────────────────────────────────────────────────────────────┘

                    ⬇️ TRAINING PHASE ⬇️

🏋️ STEP 6: Model Training (`train.py`)
┌─────────────────────────────────────────────────────────────┐
│ Adversarial Training Loop:                                  │
│ • Generator (G): Learns to convert voice characteristics    │
│ • Discriminator (D): Distinguishes real vs fake audio      │
│ • Training until convergence (generated = indistinguishable)│
│ Uses: Multi-GPU, distributed training, TensorBoard logging │
│ Output: Trained model → `/logs/44k/G_[steps].pth`         │
└─────────────────────────────────────────────────────────────┘

                    ⬇️ INFERENCE PHASE ⬇️

🎯 STEP 7: Voice Conversion (`inference_main.py`)

📥 INPUT: Your script reading or song performance
┌─────────────────────────────────────────────────────────────┐
│ 🎤 Source Audio Processing:                                │
│ • Load your voice recording                                │
│ • Extract content features (what you said)                │
│ • Extract pitch patterns (how you said it)                │
│ • Segment into processable chunks                         │
└─────────────────────────────────────────────────────────────┘
                              ⬇️
📊 FEATURE MAPPING & CONVERSION
┌─────────────────────────────────────────────────────────────┐
│ • Keep CONTENT (your words) unchanged                      │
│ • Replace SPEAKER characteristics with target voice        │
│ • Adjust PITCH patterns to match target speaker           │
│ • Apply target VOLUME/ENERGY patterns                     │
│ • Generate target SPECTROGRAM representation              │
└─────────────────────────────────────────────────────────────┘
                              ⬇️
🎵 AUDIO SYNTHESIS
┌─────────────────────────────────────────────────────────────┐
│ Options (choose one):                                       │
│ • SoVITS Generator: Fast, traditional GAN approach        │
│ • Diffusion Model: Higher quality, slower                 │
│ • Shallow Diffusion: Hybrid approach (best quality)       │
│ Enhancement: Optional NSF-HiFiGAN for audio improvement    │
└─────────────────────────────────────────────────────────────┘
                              ⬇️

🎧 STEP 8: Audio Post-Processing (Audacity)
┌─────────────────────────────────────────────────────────────┐
│ **Manual Audio Finishing in Audacity:**                    │
│ • Trim silence from beginning/end of converted audio       │
│ • Align converted vocals with backing music/instrumental   │
│ • Sync timing - start voice at the right moment           │
│ • Level matching between converted voice and backing track │
│ • Final mix and export                                     │
│ Tools: Silence Finder, Amplify/Normalize, Time Shift Tool  │
└─────────────────────────────────────────────────────────────┘
                              ⬇️

📤 OUTPUT: Final Mixed Audio
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Final Result:                                           │
│ • Your script/song content in target speaker's voice       │
│ • Properly timed and mixed with backing music (if needed)  │
│ • Professional-quality audio output                       │
│ • Ready for TikTok, interviews, or content creation       │
│ • Saved as final mixed track                              │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Key File Execution Order

### **Training Workflow:**
1. `preprocess/preprocess.py` → Clean raw audio
2. `preprocess/slicer.py` → Slice into training chunks
3. `preprocess/preprocess_flist_config.py` → Generate train/val file lists
4. `preprocess/preprocess_hubert_f0.py` → Extract features from file lists
5. `train.py` → Train voice model using generated file lists

### **Conversion Workflow:**
1. `inference_main.py` → Convert your voice to target speaker
2. Uses `inference/infer_tool.py` internally for processing
3. **Audacity** → Manual post-processing (trim silence, sync timing, mix with backing tracks)

---

## 🗂️ NON-CORE FILES & DIRECTORIES TO MOVE TO TODO

### **❌ NOT MENTIONED IN MEDIUM ARTICLE - MOVE TO TODO:**

#### **Clustering System (Not mentioned in article):**
- `/cluster/` - **ENTIRE DIRECTORY**
  - `kmeans.py` - K-means clustering implementation
  - `train_cluster.py` - Cluster training script

#### **Experiment Tracking (Not mentioned in article):**
- `/wandb/` - **ENTIRE DIRECTORY**
  - All Weights & Biases experiment logs
  - `latest-run` and all run directories

#### **Voice Decoder (Not specifically mentioned):**
- `/vdecoder/` - **ENTIRE DIRECTORY**
  - Various vocoder implementations (HiFiGAN, NSF-HiFiGAN, etc.)
  - **Note**: May be needed for audio synthesis, verify before moving

#### **Utility/Test Files (Not mentioned):**
- `data_utils.py` - Data loading utilities
- `utils.py` - General utilities
- `spkmix.py` - Speaker mixing functionality
- `m1_test.py` - M1 Mac testing script
- `test.py` - General test file
- `train_m1.py` - M1-specific training (duplicate of train.py?)
- `gen_simple_req.py` - Requirements generator
- `gitignore.txt` - Git ignore template

#### **Development Files (Not mentioned):**
- `tacotron_waveform_plot.png` - Visualization artifact
- `README.md` - Current (to be replaced with comprehensive version)

### **⚠️ POTENTIALLY NON-CORE - VERIFY BEFORE MOVING:**

#### **Google Colab Integration (Mentioned but may be duplicates):**
- `/colab_notebooks/` - **REVIEW BEFORE MOVING**
  - `EZ_Diairization.ipynb` - Mentioned in article
  - `EZ_RVC_FINAL.ipynb` - Main Colab notebook
  - `sovits4_for_colab.ipynb` - Colab-specific implementation

#### **Data Directories (Important but not code):**
- `/dataset/` - Training data (keep but organize)
- `/dataset_raw/` - Raw audio files (keep but organize)
- `/logs/` - Training logs (can be regenerated)

### **✅ CORE FILES TO KEEP (Mentioned in article):**

#### **Voice Encoders (HuBERT, ContentVec - mentioned):**
- `/vencoder/` - **KEEP** - Core voice encoding

#### **Diffusion Models (Mentioned as part of generation):**
- `/diffusion/` - **KEEP** - Gaussian diffusion implementation

#### **Preprocessing Pipeline (Explicitly mentioned):**
- `/preprocess/` - **KEEP** - Audio preprocessing and feature extraction

#### **Inference Engine (Voice conversion pipeline):**
- `/inference/` - **KEEP** - Main conversion engine

#### **Text Encoding (Mentioned as added feature):**
- `/tencoder/` - **KEEP** - Text-to-speech functionality

#### **Training Infrastructure:**
- `/model_dir/` - **KEEP** - Model architectures
- `train.py` - **KEEP** - Main training script
- `inference_main.py` - **KEEP** - Main inference script

#### **Dependencies:**
- `requirements.txt` - **KEEP** - Dependencies (to be optimized)

### **🔧 ACTUAL WORKFLOW COMMANDS:**

Based on analysis of the codebase, here are the **actual commands** currently used:

#### **1. Preprocessing Workflow:**
```bash
# Modify paths in app/preprocess/preprocess.py, then run:
python app/preprocess/preprocess.py

# Generate file lists:
python app/preprocess/preprocess_flist_config.py \
    --train_list "./dataset/filelists/train_colab_adams.txt" \
    --val_list "./dataset/filelists/val_colab_adams.txt" \
    --source_dir "dataset/" \
    --speech_encoder "hubertsoft"

# Extract features:
python app/preprocess/preprocess_hubert_f0.py \
    --f0_predictor "crepe"
```

#### **2. Training Commands:**
```bash
# From Colab notebooks - actual working commands:
python train.py -c {config_path} -m 44k -md "/path/to/dataset/44k/speaker"
```

#### **3. Configuration Files:**
- **JSON Configs**: `app/configs/44k/config_colab_*.json` (not YAML)
- **Diffusion**: `app/configs/44k/diffusion.yaml`
- **Available speakers**: adams, cruz, sza, ow (from config file names)

#### **4. Hardcoded Paths Found:**
- `input_dir = './dataset_raw/eric_adams/'` (preprocess.py:294)
- `config_path = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/dataset/configs/config.json"` (train.py:40)
- `hps = utils.get_hparams_from_file("./dataset/configs/config_colab.json")` (preprocess_hubert_f0.py:24)

### **📁 UPDATED FILE STRUCTURE - USER CHANGES IMPLEMENTED:**

**User has implemented /app structure with following changes:**
```
EZ_RVC/
├── app/
│   ├── encoders/          # Voice encoders (moved from /vencoder/)
│   ├── vocoders/          # Audio synthesis (moved from /vdecoder/)
│   ├── preprocessing/     # Audio preprocessing (moved from /preprocess/)
│   ├── training/          # Model training logic
│   ├── inference/         # Voice conversion engine
│   ├── models/           # Model architectures
│   ├── cli/              # Command-line interfaces
│   ├── web/              # Web interfaces
│   └── utils/            # Utilities and helpers
├── notebooks/            # Jupyter notebooks (outside /app/)
│   ├── colab/           # Google Colab notebooks
│   └── research/        # Local experimentation
├── data/                # Training data, models, results
├── config/              # Configuration files
├── scripts/             # Shell scripts, automation
├── tests/               # Test suite
└── requirements/        # Split dependency files
```

### **🚀 MODERNIZATION ATTACK PLAN - UPDATED:**

#### **Phase 1: Complete App Migration (High Priority)**
- [ ] **Finish moving remaining core components** to proper /app/ locations
- [ ] **Implement proper CLI interfaces** replacing hardcoded scripts
- [ ] **Create config management system** with YAML-based configuration
- [ ] **Fix all import paths** after restructuring
- [ ] **Update file list generation** to work with new structure

#### **Phase 2: Docker Infrastructure (Medium Priority)**
- [ ] **Multi-service containerization** (app, jupyter, data)
- [ ] **Separate notebook environment** from production app
- [ ] **Environment-specific containers** (local, colab, production)
- [ ] **Volume mounting strategy** for data and models

#### **Phase 3: M1 Mac Optimization (High Priority)**
- [ ] **Split requirements** for M1 vs CUDA vs Colab environments
- [ ] **MPS backend optimization** for Apple Silicon
- [ ] **Local-vs-cloud execution strategies** (preprocessing local, training cloud)
- [ ] **Memory optimization** for M1 constraints

#### **Phase 4: Enhanced Features (Lower Priority)**
- [ ] **Improved video sync integration** addressing mouth-only limitation
- [ ] **Text-to-speech pipeline refinement**
- [ ] **Model registry and versioning system**
- [ ] **Comprehensive test suite**

### **📋 CRITICAL MISSING COMPONENT IDENTIFIED:**
**File List Generation** - The missing step between slicing and feature extraction:
- `preprocess_flist_config.py` generates `train.txt` and `val.txt`
- Lists all processed audio chunks for training
- Essential for model to know which files to process
- **Must be included in any modernized preprocessing pipeline**

### **🔍 CRITICAL PATH MANAGEMENT ISSUES IDENTIFIED:**

#### **Environment Detection Problems:**
- **Colab detection scattered throughout codebase** - Multiple files contain `is_running_in_colab()` checks
- **No centralized environment management** - Each file handles environment detection independently
- **Mixed local/cloud path handling** - Inconsistent path resolution across different environments
- **Hard-coded path switching** - Manual path changes required when moving between local/colab/docker

**Impact:** Makes deployment across environments fragile and error-prone. Currently blocking clean M1 local development and smooth colab integration.

### **🎯 USER'S MODERNIZATION PLAN OF ATTACK:**

#### **Phase 1: Local M1 Mac Foundation**
**Goal:** Get base local version running on M2 Mac with truncated dataset in new /app configuration
- [ ] **Docker + Docker Compose setup** for consistent local environment
- [ ] **Basic Streamlit interface** for easy local testing and development
- [ ] **M1/M2 optimized dependencies** with MPS backend support
- [ ] **Truncated dataset workflow** for faster local iteration and testing
- [ ] **Centralized environment detection** replacing scattered colab checks

#### **Phase 2: Google Colab Integration**
**Goal:** Write updated EZ_RVC_FINAL notebook cloning /app functionality for cloud GPU usage
- [ ] **Bare-bones project extraction** from /app structure (no Docker, no Streamlit)
- [ ] **Colab-optimized notebook** with accelerated GPU support and payment integration
- [ ] **Seamless model transfer** between local development and cloud training
- [ ] **Environment-specific path handling** unified with local version

#### **Phase 3: Documentation & Replication**
**Goal:** Finalize with comprehensive README, screenshots, and replication guides
- [ ] **Comprehensive README.md** with clear setup instructions for both environments
- [ ] **Screenshot documentation** showing complete workflows
- [ ] **Local dockerized version guide** - one-command setup and usage
- [ ] **Google Colab notebook guide** - cloud training and inference walkthrough
- [ ] **Troubleshooting section** covering common M1 Mac and Colab issues

### **⚠️ IMPLEMENTATION STATUS:**
- ✅ **User has restructured to /app/** - Major progress made
- ✅ **Speaker-specific configs moved** to `/app/configs/44k/` structure
- 🔄 **Currently identifying path management issues** and planning environment detection overhaul
- 📋 **Ready for Phase 1:** M1 Mac + Docker + Streamlit foundation
- 🎯 **Next step:** Implement centralized environment management before Docker setup

---

## 🎯 Success Metrics

### **Technical Goals:**
- ✅ **Clean, maintainable codebase** with modern Python structure
- ✅ **One-command setup** via Docker Compose
- ✅ **Comprehensive documentation** linking theory to implementation
- ✅ **Enhanced video sync** with natural facial animation
- ✅ **Production-ready deployment** with monitoring and logging

### **User Experience Goals:**
- ✅ **Simple web interface** for non-technical users
- ✅ **Real-time processing** for live applications
- ✅ **Batch processing** capabilities for content creation
- ✅ **High-quality output** suitable for professional use
- ✅ **Clear connection** between code implementation and research article

This restructured project will transform EZ_RVC from a research prototype into a production-ready voice conversion platform, suitable for both technical experimentation and real-world applications like interview preparation and content creation.
