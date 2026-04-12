# 🏺 YaYan-AI (雅言) — Cross-Architecture Dialect Intelligence

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20(Edge%2FServer)-purple.svg)]()
[![Whisper](https://img.shields.io/badge/ASR-Whisper--Large--v3-orange.svg)](https://github.com/openai/whisper)
[![LLM](https://img.shields.io/badge/LLM-Llama--3.1--8B%20%7C%20Qwen--2.5--7B-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Deployment-brightgreen.svg)]()

> **Scalable Local Dialect Intelligence System**
> 從工作站到伺服器的全本地化方言情報系統

---

## 📖 Introduction

**[English]**
**YaYan-AI** is a privacy-first, fully offline AI system that converts dialectal speech
(e.g., Taiwanese Hokkien, Cantonese, Uyghur) into structured Traditional Chinese intelligence reports.

Featuring a **cross-architecture design**, it runs seamlessly on both consumer-grade workstations
(RTX 3090) and enterprise-grade dual-GPU servers (Dual RTX 4000 Ada), covering everything from
rapid prototyping to large-scale batch intelligence analysis.

**[中文]**
**雅言 (YaYan-AI)** 是一套基於本地化部署的 AI 情報系統，致力於將多種方言
（如台灣口語、粵語、維吾爾語）轉化為標準的「雅言」（正體中文情報摘要）。

本專案採用**跨架構設計**，同時支援單卡工作站（RTX 3090）與企業級伺服器（Dual RTX 4000 Ada），
實現從原型開發到大規模情報分析的無縫遷移。

---

## 🌟 Architecture & Versions

| Feature | **v1: Workstation Edition** | **v2: Server Edition** |
| :--- | :--- | :--- |
| **Use Case** | Prototyping / Edge Inference | Massive Batch Processing |
| **GPU Config** | 1× NVIDIA RTX 3090 (24GB) | **2× NVIDIA RTX 4000 Ada** (20GB × 2) |
| **Strategy** | Serial Processing | **Pipeline Parallelism** |
| **ASR Model** | Whisper-Large-v3 | Whisper-Large-v3 (GPU 0) |
| **LLM Model** | Qwen-2.5-7B (4-bit NF4) | **Meta-Llama-3.1-8B** (GPU 1) |
| **Storage** | Local SSD | RAID 10 NVMe Array (`/data`) |
| **OS** | Windows 10/11 (WSL2) | **Ubuntu Server 24.04 LTS** |
| **Interface** | Gradio WebUI | Gradio WebUI + Batch CLI |

---

## 🚀 Key Features

**🎙️ Military-Grade ASR**
- Deploys `whisper-large-v3` locally, optimized for PSTN/VoIP acoustic environments
- 本地部署最新 Whisper 模型，針對電話錄音優化，精準捕捉方言發音

**🧠 Strategic Intelligence Analysis**
- Server Edition: `Llama-3.1-8B` for deep reasoning, dialect translation, and intent analysis
- Workstation Edition: `Qwen-2.5-7B` for efficient translation and correction
- 具備方言轉正、語意修正及情報摘要生成能力

**🛡️ Air-Gapped Security**
- Fully offline execution — no data leaves the server
- Model weights pre-downloaded to local storage
- 支援完全離線模式，適合機密敏感環境

**⚡ Pipeline Parallelism** *(Server Edition Only)*
- ASR (Hearing) and LLM (Reasoning) distributed across separate GPUs
- 「聽」與「想」硬體分流，大幅提升批次處理吞吐量

---

## 🛠️ Requirements

**Common:**
- NVIDIA Driver 535+ (CUDA 12.1+)
- Python 3.10 (Conda recommended)

**Workstation:** 1× GPU with 24GB+ VRAM (Windows/Linux)

**Server:** 2× GPUs with 20GB+ VRAM each + RAID storage (Ubuntu)

---

## 📦 Installation

```bash
# 1. Clone repository
git clone https://github.com/wu840407/yayan-ai.git
cd yayan-ai
mkdir -p models_cache input_audio output_text

# 2. Create environment
conda create -n yayan_ai python=3.10 -y
conda activate yayan_ai
pip install -r requirements.txt
```

> ⚠️ First run will auto-download models (~15GB). Please be patient.
> 注意：首次執行將自動下載模型（約 15GB），請耐心等待。

---

## ▶️ Usage

### Option A — Workstation (RTX 3090 / Single GPU)

```bash
# Interactive WebUI
python app.py

# Batch processing (reads from ./input_audio)
python auto_batch.py
```

### Option B — Server (Dual RTX 4000 Ada)

```bash
# Interactive WebUI (Server Mode)
python app_rtx4000.py

# Batch processing (reads from /data/input_audio)
python auto_batch_rtx4000.py
```

---

## 🏗️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Inference Engine | PyTorch, Hugging Face Transformers |
| ASR | OpenAI Whisper-Large-v3 |
| LLM | Meta Llama-3.1-8B / Qwen-2.5-7B |
| Quantization | BitsAndBytes (NF4) |
| Audio Processing | Librosa, SoundFile |
| Interface | Gradio |
| Deployment | Docker Ready (Server Edition) |

---

## 🗺️ Roadmap

- [x] Single-GPU workstation support (RTX 3090)
- [x] Dual-GPU server pipeline parallelism (RTX 4000 × 2)
- [x] Batch processing automation
- [ ] H200 × 2 upgrade & large-scale fine-tuning (2026 Q3)
- [ ] REST API endpoint
- [ ] Custom dialect fine-tuning pipeline

---

## 📝 License

This project is open-source under the [MIT License](LICENSE).

---

## 👤 Author

**ChengRung Wu (吳承融)**
📧 wu840407@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/chengrung-wu-935b6b105)
🐙 [GitHub](https://github.com/wu840407)
