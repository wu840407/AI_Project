# 🏺 YaYan-AI (雅言) - Cross-Architecture Dialect Intelligence

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20(Edge%2FServer)-purple)
![License](https://img.shields.io/badge/License-MIT-orange)

> **Scalable Local Dialect Intelligence System | 從工作站到伺服器的全本地化方言情報系統**

## 📖 Introduction (專案簡介)

**[English]**
**YaYan-AI** is a privacy-first, offline AI system designed to convert dialectal speech (e.g., Taiwanese, Cantonese, Uyghur) into standard Traditional Chinese intelligence reports. 
This project features a **cross-architecture design**, seamlessly supporting both consumer-grade workstations (RTX 3090) and enterprise-grade servers (Dual RTX 4000), ensuring flexibility across different deployment scenarios.

**[中文]**
**雅言 (YaYan-AI)** 是一套基於本地化部署的 AI 情報系統，致力於將多種方言（如台灣口語、粵語、維吾爾語）轉化為標準的「雅言」（正體中文情報摘要）。
本專案採用**跨架構設計**，同時支援單卡工作站（RTX 3090）與企業級伺服器（Dual RTX 4000），實現從原型開發到大規模情報分析的無縫遷移。

---

## 🌟 Architecture & Versions (版本與架構)

This repository maintains specialized configurations for different hardware environments.
本專案針對不同硬體規模提供優化配置：

| Feature | **v1: Workstation Edition** | **v2: Server Edition** |
| :--- | :--- | :--- |
| **Use Case** | Prototyping / Edge Inference | **Massive Batch Processing** |
| **GPU Config** | **1x NVIDIA RTX 3090** (24GB) | **2x NVIDIA RTX 4000 Ada** (20GB x2) |
| **Strategy** | Serial Processing (序列處理) | **Pipeline Parallelism (平行管線)** |
| **ASR Model** | Whisper-Large-v3 | Whisper-Large-v3 (Run on GPU 0) |
| **LLM Model** | Qwen-2.5-7B (4-bit) | **Meta-Llama-3.1-8B** (Run on GPU 1) |
| **Storage** | Local SSD | **RAID 10 NVMe Array (/data)** |
| **OS** | Windows 10/11 (WSL2) | **Ubuntu Server 24.04 LTS** |

---

## 🚀 Key Features (核心功能)

* **🎙️ Military-Grade ASR (高精度聽寫)**
    * Deploys `whisper-large-v3` locally to handle diverse acoustic environments (PSTN/VoIP).
    * 本地部署最新 Whisper 模型，針對電話錄音優化，精準捕捉方言發音。

* **🧠 Strategic Intelligence Analysis (戰略情報分析)**
    * **Server Edition:** Utilizes **Llama-3.1-8B** for deep reasoning, dialect translation, and intent analysis.
    * **Workstation Edition:** Uses **Qwen-2.5-7B** for efficient translation and correction.
    * 具備方言轉正、語意修正及情報摘要生成能力。

* **🛡️ Air-Gapped Security (物理隔離安全)**
    * Supports fully offline execution. No data leaves your server.
    * 支援**完全離線模式**，模型權重可預先下載至本地硬碟，適合機密敏感環境。

* **⚡ Pipeline Parallelism (雙卡平行加速)**
    * *Server Edition Only*: Distributes ASR (Hearing) and LLM (Reasoning) tasks across separate GPUs.
    * 伺服器版實作「聽」與「想」的硬體分流，大幅提升批次處理吞吐量。

---

## 🛠️ Requirements (環境需求)

### Common (通用需求)
* **Driver:** NVIDIA Driver 535+ (CUDA 12.1+)
* **Python:** 3.10 (Conda environment recommended)

### Hardware Specifics (硬體需求)
* **Workstation:** Windows/Linux with 1x GPU (24GB VRAM)
* **Server:** Linux (Ubuntu) with 2x GPUs (min 20GB VRAM each) + **RAID Storage**.

---

## 📦 Installation (安裝步驟)

### 1. Clone Repository (下載專案)
    
    git clone [https://github.com/YourUsername/YaYan-AI.git](https://github.com/YourUsername/YaYan-AI.git)
    cd YaYan-AI
    mkdir -p models_cache input_audio output_text
    
### 2. Create Environment (建立環境)
    
    conda create -n yayan_ai python=3.10 -y
    conda activate yayan_ai
    pip install -r requirements.txt
    
## ▶️ Usage (使用方法)
### Option A: Running on Workstation (RTX 3090)
Uses Qwen-7B and Single GPU logic. 適用於單卡開發環境。
    # Start Web UI
    python app.py

    # Batch Process (Input folder: ./input_audio)
    python auto_batch.py
    
Note: The first run will automatically download models (~15GB). Please wait. 注意： 首次執行將自動下載模型（約 15GB），請耐心等待進度條跑完。
    
### Option B: Running on Server (Dual RTX 4000)
    # Start Web UI (Server Mode)
    python app_rtx4000.py

    # Batch Process (Input folder: /data/input_audio)
    python auto_batch_server.py
    
## 🏗️ Technical Stack (技術架構)
    
* **Inference Engine:** PyTorch, Hugging Face Transformers

* **Quantization:** BitsAndBytes (NF4) for VRAM optimization

* **Audio Processing:** Librosa, SoundFile

* **Interface:** Gradio (WebUI)

* **Deployment:** Docker Ready (Server Edition)

## 📝 License
    This project is open-source and available under the MIT License. 