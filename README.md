# 🏺 YaYan-AI (雅言) - Cross-Architecture Dialect Intelligence

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Architecture](https://img.shields.io/badge/Architecture-Dual%20RTX%204000-purple)
![License](https://img.shields.io/badge/License-MIT-orange)

> **Scalable Local Dialect Intelligence System**
> **從單兵工作站到戰情伺服器的全本地化方言情報系統**

---

## 📖 Introduction (專案簡介)

**[English]**
**YaYan-AI** is a privacy-first, offline AI system designed to convert dialectal speech (e.g., Taiwanese, Sichuanese, Cantonese) into standard Traditional Chinese intelligence reports.
Version 2.1 introduces a **Dual-GPU Pipeline**, utilizing two NVIDIA RTX 4000 Ada GPUs to separate ASR (Hearing) and LLM (Reasoning) tasks, resolving memory bottlenecks and increasing throughput.

**[中文]**
**雅言 (YaYan-AI)** 是一套基於本地化部署的 AI 情報系統，致力於將多種方言（如台灣口語、四川話、粵語）轉化為標準的「雅言」（正體中文情報摘要）。
V2.1 版本引入 **雙顯卡平行管線 (Dual-GPU Pipeline)**，利用兩張 RTX 4000 分別處理「聽」與「想」，徹底解決了 VRAM 溢出 (OOM) 問題並大幅提升處理速度。

---

## 🌟 Architecture Evolution (架構演進)

This repository maintains configurations for different hardware scales.
本專案針對不同硬體規模提供優化配置：

| Feature (功能) | **v1: Workstation (單兵版)** | **v2.1: Server (戰情版)** |
| :--- | :--- | :--- |
| **Use Case (定位)** | Prototyping / Edge Inference<br>原型開發 / 邊緣運算 | **Massive Batch Processing<br>大規模戰情分析** |
| **GPU Config (硬體)** | **1x NVIDIA RTX 3090** (24GB) | **2x NVIDIA RTX 4000 Ada** (20GB x2) |
| **Strategy (策略)** | Serial Processing (序列處理) | **Pipeline Parallelism (平行管線)** |
| **ASR (聽覺)** | Whisper-Large-v3 | **GPU 0:** Whisper-Large-v3 + Pyannote |
| **LLM (大腦)** | Qwen-2.5-7B (4-bit) | **GPU 1:** Llama-3.1-8B-Instruct (4-bit) |
| **Dialect (方言)** | Basic Prompting | **Advanced Dialect Dashboard (多方言儀表板)** |
| **OS (系統)** | Windows 10/11 (WSL2) | **Ubuntu Server 22.04 / 24.04** |

---

## 🚀 Key Features (核心功能 V2.1)

### 1. 🗣️ Multi-Dialect Dashboard (多方言戰情儀表板)
* **EN:** New dropdown menu supports **Taiwanese (Hokkien), Sichuanese, Cantonese, Shanghainese, and Shandong dialect**. Uses advanced prompt engineering to fix homophone errors (e.g., fixing Sichuanese "empty ear" errors).
* **TW:** 新增方言切換面板，支援**台語、四川話、粵語、上海話、山東話**。透過 Llama 3.1 的方言指令庫，自動修復 Whisper 的同音字錯誤（如修復四川話的空耳現象）。

### 2. 🛡️ Dual-GPU Optimization (雙卡平行優化)
* **EN:** Solved `CUDA OutOfMemory` issues by dedicating **GPU 0 for ASR** (Whisper + Pyannote) and **GPU 1 for LLM** (Llama 3.1).
* **TW:** 透過硬體分流，將「聽寫」交給第一張顯卡，「思考」交給第二張顯卡，完美解決單卡記憶體不足的崩潰問題，實現流水線式處理。

### 3. 📊 Confidence Scoring (信心指數可視化)
* **EN:** Real-time LogProb calculation displays AI transcription confidence (Green/Orange/Red indicators), helping analysts judge data reliability.
* **TW:** 實時計算 AI 聽寫的信心水準（LogProb），並以紅/黃/綠燈號顯示，輔助情報官快速判斷逐字稿的可信度。

### 4. 🧠 Strategic Analysis Modes (戰略研判模式)
* **EN:** Includes three specialized modes: **Summary**, **Strategic Intent Analysis**, and **Game Theory Suggestions**.
* **TW:** 內建三種戰術分析模式：**情報總結**、**戰略意圖研判**（分析潛在目的與心理狀態）、**謀略導變建議**（引用博弈論與孫子兵法）。

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