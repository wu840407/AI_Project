# 🏺 YaYan-AI (雅言)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%203090-green)
![License](https://img.shields.io/badge/License-MIT-orange)

> **Local Dialect Conversion Engine | 本地化方言轉正體中文系統**

## 📖 Introduction (專案簡介)

**[English]**
**YaYan-AI** is a local, offline AI system capable of converting various Chinese dialects (e.g., Taiwanese, Cantonese, Sichuanese) into standard Traditional Chinese text. It leverages **Whisper-Large-v3** for high-fidelity Automatic Speech Recognition (ASR) and **Qwen-2.5-7B-Instruct** for context-aware dialect correction and translation.
Designed to run efficiently on a single **NVIDIA RTX 3090 (24GB)** using 4-bit quantization.

**[中文]**
**雅言 (YaYan-AI)** 是一個基於 NVIDIA RTX 3090 的本地化 AI 系統，致力於消除方言隔閡，將口語（如台灣國語、粵語、四川話）轉化為標準的「雅言」（正體中文書面語）。
本專案結合了 **OpenAI Whisper-Large-v3** 的強大聽力與 **Qwen-2.5-7B** 的深度理解能力，在本地端實現高隱私、高精度的語音重塑。

---

## 🚀 Features (特色功能)

* **🎙️ High-Accuracy ASR (高精度聽寫)**
    * Uses `whisper-large-v3` to transcribe speech with high fidelity.
    * 採用 OpenAI 最新模型，精準捕捉方言發音。

* **🧠 Dialect Correction (方言轉正)**
    * Uses `Qwen-2.5-7B` (LLM) to fix ASR errors and convert colloquialisms to formal text.
    * 修正語音識別錯誤（如同音異字），並將口語語法轉為規範書面語。

* **🔒 Local Privacy (本地隱私)**
    * Everything runs locally on your GPU. No data is sent to the cloud.
    * 全程在本地 RTX 3090 運算，數據不上雲端，適合機敏資料。

* **⚡ Optimized Performance (效能優化)**
    * Implements `bitsandbytes` 4-bit quantization.
    * 實作 4-bit 量化技術，單卡 24GB 顯存即可流暢運行兩大模型。

---

## 🛠️ Requirements (環境需求)

* **OS:** Windows 10/11 (via WSL2 Ubuntu) or Linux
* **GPU:** NVIDIA GPU with 24GB VRAM (Recommended: RTX 3090 / 4090)
* **Driver:** CUDA 12.1+
* **Python:** 3.10 (Conda environment recommended)

---

## 📦 Installation (安裝步驟)

### 1. Clone Repository (下載專案)
    
    git clone [https://github.com/YourUsername/YaYan-AI.git](https://github.com/YourUsername/YaYan-AI.git)
    cd YaYan-AI
    mkdir -p models_cache input_audio output_text
    
### 2. Create Environment (建立環境)
    
    conda create -n dialect_env python=3.10 -y
    conda activate dialect_env
    
### 3. Install PyTorch (安裝 PyTorch)
    
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
### 4. Install Dependencies (安裝核心套件)
    
    pip install transformers accelerate bitsandbytes peft gradio librosa scipy soundfile protobuf sentencepiece
    
## ▶️ Usage (使用方法)
### 1. Start the System (啟動系統)
    
    conda activate dialect_env
    python app.py
    
Note: The first run will automatically download models (~15GB). Please wait. 注意： 首次執行將自動下載模型（約 15GB），請耐心等待進度條跑完。
    
### 2. Open Web UI (開啟介面)
Once the terminal shows the URL, open your browser and visit: 當終端機顯示網址後，請打開瀏覽器輸入：

http://localhost:7860

### 3. Batch Processing (批次處理)
Automatically process all files in input_audio/. 自動轉換 input_audio 資料夾內所有音檔。
    
    python auto_batch.py
    
## 🏗️ Technical Stack (技術架構)
    
    ASR Model: openai/whisper-large-v3

    LLM Model: Qwen/Qwen2.5-7B-Instruct (Quantized: NF4)

    Acceleration: bitsandbytes (4-bit quantization)

    Interface: Gradio

## 📝 License
    This project is open-source and available under the MIT License. 