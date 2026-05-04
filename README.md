# 🏺 YaYan-AI v4.5 — Multilingual Dialect Intelligence

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-4.5.0-purple.svg)]()
[![Edition](https://img.shields.io/badge/Edition-RTX6000--Server-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Deployment-brightgreen.svg)]()

> **本地化、可離線、可審計的多方言語音情報系統**

---

## 📖 簡介

**YaYan-AI v4.5** 將多種方言與少數民族語音轉換為**台灣正體中文**書面情報，全程**離線執行**、**權重不外傳**。本版針對 Quadro RTX 6000 (Turing 24GB) 雙卡伺服器最佳化。

支援：

| 類別 | 語種 |
|---|---|
| 漢語方言 | 北京話 / 山東話 / 上海話 / 四川話 / 廣東話 |
| 邊疆語族 | 維吾爾語 / 藏語 |
| 中亞 / 南亞 | 波斯語 (Farsi) / 烏爾都語 (Urdu) |
| 通用 | 英語 / 阿拉伯語 等 96+ 語言 |

---

## 🌟 Architecture & Versions

| Feature | v1: Workstation | v2: Server (Ada) | **v4.5: Server (Turing)** ⭐ |
| :--- | :--- | :--- | :--- |
| **Use Case** | Prototyping | Massive Batch | **Multi-Dialect Production** |
| **GPU Config** | 1× RTX 3090 24GB | 2× RTX 4000 Ada 20GB | **2× Quadro RTX 6000 24GB** |
| **Strategy** | Serial | Pipeline Parallelism | **LID Routing + Pipeline** |
| **OS** | Windows / WSL2 | Ubuntu 24.04 | **Ubuntu 22.04 LTS** |
| **ASR Strategy** | Whisper-Large-v3 only | Whisper-Large-v3 (GPU 0) | **3-Model Router**: `YaYan_ASR_Mandarin` / `YaYan_ASR_Eastern` / `YaYan_ASR_Global` |
| **Dialect ID** | LLM 文字猜測 | LLM 文字猜測 | **`YaYan_LID` 音訊前置識別** |
| **Tibetan / Uyghur** | ❌ | ❌ | ✅ via `YaYan_ASR_Eastern` |
| **Persian / Urdu** | ❌ | ❌ | ✅ via `YaYan_ASR_Global` |
| **VAD (長音檔切段)** | ❌ | ❌ | ✅ `YaYan_VAD` (Silero) |
| **Diarization (多人分離)** | ❌ | ❌ | ✅ optional `YaYan_Diarize` |
| **LLM** | Qwen-2.5-7B (4-bit) | Llama-3.1-8B (4-bit) | **`YaYan_Reasoner`** (4-bit, Turing FP16) |
| **對話修改 UX** | ❌ | ❌ | ✅ **編輯原文/譯文 → 一鍵重新潤飾** |
| **正體中文保證** | 靠 prompt | 靠 prompt | ✅ **OpenCC `s2twp` 強制後處理** |
| **批次自動化** | for-loop | for-loop | ✅ **watchdog 監聽 + polling fallback** |
| **離線部署** | 手動 | 手動 | ✅ **完整 download/install/verify 三段腳本** |
| **容器化** | ❌ | ❌ | ✅ **Docker + docker-compose（可選）** |
| **Storage** | Local SSD | RAID 10 NVMe | RAID 10 NVMe (`/data`) |
| **Interface** | Gradio WebUI | Gradio + CLI | **Gradio + CLI + Watch Mode + Docker** |
| **Entry Point** | `app.py` | `app_rtx4000.py` | `app_rtx6000.py` ⭐ |

---

## 🧠 模型路由

所有對外名稱均為 `YaYan_` 前綴；對應的內部來源與授權見 [`NOTICE.md`](NOTICE.md)。

| 來源語種 | 路由到 | 預期端到端準確率 |
|---|---|---|
| 北京話 / 山東話 / 普通話 | `YaYan_ASR_Mandarin` | 95-97% |
| 四川話 | `YaYan_ASR_Mandarin` | 90-93% |
| 上海話 (吳語) | `YaYan_ASR_Mandarin` (備援 `YaYan_ASR_Eastern`) | 80-87% |
| 廣東話 | `YaYan_ASR_Mandarin` | 92-95% |
| 維吾爾語 | `YaYan_ASR_Eastern` | 78-85% |
| 藏語 | `YaYan_ASR_Eastern` | 75-82% |
| 波斯語 | `YaYan_ASR_Global` | 88-92% |
| 烏爾都語 | `YaYan_ASR_Global` | 85-90% |

LLM 翻譯與潤飾統一使用 `YaYan_Reasoner`（4-bit 量化，常駐 GPU 1）。

---

## 🛠️ 硬體 / 系統需求

- Ubuntu 22.04 LTS
- NVIDIA Driver **535+** (CUDA 12.1+)
- Quadro RTX 6000 (Turing) **× 2** 或同等級以上
- Python 3.10
- 至少 100 GB 可用 SSD（模型 ~40GB + 工作空間）
- RAM 64GB+
- ffmpeg / libsndfile1（系統套件）

---

## 📦 部署流程

> 模型體積約 40GB，**直接在伺服器（或任一具網路連線的中繼機器）下載**即可。
> 連網機器與離線機器可以是同一台，也可分離。

### A. 伺服器有網路（推薦最短路徑）

```bash
# 在伺服器（Ubuntu 22.04）
git clone https://github.com/wu840407/YaYan_AI.git /opt/yayan
cd /opt/yayan

# A.1 建環境
conda create -n yayan python=3.10 -y
conda activate yayan
pip install -U huggingface_hub
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# A.2 下載模型到 /data/ai_models（約 40GB）
sudo mkdir -p /data/{ai_models,input_audio,output_text}
sudo chown -R $USER:$USER /data
MODELS_ROOT=/data/ai_models bash scripts/download_models.sh

# A.3 驗證 + 啟動
python scripts/verify_models.py
bash scripts/start_server.sh           # → http://server-ip:7860
bash scripts/start_batch.sh            # 另開 terminal：watch /data/input_audio/
```

下載完模型後可以**拔網路線**改成完全離線，所有環境變數已預設 `HF_HUB_OFFLINE=1`。

### B. 伺服器完全沒有網路（中繼機器分離流程）

需要一台具網路的「**中繼機器**」（任意 Linux/Mac/WSL 都可，**不需要 GPU**）負責下載打包：

```bash
# === 在「中繼機器」===
git clone https://github.com/wu840407/YaYan_AI.git
cd YaYan_AI

pip install -U huggingface_hub
MODELS_ROOT=./ai_models bash scripts/download_models.sh   # 下載 ~40GB
bash scripts/install_offline.sh package                    # 打包 wheels

# === 把以下三樣搬到伺服器 ===
rsync -av --progress ai_models/         user@server:/data/ai_models/
rsync -av --progress offline_packages/  user@server:/opt/yayan/offline_packages/
rsync -av --exclude='.git' ./           user@server:/opt/yayan/
```

```bash
# === 在「離線伺服器」===
ssh user@server
cd /opt/yayan

conda create -n yayan python=3.10 -y && conda activate yayan
bash scripts/install_offline.sh install     # 用 offline_packages/ 安裝
python scripts/verify_models.py              # 驗證 7 個模型就位
bash scripts/start_server.sh                 # 啟動 Web → :7860
bash scripts/start_batch.sh                  # 啟動批次（另開 terminal）
```

---

## 🐳 Docker 部署（可選 / 較穩）

**注意**：離線環境用 Docker 需把整包 image 先 `docker save` 過去，比 Conda 多一步。建議**先用 Conda 上線，再評估換 Docker**。

```bash
# === 連網機器 ===
docker build -t yayan-ai:4.5 -f docker/Dockerfile .
docker save yayan-ai:4.5 -o yayan-ai-4.5.tar

# rsync yayan-ai-4.5.tar 與 /data/ai_models 到伺服器

# === 離線伺服器 ===
docker load -i yayan-ai-4.5.tar

# 確認 nvidia-container-toolkit 已安裝
nvidia-ctk --version

# 啟動 Web UI
docker compose -f docker/docker-compose.yml up -d yayan-server

# 啟動批次（可選）
docker compose -f docker/docker-compose.yml --profile batch up -d yayan-batch

# 看 log
docker logs -f yayan-server
```

---

## 🎛️ 使用方法

### Web UI（保留上傳 + 對話修改）

```bash
python app_rtx6000.py
```

打開 http://localhost:7860：

1. **上傳**音檔（或用麥克風錄）
2. 選方言（建議「自動偵測」）
3. 點「**🚀 開始轉錄翻譯**」
4. 結果出現在右側兩個框：
   - **ASR 原文**：可直接修改
   - **台灣正體譯文**：可直接修改
5. 修改後點：
   - **🔄 依編輯後原文重新翻譯潤飾**：以你改過的原文重跑翻譯
   - **✨ 依編輯後譯文重新潤飾**：以你改過的譯文交給 LLM 微調
6. 滿意後點「**💾 儲存譯文**」→ 寫入 `/data/output_text/`

### 批次處理

```bash
# 一次性處理 /data/input_audio/ 所有檔
python auto_batch_rtx6000.py

# 持續監聽（H200 上線後自動化的入口）
python auto_batch_rtx6000.py --watch

# 啟用說話人分離（雙人通話用）
python auto_batch_rtx6000.py --diarize
```

---

## 🗂️ 專案結構

```
YaYan_AI/
├── README.md                      # 本檔
├── NOTICE.md                      # 第三方授權合規
├── requirements.txt
├── app.py                         # v1 舊版（保留）
├── app_rtx4000.py                 # v2 舊版（保留）
├── app_rtx6000.py                 # ⭐ v4.5 主入口
├── auto_batch.py                  # 舊版
├── auto_batch_rtx4000.py          # 舊版
├── auto_batch_rtx6000.py          # ⭐ v4.5 批次（含 watchdog）
├── configs/
│   ├── default.yaml               # 主設定（路徑/GPU/路由/LLM）
│   ├── model_aliases.yaml         # YaYan_* ↔ 內部 ID 對照（內部用）
│   └── prompts/
│       ├── translate.txt
│       └── refine.txt
├── yayan/                         # 核心模組
│   ├── config.py
│   ├── pipeline.py                # 端到端 orchestrator
│   ├── vad.py
│   ├── lid.py
│   ├── diarize.py
│   ├── asr/{router,sensevoice,dolphin,whisper_global}.py
│   └── llm/{client,postprocess}.py
├── scripts/
│   ├── download_models.sh         # 連網機器跑
│   ├── install_offline.sh         # package | install
│   ├── verify_models.py           # 驗證模型完整
│   ├── start_server.sh            # 啟動 Web
│   └── start_batch.sh             # 啟動批次監聽
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── .dockerignore
```

---

## ⚙️ 設定調校

`configs/default.yaml` 常用參數：

| 設定 | 預設 | 說明 |
|---|---|---|
| `paths.models_root` | `/data/ai_models` | 模型實體目錄 |
| `paths.input_dir` | `/data/input_audio` | 批次輸入 |
| `paths.output_dir` | `/data/output_text` | 譯文輸出 |
| `devices.asr_gpu` | `cuda:0` | ASR 卡 |
| `devices.llm_gpu` | `cuda:1` | LLM 卡 |
| `llm.backend` | `transformers` | 換 `vllm` 可加速（Turing 需 `enforce_eager`） |
| `llm.quantization` | `4bit` | NF4 量化 |
| `asr.enable_vad` | `true` | 長音檔切段 |
| `asr.enable_lid` | `true` | 自動方言識別 |
| `diarize.enabled` | `false` | 雙人通話再開 |
| `postprocess.opencc_config` | `s2twp.json` | 簡 → 台灣正體 |

或用環境變數覆寫：`YAYAN_MODELS_ROOT` / `YAYAN_INPUT_DIR` / `YAYAN_OUTPUT_DIR`。

---

## 🩺 故障排查

| 症狀 | 處理 |
|---|---|
| `FileNotFoundError: YaYan_* 不存在` | 先在連網機器跑 `download_models.sh`，再 rsync 到 `/data/ai_models` |
| `ImportError: vllm` | 用預設 `transformers` 後端即可，或 `pip install vllm==0.6.4` |
| `OutOfMemory` (LLM 卡) | 把 `llm.max_model_len` 從 4096 降到 2048；確認 `quantization: 4bit` |
| `Turing kernel not supported` | vLLM 後端時 `enforce_eager: true` 必開 |
| ASR 出簡體 | 確認 `postprocess.use_opencc: true`，pip 安裝 `opencc-python-reimplemented` |
| Diarization 報 gated | 不啟用即可；若要用：`HF_TOKEN=... bash download_models.sh --with-diarize` |
| 連網下載被擋 | 全程在連網機器先跑完 download，再搬資料 |

---

## 🗺️ Roadmap

- [x] **v4.5 — RTX 6000 (Turing) × 2** 多模型路由 + 對話修改 UX + 離線
- [ ] **v5.0 — H200 × 2** 升級：vLLM tensor parallel、Qwen3.6-27B、Redis queue 多 worker
- [ ] **v5.1** REST API + 客戶端 RBAC
- [ ] **v5.2** 自定義方言 LoRA fine-tune pipeline（針對藏語、上海話、維吾爾語拉到 90%+）

---

## 📝 License

MIT — 見 [`LICENSE`](LICENSE)。第三方組件之署名與授權見 [`NOTICE.md`](NOTICE.md)。

## 👤 Author

**ChengRung Wu (吳承融)** · 📧 wu840407@gmail.com · 🔗 [LinkedIn](https://www.linkedin.com/in/chengrung-wu-935b6b105) · 🐙 [GitHub](https://github.com/wu840407)
