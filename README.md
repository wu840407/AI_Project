# 🏺 YaYan-AI v4.6

> **多方言語音情報系統** — 內網離線部署的 22 中文方言語音轉文字 + 翻譯平台

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch-2.3.1+cu121-ee4c2c.svg)](https://pytorch.org/)

---

## ✨ 功能特色

- **22 種中文方言識別** — 普通話、粵語、吳語（上海/蘇州/寧波/溫州）、閩南語/台語、福州話、客家話、湘語、贛語、晉語、四川話、東北話、山東話、河南話等
- **40+ 全球語言翻譯** — 日韓、歐美、中東、東南亞語言識別後統一翻譯為台灣正體中文
- **字級時間戳** — 輸出格式 `[A方 00:01-00:05] 你好`，可精確定位錄音時間點
- **5 人說話人分離** — A方 / B方 / C方 / D方 / E方
- **逐段語言識別** — 混合語音場景每段獨立 LID 判斷，不會「整段被一種語言主導」
- **完全離線運行** — 軍政府網域可用，所有模型本地化部署
- **可編輯回饋循環** — 使用者修改 ASR 原文或譯文後，LLM 重新潤飾

---

## 🖼️ 截圖

```
┌────────────────────────────────────────────────────────────────────┐
│ 🏺 YaYan-AI v4.6                                                   │
│ Edition: RTX6000-Server | ASR: Dolphin-CN-Dialect / Whisper-v3     │
│                          LLM: Qwen3-14B + NF4                      │
├──────────────────────────────┬─────────────────────────────────────┤
│ 🎤 上傳音檔                  │ 📜 識別原文（可編輯）               │
│ [audio waveform]             │ [A方 00:01-00:05] 大家好我是阿明    │
│                              │ [B方 00:06-00:09] 歹勢啦今天有事     │
│ 來源語言: 🔍 自動偵測         │ [A方 00:10-00:14] 我兒子今天去福州  │
│ ☑ 啟用說話人分離             │ [C方 00:15-00:18] 食飯了無           │
│                              │                                      │
│ [🚀 開始轉錄翻譯]            │ 🇹🇼 台灣正體中文譯文（可編輯）       │
│                              │ [A方 00:01-00:05] 大家好我是阿明     │
│ 📊 識別資訊                  │ [B方 00:06-00:09] 不好意思今天有事   │
│ 語言分布:                    │ [A方 00:10-00:14] 我兒子今天去福州   │
│   閩南語 45%｜普通話 35%     │ [C方 00:15-00:18] 吃飯了嗎           │
│   福州話 13%｜客家話 7%      │                                      │
│ 說話人: 3 位｜段數: 62       │ [🔄 重新潤飾] [💾 另存新檔]          │
│                              │                                      │
│ 🎯 識別精準度                 │                                      │
│      87.5 / 100              │                                      │
│  品質良好                    │                                      │
└──────────────────────────────┴─────────────────────────────────────┘
```

---

## 🧱 技術棧

| 層 | 模型 / 套件 | 版本 | 用途 |
|---|---|---|---|
| **ASR — 中文方言** | DataoceanAI/dolphin-small | 2026-05 | 22 方言 + 字級時間戳 |
| **ASR — 全球** | openai/whisper-large-v3 | - | 99 語言（日韓歐美中東東南亞）|
| **LID** | speechbrain/lang-id-voxlingua107-ecapa | - | 107 語言識別 |
| **VAD** | snakers4/silero-vad | 6.x | 語音活動偵測（pip-bundled）|
| **Diarize** | pyannote/speaker-diarization-3.1 | - | 說話人分離（5 人）|
| **LLM** | Qwen/Qwen3-14B | - | 翻譯與潤飾（BF16 + NF4 動態量化）|
| **Framework** | PyTorch / transformers / gradio | 2.3.1 / 4.51.3 / 4.44.1 | - |
| **OS / GPU** | Ubuntu 24.04 / 2× RTX 6000 (Turing 24GB) | - | - |

---

## 🚀 安裝部署

### 硬體需求

- **GPU**：≥ 24GB VRAM × 1 GPU（或 24GB × 2 雙 GPU 分配 ASR/LLM）
- **RAM**：≥ 32GB
- **儲存**：≥ 50GB（含所有模型）
- **CUDA driver**：≥ 535（建議 580+）

### 系統相依

```bash
sudo apt install -y ffmpeg libsndfile1
```

### Python 環境

```bash
# 建立 conda env
conda create -n yayan python=3.10 -y
conda activate yayan

# 安裝 PyTorch + CUDA
pip install torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 安裝其餘相依
pip install -r requirements.txt

# 安裝 Dolphin SDK
pip install dataoceanai-dolphin
```

### 下載模型

第一次部署需從 HuggingFace 下載 ~38GB 模型：

```bash
bash scripts/download_models.sh --with-diarize
```

下載清單：

| Alias | HF Repo | 大小 |
|---|---|---|
| `YaYan_Reasoner` | `Qwen/Qwen3-14B` | ~28GB |
| `YaYan_ASR_Dialect` | `DataoceanAI/dolphin-small` | ~1.6GB |
| `YaYan_ASR_Global` | `openai/whisper-large-v3` | ~3GB |
| `YaYan_LID` | `speechbrain/lang-id-voxlingua107-ecapa` | ~100MB |
| `YaYan_Diarize` | `pyannote/speaker-diarization-3.1` | ~10MB |
| `YaYan_Diarize_Seg` | `pyannote/segmentation-3.0` | ~6MB |
| `YaYan_Diarize_Embed` | `pyannote/wespeaker-voxceleb-resnet34-LM` | ~30MB |

> ⚠️ pyannote 3 個 repo 為 gated，需先在 HF 接受授權並設定 `HF_TOKEN`

### 驗證

```bash
python scripts/verify_models.py
```

預期全綠：

```
✅ YaYan_Reasoner     (28,xxx MB) [dtype=bfloat16]
✅ YaYan_ASR_Dialect   (1,6xx MB)  Dolphin-CN-Dialect-Small
✅ YaYan_ASR_Global    (3,xxx MB)  Whisper-large-v3
✅ YaYan_LID              (xxx MB)  VoxLingua107 ECAPA
✅ silero-vad (pip-bundled)
✅ dolphin SDK
✅ YaYan_Diarize / YaYan_Diarize_Seg / YaYan_Diarize_Embed
```

### 啟動

```bash
bash scripts/start_server.sh
```

打開瀏覽器：`http://localhost:7860` 或 `http://<server-ip>:7860`

---

## 📁 專案結構

```
YaYan_AI/
├── app_rtx6000.py              # Gradio Web UI 入口
├── auto_batch_rtx6000.py       # Batch 自動監聽模式
├── requirements.txt
├── configs/
│   ├── default.yaml            # 主設定（routing、LLM 參數、devices）
│   ├── model_aliases.yaml      # alias → HF repo 對照
│   └── prompts/
│       ├── translate.txt       # 翻譯 prompt（22 方言意譯範例）
│       └── refine.txt          # 潤飾 prompt
├── yayan/
│   ├── __init__.py
│   ├── config.py               # 設定載入
│   ├── pipeline.py             # 端到端 orchestrator
│   ├── vad.py                  # silero-vad 切片
│   ├── lid.py                  # 語種識別
│   ├── diarize.py              # pyannote 說話人分離
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── router.py           # 依 routing 分配到 Dolphin / Whisper
│   │   ├── dolphin.py          # Dolphin-CN-Dialect 包裝
│   │   └── whisper_global.py   # Whisper-large-v3 包裝
│   └── llm/
│       ├── __init__.py
│       ├── client.py           # transformers / vllm / openai 三後端
│       └── postprocess.py      # OpenCC + thinking tag 清理
└── scripts/
    ├── start_server.sh
    ├── download_models.sh
    ├── verify_models.py
    └── verify_offline.py
```

---

## 🎯 支援的語言與方言

### 漢語方言（22 種，走 Dolphin-CN-Dialect）

| 大類 | 子方言 |
|---|---|
| **北方官話** | 普通話、東北話、山東話、河南話、西安話、蘭州話 |
| **西南官話** | 四川話、武漢話 |
| **江淮官話** | 南京話 |
| **吳語** | 上海話、蘇州話、寧波話、溫州話 |
| **粵語** | 廣東話 |
| **閩語** | 閩南語/台語、潮汕話、海南話、福州話（閩東語）|
| **客贛湘晉** | 客家話、湘語、贛語、晉語 |

### 中亞 / 少數民族

藏語、維吾爾語

### 全球語言（走 Whisper-large-v3）

日文、韓文、英文、法文、德文、俄文、西班牙文、波斯文、阿拉伯文、烏爾都文、印地文、泰文、越南文、馬來文、印尼文

---

## 🔄 使用流程

1. **上傳音檔**（或現場錄音）
2. **選擇來源語言**（建議「自動偵測」）
3. **可選：啟用說話人分離**（雙人以上對話建議）
4. 點 **🚀 開始轉錄翻譯**
5. 取得：
   - ASR 原文（含 `[A方 00:01-00:05]` 時間戳）
   - 台灣正體中文譯文
   - 識別精準度分數（0-100）
   - 語言分布統計
6. **編輯回饋**（可選）：
   - 修改 ASR 原文 → 「依編輯後原文重新翻譯潤飾」
   - 修改譯文 → 「依編輯後譯文重新潤飾」
7. **💾 另存新檔** 下載 txt 結果

---

## 🔧 設定調校

`configs/default.yaml` 主要參數：

```yaml
asr:
  enable_vad: true         # silero-vad 切片
  enable_lid: true         # 永遠開逐段 LID
  enable_diarize: true     # 預設開說話人分離

audio:
  vad_threshold: 0.5       # VAD 敏感度（0.3 = 寬鬆，0.7 = 嚴格）
  max_chunk_seconds: 30    # 單段最長秒數

diarize:
  min_speakers: 1
  max_speakers: 5          # ABCDE

llm:
  quantization: nf4        # bitsandbytes 動態量化
  temperature: 0.6
  top_p: 0.9
  repetition_penalty: 1.05
```

---

## 🌐 HTTPS 與服務化（生產部署）

### Nginx 反向代理 + AD CA 憑證

```bash
# 申請 AD CA 簽的憑證（Windows AD CS）
certreq -new aiserverrtx6000.inf aiserverrtx6000.csr
certreq -submit -config "<CA-server>\<CA-name>" aiserverrtx6000.csr aiserverrtx6000.cer
certreq -accept aiserverrtx6000.cer
# 從 certlm.msc 匯出 PFX

# 在 Linux 端轉為 PEM
sudo openssl pkcs12 -in aiserverrtx6000.pfx -clcerts -nokeys -out /etc/nginx/ssl/aiserverrtx6000.crt
sudo openssl pkcs12 -in aiserverrtx6000.pfx -nocerts -nodes -out /etc/nginx/ssl/aiserverrtx6000.key
```

Nginx 設定見 `deploy/nginx-yayan.conf`。

### systemd 服務化

```bash
sudo cp deploy/yayan.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yayan
sudo systemctl start yayan
sudo journalctl -u yayan -f
```

---

## 🚧 已知限制

- **Dolphin LID 對近親語言區分能力一般**：閩南 vs 國語、客家 vs 國語可能誤判
- **短段（< 1 秒）無法做 LID**：直接走鄰近投票
- **第 6 位說話人之後合併到 E**：罕見場景
- **Turing 架構限制**：不支援 vLLM / autoawq triton 路徑，固定走 transformers + bitsandbytes NF4

---

## 🗺️ Roadmap

### v4.7（規劃中）

- [ ] LID 升級到 Whisper-large-v3 detect_language API + VoxLingua107 ensemble
- [ ] 滑動窗口 LID（前後 1.5 秒 context）
- [ ] 閩南語專用模型（`luigisaetta/whisper-medium-zh-tw`）

### v5.0（未來）

- [ ] LLM 升級到 Qwen3.6-27B GGUF + llama-cpp-python backend
- [ ] Docker 容器化部署
- [ ] 多機分散式（ASR 機器 + LLM 機器分離）
- [ ] Web API（REST/gRPC）給其他應用呼叫

---

## 📜 License

Apache 2.0

第三方模型授權：
- Qwen3-14B：Apache 2.0
- Dolphin-CN-Dialect：Apache 2.0
- Whisper-large-v3：MIT
- VoxLingua107 ECAPA：Apache 2.0
- pyannote 3.1：MIT（含 CC-BY-4.0 embedding）

---

## 🙏 致謝

- **Alibaba Qwen Team** — Qwen3 系列模型
- **DataoceanAI + 清華大學** — Dolphin-CN-Dialect
- **OpenAI** — Whisper
- **SpeechBrain** — VoxLingua107 LID
- **pyannote.audio** — 說話人分離
- **HuggingFace** — Transformers ecosystem

---

## 📞 維護資訊

- 內部部署：`aiserverrtx6000.mib.mil.tw`
- 版本：v4.6.0（2026-05）
- 維護者：[wu840407](https://github.com/wu840407)
