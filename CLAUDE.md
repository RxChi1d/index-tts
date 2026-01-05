<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

此檔案提供 Claude Code (claude.ai/code) 在此儲存庫中工作時的指引。

## 專案概述

IndexTTS2 是一個突破性的自回歸零樣本文字轉語音系統，支援情感表達和持續時間控制。此專案由 Bilibili IndexTeam 開發，提供高品質的語音合成能力，包含聲音克隆、情感控制、多語言支援等功能。

### 主要特色

- **精確持續時間控制**：首個支援精確合成持續時間控制的自回歸 TTS 模型
- **情感與音色解耦**：可獨立控制音色和情感表現
- **多模態情感控制**：支援音訊參考、向量控制、文字描述等多種輸入方式
- **零樣本語音克隆**：使用單一參考音訊即可克隆目標音色
- **高表現力語音合成**：達到 SOTA 級別的情感表現力

### 檔案組織規則

**目錄結構：**

```
index-tts/
├── indextts/              # 核心程式碼目錄
│   ├── BigVGAN/          # BigVGAN 聲碼器實作
│   ├── accel/            # 加速相關程式碼（DeepSpeed、CUDA kernel）
│   ├── gpt/              # GPT 模型實作
│   ├── s2mel/            # 語音到梅爾頻譜轉換
│   ├── utils/            # 工具函式
│   ├── vqvae/            # VQ-VAE 實作
│   ├── infer.py          # IndexTTS1 推論介面
│   └── infer_v2.py       # IndexTTS2 推論介面
├── examples/             # 範例音訊檔案
├── tools/                # 工具腳本
│   ├── i18n/            # 國際化相關工具
│   └── gpu_check.py     # GPU 環境檢查工具
├── tests/                # 測試檔案
├── archive/              # 舊版文件與程式碼
├── docs/                 # 文件目錄
├── assets/               # 專案資源（圖片、影片等）
├── webui.py              # Web 介面啟動檔
├── checkpoints/          # 模型檢查點目錄（需下載）
├── pyproject.toml        # Python 專案設定（uv 管理）
└── README.md             # 專案說明文件
```

**主要模組說明：**

- `indextts/infer_v2.py`：IndexTTS2 主要推論介面，包含 `IndexTTS2` 類別
- `indextts/infer.py`：IndexTTS1 舊版推論介面（向下相容）
- `webui.py`：Gradio 網頁介面，提供視覺化互動操作
- `tools/gpu_check.py`：GPU 環境診斷工具


## 語言規則

**重要：請嚴格遵循以下語言規則**

1. **Claude.md 內容**：使用zh-tw
2. **對話語言**：使用zh-tw
3. **程式碼註解**：使用en
4. **函數/變數命名**：使用en
5. **Git commit 訊息**：使用en
6. **文件字串 (docstrings)**：使用en
7. **專案文檔**：使用zh-tw
8. **其他發布用文件**：使用zh-tw

## 撰寫風格與格式

- **專案文檔、說明文字、文件模板**：遵循 Google 風格。
- **Commit 與 PR 訊息**：遵循 Conventional Commit 格式與 Google 風格。
- **Changelog**：遵循 Keep a Changelog 格式。
- **分支名稱**：遵循 Conventional Branch Naming。

### Commit 撰寫規範

- **格式**：遵循 **Conventional Commits** 規範
- **風格**：Google 風格

#### 格式要求

```
<type>(<scope>): <description>    ← 第一行（50-72 字符）

[optional body]                   ← 詳細說明（72 字符換行）

[optional footer(s)]              ← 破壞性變更、問題參考
```

**重要說明**：
- **第一行**：GitHub 自動生成 release notes 使用
- **內容主體**：複雜變更的詳細解釋（不會出現在 release notes 中）
- **腳註**：破壞性變更和問題參考

### Pull Request 撰寫規範

**重要**：建立 PR 時必須遵循以下規範：

#### PR 標題格式
- 必須遵循約定式提交格式：`<type>(<scope>): <description>`
- 範例：`feat: add async operations with progress callbacks`

#### PR 內容格式
- 參考 `.github/pull_request_template.md` 中的模板
- 包含完整的變更說明、測試資訊、檢查清單

#### PR 標籤
- 根據 PR 標題自動分類（Release Drafter 自動處理）
- 確保選擇正確的變更類型

#### PR 描述要求
- 清楚描述變更內容和原因
- 列出相關的測試項目
- 確認所有檢查清單項目

**模板位置**：`.github/pull_request_template.md`
**風格**：Google 風格

## AI 行為規範
- **絕不假設缺漏的上下文，如有疑問務必提出問題確認。**
- **嚴禁臆造不存在的函式或套件**
- **在程式碼或測試中引用檔案路徑或模組名稱前，務必確認其存在。**
- **除非有明確指示，或任務需求（見 `TASK.md`），**否則**不得刪除或覆蓋現有程式碼。**
- **需要分析或拆解問題，通過 sequential thinking 進行更深度思考**
- **與 GitHub 互動需使用 gh CLI**
- **不准在未經允許的情況下，擅自在任何的文檔、訊息等文字中，包含 AI 編輯器或是 AI 模型的名稱**，例如:
  - Generated with [Claude Code]
  - Co-Authored-By: Claude

## Shell 工具使用指引

⚠️ **重要**：使用以下專業工具替代傳統 Unix 指令（若缺少請安裝）：

| 任務類型 | 必須使用 | 禁止使用 |
|---------|---------|---------|
| 檔案搜尋 | `fd` | `find`, `ls -R` |
| 文字搜尋 | `rg` (ripgrep) | `grep`, `ag` |
| 程式碼結構分析 | `ast-grep` | `grep`, `sed` |
| 互動式選擇 | `fzf` | 手動篩選 |
| 處理 JSON | `jq` | `python -m json.tool` |
| 處理 YAML/XML | `yq` | 手動解析 |
