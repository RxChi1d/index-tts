# IndexTTS2 API 使用指南

本文件說明如何使用 IndexTTS2 FastAPI 服務進行語音合成。

## 安裝

### 1. 安裝 API 依賴

```bash
uv sync --extra api
```

這將安裝以下套件：
- `fastapi>=0.116.2` - Web 框架
- `uvicorn[standard]>=0.35.0` - ASGI 伺服器
- `python-multipart>=0.0.20` - 檔案上傳支援
- `aiofiles>=24.1.0` - 非同步檔案操作
- `aiohttp>=3.13.2` - 非同步 HTTP 客戶端

## 啟動服務

### 基本啟動

```bash
uv run api_v2.py
```

### 使用 FP16 加速（推薦）

```bash
uv run api_v2.py --fp16
```

### 完整參數範例

```bash
uv run api_v2.py \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --output-dir outputs \
  --link-expiry-days 7 \
  --host 0.0.0.0 \
  --port 8000 \
  --fp16 \
  --cuda-kernel
```

### 啟動參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--model-dir` | 模型檢查點目錄 | `checkpoints` |
| `--config` | 設定檔路徑 | `checkpoints/config.yaml` |
| `--output-dir` | 輸出目錄 | `outputs` |
| `--link-expiry-days` | 下載連結有效期限（天） | `7` |
| `--host` | 伺服器位址 | `0.0.0.0` |
| `--port` | 伺服器埠號 | `8000` |
| `--fp16` | 使用 FP16 推論 | `False` |
| `--deepspeed` | 使用 DeepSpeed | `False` |
| `--cuda-kernel` | 使用 CUDA kernel | `False` |
| `--device` | 指定裝置 (cpu/cuda/mps/xpu) | 自動偵測 |

## API 端點

服務啟動後，訪問 http://localhost:8000/docs 查看完整的 Swagger API 文件。

### 1. 健康檢查

**端點**: `GET /health`

**範例**:
```bash
curl http://localhost:8000/health
```

**回應**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. API 資訊

**端點**: `GET /api/v2/info`

**範例**:
```bash
curl http://localhost:8000/api/v2/info
```

**回應**:
```json
{
  "version": "2.0.0",
  "api_version": "v2",
  "supported_emotion_modes": [0, 1, 2, 3],
  "max_text_tokens_limit": 500,
  "output_directory": "/path/to/outputs",
  "link_expiry_days": 7
}
```

### 3. 同步 TTS 合成

**端點**: `POST /api/v2/tts/sync`

#### 3.1 回應格式

預設情況下，同步端點返回 JSON 格式：

```json
{
  "download": {
    "download_url": "/api/v2/download/550e8400-...",
    "link_id": "550e8400-...",
    "expires_at": "2026-01-11T10:00:00Z",
    "file_size": 153600,
    "file_format": "wav",
    "duration": 2.5
  },
  "created_at": "2026-01-04T10:00:00Z",
  "completed_at": "2026-01-04T10:00:02Z"
}
```

若要直接下載檔案，使用 `return_file=true` 參數。

#### 3.2 音訊輸入方式

**支援音訊輸入方式**:
1. 檔案上傳 (`speaker_audio`): 客戶端上傳音訊檔案
2. 本地路徑 (`speaker_audio_path`): 指定伺服器端音訊檔案路徑

**注意**:
- 二選一，不能同時指定
- 出於安全考量，不支援 URL 輸入

**範例 1: 預設 JSON 回應（推薦）**

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav"
```

**回應**:
```json
{
  "download": {
    "download_url": "/api/v2/download/550e8400-...",
    "link_id": "550e8400-...",
    "expires_at": "2026-01-11T10:00:00Z",
    "file_size": 153600,
    "file_format": "wav",
    "duration": 2.5
  },
  "created_at": "2026-01-04T10:00:00Z",
  "completed_at": "2026-01-04T10:00:02Z"
}
```

然後使用 download_url 下載檔案：
```bash
curl http://localhost:8000/api/v2/download/550e8400-... --output output.wav
```

**範例 2: 直接下載音訊檔案**

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "return_file=true" \
  --output output.wav
```

**範例 3: 上傳音訊檔案**

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio=@examples/voice_01.wav"
```

### 4. 非同步 TTS 合成

非同步模式適合長文本或不穩定網路環境。

**步驟 1: 提交任務**

```bash
curl -X POST http://localhost:8000/api/v2/tts/async \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav"
```

**回應**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

**步驟 2: 查詢任務狀態**

```bash
curl http://localhost:8000/api/v2/tts/status/550e8400-e29b-41d4-a716-446655440000
```

**回應（處理中）**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "created_at": "2026-01-03T10:00:00",
  "completed_at": null,
  "download": null,
  "error": null
}
```

**回應（完成時）**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2026-01-03T10:00:00",
  "completed_at": "2026-01-03T10:00:30",
  "download": {
    "download_url": "/api/v2/download/abc123...",
    "link_id": "abc123...",
    "expires_at": "2026-01-10T10:00:30",
    "file_size": 153600,
    "file_format": "wav",
    "duration": 2.5
  },
  "error": null
}
```

**步驟 3: 使用下載連結取得結果**

```bash
# 方法 1: 使用狀態回應中的 download_url（推薦）
curl http://localhost:8000/api/v2/download/abc123... --output result.wav

# 方法 2: 直接使用 task_id（舊方法，仍支援）
curl http://localhost:8000/api/v2/tts/result/550e8400-e29b-41d4-a716-446655440000 \
  --output result.wav
```

## 情感控制

### 模式 0: 使用音色參考音訊的情感（預設）

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "emotion_mode=0" \
  --output output.wav
```

### 模式 1: 使用情感參考音訊

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "emotion_mode=1" \
  -F "emotion_audio_path=examples/emo_sad.wav" \
  -F "emotion_weight=0.8" \
  --output output.wav
```

### 模式 2: 使用情感向量

情感向量包含 8 個數值：`[HAPPY, ANGRY, SAD, AFRAID, DISGUSTED, MELANCHOLIC, SURPRISED, CALM]`

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "emotion_mode=2" \
  -F 'emotion_vector=[0.1, 0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.2]' \
  --output output.wav
```

### 模式 3: 使用情感描述文字

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "emotion_mode=3" \
  -F "emotion_text=委屈巴巴" \
  --output output.wav
```

## 進階參數

### GPT2 採樣參數

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "do_sample=true" \
  -F "temperature=1.2" \
  -F "top_p=0.9" \
  -F "top_k=50" \
  --output output.wav
```

### Beam Search（不使用採樣）

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=你好，我是 IndexTTS2 語音合成系統。" \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "do_sample=false" \
  -F "num_beams=5" \
  --output output.wav
```

### 文本分段控制

```bash
curl -X POST http://localhost:8000/api/v2/tts/sync \
  -F "text=這是一段很長的文字..." \
  -F "speaker_audio_path=examples/voice_01.wav" \
  -F "max_text_tokens=150" \
  --output output.wav
```

### 完整參數列表

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `text` | string | 必填 | 要合成的文字 |
| `speaker_audio` | file | - | 上傳音色參考音訊檔案 |
| `speaker_audio_path` | string | - | 或提供伺服器端音訊檔案路徑 |
| | | | **（二選一，必填）** |
| `emotion_mode` | int | 0 | 情感控制模式 (0-3) |
| `emotion_audio` | file | - | 上傳情感參考音訊檔案（mode=1 可選） |
| `emotion_audio_path` | string | - | 或提供伺服器端音訊檔案路徑 |
| | | | **（二選一，mode=1 必填）** |
| `emotion_text` | string | "" | 情感描述文字（mode=3） |
| `emotion_weight` | float | 0.65 | 情感權重 (0.0-1.0) |
| `emotion_vector` | array | null | 8 元素情感向量（mode=2） |
| `use_random` | bool | false | 啟用隨機採樣 |
| `max_text_tokens` | int | 120 | 每段最大 token 數 |
| `do_sample` | bool | true | 啟用採樣 |
| `top_p` | float | 0.8 | Top-p 採樣 |
| `top_k` | int | 30 | Top-k 採樣 |
| `temperature` | float | 0.8 | 採樣溫度 |
| `length_penalty` | float | 0.0 | 長度懲罰 |
| `num_beams` | int | 3 | Beam search 寬度 |
| `repetition_penalty` | float | 10.0 | 重複懲罰 |
| `max_mel_tokens` | int | null | 最大 mel token 數 |
| `return_file` | bool | false | 直接返回音訊檔案而非 JSON |

## 測試

### 執行測試腳本

確保 API 服務已啟動：

```bash
# Terminal 1: 啟動 API 服務
uv run api_v2.py --fp16

# Terminal 2: 執行測試
uv run test_api_v2.py
```

測試腳本會自動：
- 檢查伺服器狀態
- 測試所有端點
- 測試情感控制模式
- 測試錯誤處理
- 產生測試報告於 `test_outputs/test_report.json`

## Python 客戶端範例

### 同步請求

```python
import requests

url = "http://localhost:8000/api/v2/tts/sync"

data = {
    "text": "你好，我是 IndexTTS2 語音合成系統。",
    "speaker_audio_path": "examples/voice_01.wav",
}

response = requests.post(url, data=data)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("合成成功！")
else:
    print(f"錯誤: {response.status_code}")
    print(response.text)
```

### 非同步請求

```python
import requests
import time

# 提交任務
url = "http://localhost:8000/api/v2/tts/async"
data = {
    "text": "你好，我是 IndexTTS2 語音合成系統。",
    "speaker_audio_path": "examples/voice_01.wav",
}

response = requests.post(url, data=data)
task_id = response.json()["task_id"]
print(f"任務 ID: {task_id}")

# 輪詢狀態
while True:
    status_response = requests.get(f"http://localhost:8000/api/v2/tts/status/{task_id}")
    status = status_response.json()["status"]
    print(f"狀態: {status}")

    if status == "completed":
        break
    elif status == "failed":
        print("合成失敗！")
        exit(1)

    time.sleep(2)

# 取得結果
result_response = requests.get(f"http://localhost:8000/api/v2/tts/result/{task_id}")
with open("output.wav", "wb") as f:
    f.write(result_response.content)
print("合成成功！")
```

## 錯誤處理

API 使用標準 HTTP 狀態碼和結構化錯誤回應：

```json
{
  "error": {
    "code": "HTTP_400",
    "message": "emotion_vector must contain exactly 8 float values",
    "details": null
  }
}
```

常見錯誤碼：
- `400`: 參數驗證錯誤
- `404`: 任務或資源不存在
- `408`: 請求超時
- `410`: 結果已過期
- `422`: 無法處理的實體（Pydantic 驗證失敗）
- `500`: 伺服器內部錯誤
- `503`: 服務暫時無法使用（模型未載入）

## 效能建議

1. **使用 FP16**: 啟動時加上 `--fp16` 參數可顯著加速推論
2. **使用 CUDA kernel**: 加上 `--cuda-kernel` 參數使用 BigVGAN 自訂 CUDA kernel
3. **長文本使用非同步端點**: 避免 HTTP 超時問題
4. **調整 batch size**: 可透過環境變數或設定檔調整
5. **本地音訊**: 優先使用 `speaker_audio_type=path` 避免上傳延遲

## 注意事項

- 伺服器重啟後，所有非同步任務資訊將遺失（記憶體儲存）
- 下載連結預設 7 天後失效
- 輸出檔案不會自動清理，需手動管理
- 繁體中文文字會自動轉換為簡體中文
- 單次請求超時設定為 2 分鐘（同步端點）

## 疑難排解

### 模型載入失敗

確認模型檔案存在：
```bash
ls checkpoints/
# 應包含: bpe.model, gpt.pth, config.yaml, s2mel.pth, wav2vec2bert_stats.pt
```

### API 服務無法啟動

檢查埠號是否被佔用：
```bash
lsof -i :8000
```

更換埠號：
```bash
uv run api_v2.py --port 8001
```

### GPU 記憶體不足

使用 CPU 模式：
```bash
uv run api_v2.py --device cpu
```

或使用更小的模型設定檔。

## 更多資訊

- API 文件: http://localhost:8000/docs
- ReDoc 文件: http://localhost:8000/redoc
- 專案 GitHub: https://github.com/index-tts/index-tts
