# Change: 新增合併文字後呼叫 CLI 的 TTS 腳本

## Why
目前缺少一個固定流程的腳本，用於從資料夾挑選文本、合併內容並呼叫 `cli_v2.py` 進行 TTS。

## What Changes
- 新增專案根目錄腳本（暫定 `concat_tts.py`），以程式內變數設定輸入資料夾、參考音訊與 CLI 參數。
- 讀取 `.txt` 檔案、依檔名排序取前 20 個，移除所有換行後合併為單行文字。
- 合併文字繁轉簡後，呼叫 `indextts/cli_v2.py` 進行 TTS，並保留 CLI 輸出。

## Impact
- Affected specs: 新增 `run-batch-tts` 能力規格。
- Affected code: 專案根目錄新增腳本檔案（暫定 `concat_tts.py`）。

## Assumptions / Open Questions
- 「呼叫 COI」推定為「呼叫 CLI」；若不是請更正。
- 若 `.txt` 檔案不足 20 份，預設使用全部檔案（可再調整）。
