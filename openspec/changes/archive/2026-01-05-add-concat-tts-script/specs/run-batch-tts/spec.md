## ADDED Requirements
### Requirement: 檔案選取與合併
腳本 MUST 讀取指定資料夾內的 `.txt` 檔案路徑，依檔名排序後取前 20 個（不足 20 個則取全部），移除所有換行字元（包含檔案內與檔案間）後串接成單行文字。

#### Scenario: 少於 20 個檔案
- **WHEN** 資料夾僅有 5 個 `.txt`
- **THEN** 腳本使用這 5 個檔案合併。

### Requirement: 繁轉簡後輸入 CLI
腳本 MUST 將合併後文字轉為簡體中文，再作為輸入呼叫 `indextts/cli_v2.py` 進行 TTS。

#### Scenario: 文字包含繁體
- **WHEN** 合併文字包含繁體中文
- **THEN** 傳入 CLI 的文字為簡體中文。

### Requirement: CLI 輸出完整顯示
腳本 MUST 以不攔截 stdout/stderr 的方式執行 CLI，讓輸出完整顯示在終端。

#### Scenario: CLI 執行輸出
- **WHEN** CLI 執行過程輸出日誌
- **THEN** 終端可直接看到完整輸出。

### Requirement: 以程式內變數設定參數
腳本 MUST 以檔案頂端或主程式內變數設定輸入資料夾、參考音訊與相關參數，不提供命令列介面。

#### Scenario: 調整參數
- **WHEN** 使用者要更換參考音訊或參數
- **THEN** 直接修改腳本變數即可完成設定。
