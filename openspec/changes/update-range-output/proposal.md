# Change: 支援自訂檔案範圍與輸出命名

## Why
目前腳本固定取前 20 個檔案，且輸出檔名無法對應使用者選取的範圍，影響批次流程管理。此外，未補零的檔名在字典序排序下容易造成實際串接順序與預期不一致，且缺乏在 TTS 前的確認清單。

## What Changes
- 新增範圍選取設定（1-based、含起訖），支援指定例如第 21–40 個檔案。
- 若範圍超出可用檔案數，腳本需直接報錯。
- 依選取範圍自動命名輸出檔，固定 3 位數格式並輸出到 `outputs/`（例如 `021-040.wav`）。
- 以檔名前綴數字排序（無前綴者置後），避免未補零造成的排序跳號。
- 在執行 TTS 前列印選取檔案清單，順序與實際串接一致。
- 說明 `RANGE_END` 與 `MAX_FILES` 的優先規則（設定 `RANGE_END` 時忽略 `MAX_FILES`；僅設定 `RANGE_START` 時以 `MAX_FILES` 補足範圍）。

## Impact
- Affected specs: `run-batch-tts`（檔案選取與輸出命名）。
- Affected code: `concat_tts.py`。
