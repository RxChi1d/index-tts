#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import tempfile
from typing import Iterable, List, Optional, Tuple

from opencc import OpenCC

# === User-configurable settings ===
TEXT_DIR = "path/to/txts"
SPK_AUDIO_PATH = "examples/voice_01.wav"
MODEL_DIR = "checkpoints"
CONFIG_PATH = "checkpoints/config.yaml"
DEVICE = None  # e.g. "cuda", "cpu"
FP16 = False
DEEPSPEED = False
CUDA_KERNEL = False
VERBOSE = False
DRY_RUN = False
REPORT = False

EMO_MODE = 0  # 0: speaker, 1: reference audio, 2: vector, 3: text
EMO_AUDIO_PATH = None
EMO_TEXT = ""
EMO_WEIGHT = 0.65
EMO_VECTOR = None  # e.g. [0.1, 0, 0, 0, 0, 0, 0, 0]
USE_RANDOM = False

MAX_TEXT_TOKENS = 120
DO_SAMPLE = True
TOP_P = 0.8
TOP_K = 30
TEMPERATURE = 0.8
LENGTH_PENALTY = 0.0
NUM_BEAMS = 3
REPETITION_PENALTY = 10.0
MAX_MEL_TOKENS = None

MAX_FILES = 20
RANGE_START = None  # 1-based inclusive
RANGE_END = None  # 1-based inclusive
# Note: Set either MAX_FILES (with RANGE_END unset) or RANGE_END (with MAX_FILES ignored).
TEXT_OUTPUT_PATH = None  # When set, must end with .txt
KEEP_TEMP_TEXT = True


LEADING_INT_RE = re.compile(r"^(\d+)")


def _numeric_sort_key(name: str) -> Tuple[int, int, str]:
    """依檔名前綴數字排序，沒有數字則排後面。"""
    match = LEADING_INT_RE.match(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def list_text_files(folder_path: str) -> List[str]:
    """列出資料夾內的 .txt 檔案，依檔名前綴數字排序。"""
    entries = sorted(os.listdir(folder_path), key=_numeric_sort_key)
    text_files = []
    for name in entries:
        if not name.lower().endswith(".txt"):
            continue
        full_path = os.path.join(folder_path, name)
        if os.path.isfile(full_path):
            text_files.append(full_path)
    return text_files


def select_text_files(
    text_files: Iterable[str],
    limit: int,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
) -> List[str]:
    """依序取指定範圍或前 limit 個檔案。"""
    files = list(text_files)
    if range_start is None and range_end is None:
        return files[:limit]

    start = range_start if range_start is not None else 1
    if range_end is not None:
        end = range_end
    else:
        end = start + limit - 1

    if start < 1 or end < 1:
        raise ValueError("範圍起訖必須是正整數。")
    if start > end:
        raise ValueError("範圍起始值不得大於結束值。")
    if start > len(files) or end > len(files):
        raise ValueError("指定範圍超出檔案數量。")

    return files[start - 1 : end]


def read_text_file(path: str) -> str:
    """讀取 UTF-8 文字檔案。"""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def remove_newlines(text: str) -> str:
    """移除所有換行字元（包含 \r 與 \n）。"""
    return text.replace("\r", "").replace("\n", "")


def concat_texts(paths: Iterable[str]) -> str:
    """讀取多個檔案並移除換行後合併。"""
    chunks = []
    for path in paths:
        content = read_text_file(path)
        chunks.append(remove_newlines(content))
    return "".join(chunks)


def convert_to_simplified(text: str, converter: Optional[OpenCC] = None) -> str:
    """將文字繁轉簡。"""
    if converter is None:
        converter = OpenCC("t2s")
    return converter.convert(text)


def build_cli_command(text_file_path: str) -> List[str]:
    """組出 CLI 指令參數。"""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cli_script = os.path.join(repo_root, "indextts", "cli_v2.py")

    cmd = [
        sys.executable,
        cli_script,
        "--text-file",
        text_file_path,
        "--spk-audio",
        SPK_AUDIO_PATH,
        "--model-dir",
        MODEL_DIR,
        "--config",
        CONFIG_PATH,
        "--emo-mode",
        str(EMO_MODE),
        "--emo-weight",
        str(EMO_WEIGHT),
        "--max-text-tokens",
        str(MAX_TEXT_TOKENS),
        "--top-p",
        str(TOP_P),
        "--top-k",
        str(TOP_K),
        "--temperature",
        str(TEMPERATURE),
        "--length-penalty",
        str(LENGTH_PENALTY),
        "--num-beams",
        str(NUM_BEAMS),
        "--repetition-penalty",
        str(REPETITION_PENALTY),
    ]

    if DEVICE:
        cmd.extend(["--device", DEVICE])
    if FP16:
        cmd.append("--fp16")
    if DEEPSPEED:
        cmd.append("--deepspeed")
    if CUDA_KERNEL:
        cmd.append("--cuda-kernel")
    if VERBOSE:
        cmd.append("--verbose")
    if DRY_RUN:
        cmd.append("--dry-run")
    if REPORT:
        cmd.append("--report")
    if USE_RANDOM:
        cmd.append("--use-random")

    if EMO_MODE == 1 and EMO_AUDIO_PATH:
        cmd.extend(["--emo-audio", EMO_AUDIO_PATH])
    if EMO_MODE == 2 and EMO_VECTOR:
        cmd.append("--emo-vector")
        cmd.extend([str(value) for value in EMO_VECTOR])
    if EMO_MODE == 3 and EMO_TEXT:
        cmd.extend(["--emo-text", EMO_TEXT])

    if DO_SAMPLE:
        cmd.append("--do-sample")
    else:
        cmd.append("--no-do-sample")

    if MAX_MEL_TOKENS is not None:
        cmd.extend(["--max-mel-tokens", str(MAX_MEL_TOKENS)])

    return cmd


def validate_settings(selected_files: List[str]) -> None:
    """檢查輸入設定是否完整。"""
    if not os.path.isdir(TEXT_DIR):
        raise ValueError("TEXT_DIR 不是有效的資料夾。")
    if not selected_files:
        raise ValueError("未找到可用的 .txt 檔案。")
    if not SPK_AUDIO_PATH:
        raise ValueError("請設定 SPK_AUDIO_PATH。")
    if not os.path.exists(SPK_AUDIO_PATH):
        raise ValueError("SPK_AUDIO_PATH 檔案不存在。")
    if EMO_MODE == 1 and not EMO_AUDIO_PATH:
        raise ValueError("EMO_MODE=1 需要設定 EMO_AUDIO_PATH。")
    if EMO_MODE == 2:
        if not EMO_VECTOR or len(EMO_VECTOR) != 8:
            raise ValueError("EMO_MODE=2 需要設定 8 維 EMO_VECTOR。")
    if TEXT_OUTPUT_PATH is not None and not TEXT_OUTPUT_PATH.endswith(".txt"):
        raise ValueError("TEXT_OUTPUT_PATH 必須是 .txt 檔案。")


def write_text_file(text: str, output_path: Optional[str] = None) -> Tuple[str, bool]:
    """寫入文字到檔案，回傳路徑與是否為暫存檔。"""
    final_output_path = output_path or TEXT_OUTPUT_PATH
    if final_output_path:
        os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
        with open(final_output_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return final_output_path, False

    temp_file = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    )
    temp_file.write(text)
    temp_file.close()
    return temp_file.name, True


def main() -> None:
    text_files = list_text_files(TEXT_DIR)
    selected_files = select_text_files(
        text_files,
        MAX_FILES,
        range_start=RANGE_START,
        range_end=RANGE_END,
    )
    validate_settings(selected_files)

    print("選取檔案清單：")
    for path in selected_files:
        print(f"- {os.path.basename(path)}")

    text_output_path = None
    if TEXT_OUTPUT_PATH is None and (RANGE_START is not None or RANGE_END is not None):
        range_start = RANGE_START if RANGE_START is not None else 1
        range_end = (
            RANGE_END
            if RANGE_END is not None
            else range_start + MAX_FILES - 1
        )
        repo_root = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(repo_root, "outputs")
        base_name = f"{range_start:03d}-{range_end:03d}.txt"
        text_output_path = os.path.join(output_dir, base_name)

    raw_text = concat_texts(selected_files)
    if not raw_text:
        raise ValueError("合併後文字為空，請檢查輸入檔案內容。")

    converter = OpenCC("t2s")
    normalized_text = convert_to_simplified(raw_text, converter)

    print(f"已選取 {len(selected_files)} 個檔案，來源資料夾：{TEXT_DIR}")
    text_file_path, is_temp = write_text_file(normalized_text, text_output_path)
    cmd = build_cli_command(text_file_path)
    subprocess.run(cmd, check=True)
    if is_temp and not KEEP_TEMP_TEXT:
        os.remove(text_file_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
