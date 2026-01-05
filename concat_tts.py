#!/usr/bin/env python3
import os
import subprocess
import sys
from typing import Iterable, List, Optional

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


def list_text_files(folder_path: str) -> List[str]:
    """列出資料夾內的 .txt 檔案，依檔名排序。"""
    entries = sorted(os.listdir(folder_path))
    text_files = []
    for name in entries:
        if not name.lower().endswith(".txt"):
            continue
        full_path = os.path.join(folder_path, name)
        if os.path.isfile(full_path):
            text_files.append(full_path)
    return text_files


def select_text_files(text_files: Iterable[str], limit: int) -> List[str]:
    """依序取前 limit 個檔案。"""
    return list(text_files)[:limit]


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


def build_cli_command(text: str) -> List[str]:
    """組出 CLI 指令參數。"""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cli_script = os.path.join(repo_root, "indextts", "cli_v2.py")

    cmd = [
        sys.executable,
        cli_script,
        "--text",
        text,
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


def main() -> None:
    text_files = list_text_files(TEXT_DIR)
    selected_files = select_text_files(text_files, MAX_FILES)
    validate_settings(selected_files)

    raw_text = concat_texts(selected_files)
    if not raw_text:
        raise ValueError("合併後文字為空，請檢查輸入檔案內容。")

    converter = OpenCC("t2s")
    normalized_text = convert_to_simplified(raw_text, converter)

    print(f"已選取 {len(selected_files)} 個檔案，來源資料夾：{TEXT_DIR}")
    cmd = build_cli_command(normalized_text)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
