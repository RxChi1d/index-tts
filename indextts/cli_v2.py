import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime

from opencc import OpenCC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def build_parser():
    """建立 CLI 參數解析器。"""
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Command Line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text to synthesize")
    input_group.add_argument("--text-file", type=str, help="Path to a .txt file")
    input_group.add_argument(
        "--text-dir", type=str, help="Path to a folder of .txt files"
    )

    parser.add_argument(
        "--spk-audio",
        type=str,
        required=True,
        help="Path to speaker prompt audio (wav)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Model checkpoints directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Use FP16 for inference"
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        default=False,
        help="Use DeepSpeed if available",
    )
    parser.add_argument(
        "--cuda-kernel",
        action="store_true",
        default=False,
        help="Use CUDA kernel if available",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (cpu, cuda, mps, xpu)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose mode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview outputs without synthesis",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Write batch report to outputs/",
    )

    parser.add_argument(
        "--emo-mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Emotion control mode (0: speaker, 1: reference audio, 2: vector, 3: text)",
    )
    parser.add_argument(
        "--emo-audio", type=str, default=None, help="Emotion reference audio path"
    )
    parser.add_argument(
        "--emo-text", type=str, default="", help="Emotion description text"
    )
    parser.add_argument("--emo-weight", type=float, default=0.65, help="Emotion weight")
    parser.add_argument(
        "--emo-vector",
        type=float,
        nargs=8,
        default=None,
        metavar=(
            "HAPPY",
            "ANGRY",
            "SAD",
            "AFRAID",
            "DISGUSTED",
            "MELANCHOLIC",
            "SURPRISED",
            "CALM",
        ),
        help="Emotion vector with 8 values",
    )
    parser.add_argument(
        "--use-random",
        action="store_true",
        default=False,
        help="Enable random sampling",
    )

    parser.add_argument(
        "--max-text-tokens", type=int, default=120, help="Max tokens per segment"
    )

    parser.add_argument(
        "--do-sample", dest="do_sample", action="store_true", default=True
    )
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--max-mel-tokens", type=int, default=None)

    return parser


def ensure_model_files(model_dir, config_path):
    """確認模型與必要檔案存在。"""
    if not os.path.exists(model_dir):
        print(
            f"Model directory {model_dir} does not exist. Please download the model first."
        )
        sys.exit(1)

    required_files = [
        "bpe.model",
        "gpt.pth",
        "config.yaml",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
    ]
    for filename in required_files:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            print(f"Required file {file_path} does not exist. Please download it.")
            sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Config file {config_path} does not exist.")
        sys.exit(1)


def contains_cjk(text):
    """判斷字串是否包含中文。"""
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def normalize_text(text, converter):
    """繁轉簡處理，無中文則直接回傳。"""
    if contains_cjk(text):
        return converter.convert(text)
    return text


def build_output_path(base_name, output_dir):
    """產生輸出路徑並處理同名衝突。"""
    os.makedirs(output_dir, exist_ok=True)
    candidate = os.path.join(output_dir, f"{base_name}.wav")
    if not os.path.exists(candidate):
        return candidate, base_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"{base_name}_{timestamp}"
    return os.path.join(output_dir, f"{final_name}.wav"), final_name


def read_text_file(path):
    """讀取 UTF-8 文字檔案。"""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def collect_folder_items(folder_path):
    """取得資料夾中的 .txt 檔案清單與跳過項目。"""
    entries = sorted(os.listdir(folder_path))
    text_files = []
    skipped = []
    for name in entries:
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path):
            skipped.append(name)
            continue
        if not name.lower().endswith(".txt"):
            skipped.append(name)
            continue
        text_files.append(full_path)
    return text_files, skipped


def validate_inputs(args):
    """驗證輸入參數與檔案存在性。"""
    if not os.path.exists(args.spk_audio):
        print(f"Speaker audio file {args.spk_audio} does not exist.")
        sys.exit(1)

    if args.text is not None and len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        sys.exit(1)

    if args.text_file is not None:
        if not os.path.exists(args.text_file):
            print(f"Text file {args.text_file} does not exist.")
            sys.exit(1)
        if not args.text_file.lower().endswith(".txt"):
            print(f"ERROR: Text file {args.text_file} must be a .txt file.")
            sys.exit(1)

    if args.text_dir is not None:
        if not os.path.isdir(args.text_dir):
            print(f"Text folder {args.text_dir} does not exist or is not a directory.")
            sys.exit(1)

    if args.emo_mode == 1 and not args.emo_audio:
        print("ERROR: --emo-audio is required when --emo-mode=1.")
        sys.exit(1)
    if args.emo_audio is not None and not os.path.exists(args.emo_audio):
        print(f"Emotion audio file {args.emo_audio} does not exist.")
        sys.exit(1)

    if args.emo_mode == 2 and args.emo_vector is None:
        print("ERROR: --emo-vector is required when --emo-mode=2.")
        sys.exit(1)


def build_generation_kwargs(args, tts):
    """建立推論參數。"""
    max_mel_tokens = args.max_mel_tokens
    if max_mel_tokens is None:
        max_mel_tokens = int(tts.cfg.gpt.max_mel_tokens)

    return {
        "do_sample": bool(args.do_sample),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k) if int(args.top_k) > 0 else None,
        "temperature": float(args.temperature),
        "length_penalty": float(args.length_penalty),
        "num_beams": int(args.num_beams),
        "repetition_penalty": float(args.repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }


def resolve_emotion_inputs(args, tts):
    """依情感控制模式整理輸入。"""
    emo_audio_prompt = args.emo_audio
    emo_vector = None
    use_emo_text = False
    emo_text = None

    if args.emo_mode == 0:
        emo_audio_prompt = None
    elif args.emo_mode == 1:
        pass
    elif args.emo_mode == 2:
        emo_vector = tts.normalize_emo_vec(list(args.emo_vector), apply_bias=True)
    elif args.emo_mode == 3:
        use_emo_text = True
        emo_text = args.emo_text if args.emo_text != "" else None

    return emo_audio_prompt, emo_vector, use_emo_text, emo_text


def infer_single(tts, text, output_path, args):
    """執行單次推論。"""
    emo_audio_prompt, emo_vector, use_emo_text, emo_text = resolve_emotion_inputs(
        args, tts
    )
    generation_kwargs = build_generation_kwargs(args, tts)

    return tts.infer(
        spk_audio_prompt=args.spk_audio,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_audio_prompt,
        emo_alpha=float(args.emo_weight),
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=bool(args.use_random),
        verbose=bool(args.verbose),
        max_text_tokens_per_segment=int(args.max_text_tokens),
        **generation_kwargs,
    )


def run_text_mode(tts, converter, args, output_dir, dry_run):
    """處理直接文字輸入。"""
    raw_text = args.text.strip()
    normalized_text = normalize_text(raw_text, converter)
    base_name = f"spk_{int(time.time())}"
    output_path, final_name = build_output_path(base_name, output_dir)
    if not dry_run:
        infer_single(tts, normalized_text, output_path, args)
    mapping = [("<inline_text>", output_path)]
    return [output_path], [], [], mapping, final_name


def run_file_mode(tts, converter, args, output_dir, dry_run):
    """處理單一檔案輸入。"""
    try:
        raw_text = read_text_file(args.text_file).strip()
    except Exception as exc:
        print(f"ERROR: Failed to read file {args.text_file}: {exc}")
        sys.exit(1)

    if len(raw_text) == 0:
        print(f"ERROR: Text file {args.text_file} is empty.")
        sys.exit(1)

    normalized_text = normalize_text(raw_text, converter)
    base_name = os.path.splitext(os.path.basename(args.text_file))[0]
    output_path, final_name = build_output_path(base_name, output_dir)
    if not dry_run:
        infer_single(tts, normalized_text, output_path, args)
    mapping = [(args.text_file, output_path)]
    return [output_path], [], [], mapping, final_name


def run_folder_mode(tts, converter, args, output_dir, dry_run):
    """處理資料夾批次輸入。"""
    text_files, skipped = collect_folder_items(args.text_dir)
    outputs = []
    errors = []
    mapping = []

    if len(text_files) == 0:
        print("No .txt files found. Processed 0 files.")
        return outputs, skipped, errors, mapping, None

    total_files = len(text_files)
    print(f"\n{'='*60}")
    print(f"Starting batch processing: {total_files} file(s) found")
    print(f"{'='*60}\n")

    for idx, text_path in enumerate(text_files, 1):
        file_name = os.path.basename(text_path)
        print(f"\n[{idx}/{total_files}] Processing: {file_name}")
        print("-" * 60)

        try:
            raw_text = read_text_file(text_path).strip()
        except Exception as exc:
            print(f"  ✗ ERROR: Failed to read file - {exc}")
            skipped.append(os.path.basename(text_path))
            errors.append(f"{text_path}: {exc}")
            continue

        if len(raw_text) == 0:
            print(f"  ✗ SKIPPED: File is empty")
            skipped.append(os.path.basename(text_path))
            continue

        normalized_text = normalize_text(raw_text, converter)
        base_name = os.path.splitext(os.path.basename(text_path))[0]
        output_path, _ = build_output_path(base_name, output_dir)

        if not dry_run:
            print(f"  → Synthesizing...")
            infer_single(tts, normalized_text, output_path, args)
            print(f"  ✓ SUCCESS: Saved to {output_path}")
        else:
            print(f"  → DRY RUN: Would save to {output_path}")

        outputs.append(output_path)
        mapping.append((text_path, output_path))

    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"{'='*60}\n")

    return outputs, skipped, errors, mapping, None


def print_batch_summary(outputs, skipped, errors, mapping):
    """輸出批次摘要資訊。"""
    print("Batch summary:")
    print(f"- processed: {len(outputs)}")
    print(f"- skipped: {len(skipped)}")
    if mapping:
        print("- output mapping:")
        for source_path, output_path in mapping:
            print(f"  - {source_path} -> {output_path}")
    if skipped:
        print(f"- skipped files: {', '.join(skipped)}")
    if errors:
        print(f"- errors: {len(errors)}")
        for err in errors:
            print(f"  - {err}")


def write_report(report_path, outputs, skipped, errors, mapping):
    """輸出報告檔至 outputs/。"""
    report = {
        "processed": len(outputs),
        "outputs": outputs,
        "skipped": skipped,
        "errors": errors,
        "mapping": [{"source": source, "output": output} for source, output in mapping],
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report_path


def main():
    parser = build_parser()
    args = parser.parse_args()

    validate_inputs(args)
    converter = OpenCC("t2s")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "outputs")
    report_path = None
    if args.report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"batch_report_{timestamp}.json")

    if args.dry_run:
        print("DRY RUN: no synthesis will be performed.")
        if args.text is not None:
            outputs, skipped, errors, mapping, _ = run_text_mode(
                None, converter, args, output_dir, True
            )
        elif args.text_file is not None:
            outputs, skipped, errors, mapping, _ = run_file_mode(
                None, converter, args, output_dir, True
            )
        else:
            outputs, skipped, errors, mapping, _ = run_folder_mode(
                None, converter, args, output_dir, True
            )
        print_batch_summary(outputs, skipped, errors, mapping)
        if report_path:
            write_report(report_path, outputs, skipped, errors, mapping)
            print(f"Report: {report_path}")
        return

    ensure_model_files(args.model_dir, args.config)
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_fp16=args.fp16,
        device=args.device,
        use_cuda_kernel=args.cuda_kernel,
        use_deepspeed=args.deepspeed,
    )

    if args.text is not None:
        outputs, skipped, errors, mapping, _ = run_text_mode(
            tts, converter, args, output_dir, False
        )
        print(f"Output: {outputs[0]}")
        print_batch_summary(outputs, skipped, errors, mapping)
        if report_path:
            write_report(report_path, outputs, skipped, errors, mapping)
            print(f"Report: {report_path}")
        return

    if args.text_file is not None:
        outputs, skipped, errors, mapping, _ = run_file_mode(
            tts, converter, args, output_dir, False
        )
        print(f"Output: {outputs[0]}")
        print_batch_summary(outputs, skipped, errors, mapping)
        if report_path:
            write_report(report_path, outputs, skipped, errors, mapping)
            print(f"Report: {report_path}")
        return

    outputs, skipped, errors, mapping, _ = run_folder_mode(
        tts, converter, args, output_dir, False
    )
    print_batch_summary(outputs, skipped, errors or [], mapping)
    if report_path:
        write_report(report_path, outputs, skipped, errors or [], mapping)
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
