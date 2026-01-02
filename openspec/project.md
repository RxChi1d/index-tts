# Project Context

## Purpose
IndexTTS2 is a groundbreaking autoregressive zero-shot text-to-speech system developed by the Bilibili IndexTeam. It aims to provide high-quality speech synthesis with features like:
- **Precise Duration Control**: First autoregressive TTS with precise synthesis duration control.
- **Emotion & Timbre Decoupling**: Independent control over voice timbre and emotional expression.
- **Multimodal Emotion Control**: Supports audio reference, vector control, and text description.
- **Zero-shot Voice Cloning**: Clone target voice with a single reference audio.
- **High Expressiveness**: SOTA-level emotional expressiveness.

## Tech Stack
- **Primary Language**: Python
- **Frameworks**: PyTorch, Gradio (WebUI)
- **Acceleration**: DeepSpeed, CUDA kernels
- **Package Management**: uv
- **Core Components**:
  - BigVGAN (Vocoder)
  - GPT (Autoregressive model)
  - VQ-VAE
  - Transformers

## Project Conventions

### Code Style
- **Language**:
  - Documentation, Commits, PRs: **zh-tw** (Traditional Chinese).
  - Code Comments, Function/Variable Names: **English**.
  - Docstrings: **zh-tw**.
- **Style Guides**:
  - Documentation/Text: Google Style.
  - Python: Google Style (implied for docstrings).

### Architecture Patterns
- **Directory Structure**:
  - `indextts/`: Core library code.
  - `indextts/infer_v2.py`: Main inference interface for IndexTTS2.
  - `webui.py`: Gradio-based Web UI entry point.
  - `tools/`: Utility scripts (i18n, GPU checks).
- **Modularity**: Separation of concerns into `accel`, `gpt`, `s2mel`, `vqvae`, `BigVGAN`.

### Testing Strategy
- **Framework**: `pytest` (inferred from `tests/` directory).
- **Requirements**:
  - Changes must be tested.
  - Untested changes require explicit justification.
  - Refer to `tests/` for existing test cases (e.g., `padding_test.py`, `regression_test.py`).

### Git Workflow
- **Commits**: Conventional Commits format (`<type>(<scope>): <description>`) in **zh-tw**.
- **Branches**: Conventional Branch Naming.
- **Pull Requests**:
  - Must follow the template in `.github/pull_request_template.md`.
  - Titles must follow Conventional Commits.
  - Must include change description, test info, and checklist.

## Domain Context
- **TTS Pipeline**: Text -> Tokens -> GPT (Duration/Prosody) -> Mel Spectrogram -> Vocoder (BigVGAN) -> Audio.
- **Zero-shot Learning**: Capability to synthesize speech for unseen speakers using reference audio.
- **Emotion Control**: Handling emotion vectors and decoupling them from speaker identity vectors.

## Important Constraints
- **Language**: Strict adherence to zh-tw for documentation and English for code is critical.
- **No Hallucination**: Do not assume non-existent paths or libraries; verify before use.
- **Tooling**: Use specific CLI tools (`fd`, `rg`, `gh`, `uv`) as mandated.

## External Dependencies
- **HuggingFace**: Model checkpoints and caches.
- **Pre-trained Models**: BigVGAN, Qwen (likely for text encoding/understanding based on checkpoints), etc.
