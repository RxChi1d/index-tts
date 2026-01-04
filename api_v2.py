#!/usr/bin/env python3
"""
IndexTTS2 FastAPI Service

Provides RESTful API endpoints for IndexTTS2 text-to-speech synthesis.
Supports synchronous and asynchronous operations with flexible audio input/output options.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import signal
import sys
import tempfile
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiohttp
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from opencc import OpenCC
from pydantic import BaseModel, Field, field_validator, ValidationError

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global state
app_state = {
    "tts": None,
    "converter": None,
    "output_dir": None,
    "link_expiry_days": 7,
    "tasks": {},
    "download_links": {},
    "shutdown_event": asyncio.Event(),
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===== Enums =====
class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"





# ===== Pydantic Models =====
class ErrorDetail(BaseModel):
    """Structured error response."""

    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail


class TaskInfo(BaseModel):
    """Task information for async processing."""

    task_id: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_path: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None


class TTSRequest(BaseModel):
    """TTS synthesis request model aligned with cli_v2 parameters."""

    text: str = Field(..., min_length=1, description="Text to synthesize")

    # Emotion control
    emotion_mode: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Emotion control mode: 0=speaker, 1=reference audio, 2=vector, 3=text",
    )
    emotion_text: str = Field(
        default="", description="Emotion description text for mode 3"
    )
    emotion_weight: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Emotion weight/alpha"
    )
    emotion_vector: Optional[list[float]] = Field(
        default=None, description="8-element emotion vector for mode 2"
    )
    use_random: bool = Field(default=False, description="Enable random sampling")

    # Text segmentation
    max_text_tokens: int = Field(
        default=120, ge=1, le=500, description="Max tokens per segment"
    )

    # GPT2 sampling parameters
    do_sample: bool = Field(default=True, description="Enable sampling")
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=30, ge=0, description="Top-k sampling (0=disabled)")
    temperature: float = Field(default=0.8, gt=0.0, description="Sampling temperature")
    length_penalty: float = Field(default=0.0, description="Length penalty")
    num_beams: int = Field(default=3, ge=1, description="Beam search width")
    repetition_penalty: float = Field(
        default=10.0, ge=1.0, description="Repetition penalty"
    )
    max_mel_tokens: Optional[int] = Field(
        default=None, ge=1, description="Max mel tokens (None=use config default)"
    )

    @field_validator("emotion_vector")
    @classmethod
    def validate_emotion_vector(cls, v):
        """Validate emotion vector has exactly 8 elements."""
        if v is not None and len(v) != 8:
            raise ValueError("emotion_vector must contain exactly 8 float values")
        return v


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class APIInfoResponse(BaseModel):
    """API information response."""

    version: str
    api_version: str
    supported_emotion_modes: list[int]
    max_text_tokens_limit: int
    output_directory: str
    link_expiry_days: int


class TaskStatusResponse(BaseModel):
    """Task status response."""

    task_id: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class AsyncTaskResponse(BaseModel):
    """Async task submission response."""

    task_id: str
    status: TaskStatus


class DownloadInfo(BaseModel):
    """Download link and file metadata."""

    download_url: str = Field(..., description="Download URL path (relative)")
    link_id: str = Field(..., description="Unique link identifier")
    expires_at: datetime = Field(..., description="Link expiration timestamp")
    file_size: int = Field(..., description="File size in bytes")
    file_format: str = Field(default="wav", description="Audio file format")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")


class SyncTTSResponse(BaseModel):
    """Synchronous TTS response with download information."""

    download: DownloadInfo = Field(..., description="Download information")
    created_at: datetime = Field(..., description="Request creation timestamp")
    completed_at: datetime = Field(..., description="Synthesis completion timestamp")


class EnhancedTaskStatusResponse(BaseModel):
    """Enhanced task status response with download information."""

    task_id: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    download: Optional[DownloadInfo] = Field(
        None,
        description="Download information (only available when status is completed)"
    )
    error: Optional[str] = None


# ===== Utility Functions =====
def contains_cjk(text: str) -> bool:
    """Check if text contains Chinese characters."""
    import re

    return re.search(r"[\u4e00-\u9fff]", text) is not None


def normalize_text(text: str, converter: OpenCC) -> str:
    """Convert Traditional Chinese to Simplified Chinese."""
    if contains_cjk(text):
        return converter.convert(text)
    return text


def validate_audio_file(file_path: str) -> bool:
    """Validate audio file exists and has valid extension."""
    if not os.path.exists(file_path):
        return False
    valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return Path(file_path).suffix.lower() in valid_extensions





def build_generation_kwargs(request: TTSRequest, tts) -> dict[str, Any]:
    """Build generation kwargs from request, aligned with cli_v2."""
    max_mel_tokens = request.max_mel_tokens
    if max_mel_tokens is None:
        max_mel_tokens = int(tts.cfg.gpt.max_mel_tokens)

    return {
        "do_sample": bool(request.do_sample),
        "top_p": float(request.top_p),
        "top_k": int(request.top_k) if int(request.top_k) > 0 else None,
        "temperature": float(request.temperature),
        "length_penalty": float(request.length_penalty),
        "num_beams": int(request.num_beams),
        "repetition_penalty": float(request.repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }


def resolve_emotion_inputs(request: TTSRequest, tts) -> tuple:
    """Resolve emotion inputs based on emotion_mode, aligned with cli_v2."""
    emo_audio_prompt = None
    emo_vector = None
    use_emo_text = False
    emo_text = None

    if request.emotion_mode == 0:
        # Use speaker audio emotion
        pass
    elif request.emotion_mode == 1:
        # Use reference audio (will be set by caller)
        pass
    elif request.emotion_mode == 2:
        # Use emotion vector
        if request.emotion_vector is None:
            raise HTTPException(
                status_code=400,
                detail="emotion_vector is required when emotion_mode=2",
            )
        emo_vector = tts.normalize_emo_vec(
            list(request.emotion_vector), apply_bias=True
        )
    elif request.emotion_mode == 3:
        # Use emotion text
        use_emo_text = True
        emo_text = request.emotion_text if request.emotion_text != "" else None

    return emo_audio_prompt, emo_vector, use_emo_text, emo_text


def infer_single(
    tts,
    text: str,
    speaker_audio_path: str,
    emotion_audio_path: Optional[str],
    output_path: str,
    request: TTSRequest,
) -> str:
    """Execute single TTS inference."""
    emo_audio_prompt, emo_vector, use_emo_text, emo_text = resolve_emotion_inputs(
        request, tts
    )

    # Override emo_audio_prompt if emotion_mode == 1
    if request.emotion_mode == 1:
        emo_audio_prompt = emotion_audio_path

    generation_kwargs = build_generation_kwargs(request, tts)

    return tts.infer(
        spk_audio_prompt=speaker_audio_path,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_audio_prompt,
        emo_alpha=float(request.emotion_weight),
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=bool(request.use_random),
        verbose=False,
        max_text_tokens_per_segment=int(request.max_text_tokens),
        **generation_kwargs,
    )


def create_download_link_with_metadata(task_id: str, file_path: str) -> dict[str, Any]:
    """
    Create download link with complete metadata.

    Args:
        task_id: Task identifier
        file_path: Path to result file

    Returns:
        Dictionary with link_id, download_url, expires_at, file_size, file_format, duration
    """
    # Generate unique link ID
    link_id = str(uuid.uuid4())
    download_url = f"/api/v2/download/{link_id}"
    expires_at = datetime.now() + timedelta(days=app_state["link_expiry_days"])

    # Get file metadata
    file_metadata = get_audio_metadata(file_path)

    # Store in download_links with extended metadata
    app_state["download_links"][link_id] = {
        "file_path": file_path,
        "task_id": task_id,
        "expires_at": expires_at,
        "file_size": file_metadata["file_size"],
        "file_format": file_metadata["file_format"],
        "duration": file_metadata["duration"],
    }

    return {
        "link_id": link_id,
        "download_url": download_url,
        "expires_at": expires_at,
        "file_size": file_metadata["file_size"],
        "file_format": file_metadata["file_format"],
        "duration": file_metadata["duration"],
    }


def create_download_link(task_id: str, file_path: str) -> str:
    """
    Create download link (legacy wrapper for backward compatibility).

    Args:
        task_id: Task identifier
        file_path: Path to result file

    Returns:
        Download URL path
    """
    metadata = create_download_link_with_metadata(task_id, file_path)
    return metadata["download_url"]


def cleanup_temp_files(*file_paths: str) -> None:
    """Clean up temporary files."""
    for path in file_paths:
        if path and os.path.exists(path) and path.startswith(tempfile.gettempdir()):
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {path}: {e}")


def get_audio_metadata(file_path: str) -> dict[str, Any]:
    """
    Extract audio file metadata.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with file_size, file_format, duration
    """
    metadata = {
        "file_size": 0,
        "file_format": "wav",
        "duration": None
    }

    try:
        # Get file size
        metadata["file_size"] = os.path.getsize(file_path)

        # Get file format from extension
        file_format = Path(file_path).suffix.lstrip(".").lower()
        if file_format:
            metadata["file_format"] = file_format

        # Get audio duration (for WAV files)
        if file_format == "wav":
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                if rate > 0:
                    metadata["duration"] = frames / float(rate)

    except Exception as e:
        logger.warning(f"Failed to extract metadata from {file_path}: {e}")

    return metadata


# ===== Background Task Processor =====
async def process_async_task(
    task_id: str,
    text: str,
    speaker_audio_path: str,
    emotion_audio_path: Optional[str],
    request: TTSRequest,
    cleanup_paths: list[str],
):
    """Background task processor for async TTS synthesis."""
    try:
        app_state["tasks"][task_id]["status"] = TaskStatus.PROCESSING

        # Generate output path
        output_path = os.path.join(
            app_state["output_dir"],
            f"async_{task_id}_{int(datetime.now().timestamp())}.wav",
        )

        # Perform TTS inference
        tts = app_state["tts"]
        converter = app_state["converter"]
        normalized_text = normalize_text(text, converter)

        result_path = await asyncio.to_thread(
            infer_single,
            tts,
            normalized_text,
            speaker_audio_path,
            emotion_audio_path,
            output_path,
            request,
        )

        # Create download link with metadata
        download_metadata = create_download_link_with_metadata(task_id, result_path)

        # Update task status
        app_state["tasks"][task_id].update(
            {
                "status": TaskStatus.COMPLETED,
                "completed_at": datetime.now(),
                "result_path": result_path,
                "download_url": download_metadata["download_url"],
                "link_id": download_metadata["link_id"],
            }
        )

        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        app_state["tasks"][task_id].update(
            {
                "status": TaskStatus.FAILED,
                "completed_at": datetime.now(),
                "error": str(e),
            }
        )

    finally:
        # Cleanup temporary files
        cleanup_temp_files(*cleanup_paths)


async def cleanup_expired_links():
    """Background task to cleanup expired download links."""
    while not app_state["shutdown_event"].is_set():
        try:
            now = datetime.now()
            expired_links = [
                link_id for link_id, info in app_state["download_links"].items()
                if info["expires_at"] < now
            ]
            for link_id in expired_links:
                del app_state["download_links"][link_id]

            if expired_links:
                logger.info(f"Cleaned up {len(expired_links)} expired download links")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

        await asyncio.sleep(3600)  # Run every hour


# ===== Lifecycle Management =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("IndexTTS2 API starting up...")

    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_links())

    # Model will be loaded by main() before server starts
    yield

    # Shutdown
    logger.info("IndexTTS2 API shutting down...")
    app_state["shutdown_event"].set()

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# ===== FastAPI Application =====
app = FastAPI(
    title="IndexTTS2 API",
    description="RESTful API for IndexTTS2 text-to-speech synthesis",
    version="2.0.0",
    lifespan=lifespan,
)


# ===== API Endpoints =====
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = app_state["tts"] is not None

    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False},
        )

    return {"status": "healthy", "model_loaded": True}


@app.get("/api/v2/info", response_model=APIInfoResponse)
async def api_info():
    """API information endpoint."""
    return {
        "version": "2.0.0",
        "api_version": "v2",
        "supported_emotion_modes": [0, 1, 2, 3],
        "max_text_tokens_limit": 500,
        "output_directory": app_state["output_dir"],
        "link_expiry_days": app_state["link_expiry_days"],
    }


@app.post("/api/v2/tts/sync")
async def tts_sync(
    text: str = Form(...),
    speaker_audio: Optional[UploadFile] = File(None, description="Upload speaker audio file"),
    speaker_audio_path: Optional[str] = Form(None, description="Server-side audio file path"),
    emotion_audio: Optional[UploadFile] = File(None),
    emotion_audio_path: Optional[str] = Form(None),
    emotion_mode: int = Form(0),
    emotion_text: str = Form(""),
    emotion_weight: float = Form(0.65),
    emotion_vector: Optional[str] = Form(None),
    use_random: bool = Form(False),
    max_text_tokens: int = Form(120),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.8),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: Optional[int] = Form(None),
    return_file: bool = Form(False, description="Return audio file directly instead of JSON"),
):
    """
    Synchronous TTS synthesis endpoint.

    Audio Input:
    - speaker_audio: Upload speaker audio file (WAV)
    - speaker_audio_path: Or provide server-side file path
      (Exactly one is required)

    - emotion_audio: Upload emotion reference audio (optional)
    - emotion_audio_path: Or provide server-side file path
      (Cannot specify both)

    Response Formats:
    - Default: JSON with download information (SyncTTSResponse)
    - return_file=true: Direct audio file download (FileResponse)
    """
    if app_state["tts"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleanup_paths = []
    start_time = datetime.now()

    try:
        # Handle speaker audio input
        if speaker_audio and speaker_audio_path:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both speaker_audio and speaker_audio_path",
            )
        
        if not speaker_audio and not speaker_audio_path:
            raise HTTPException(
                status_code=400,
                detail="Either speaker_audio or speaker_audio_path is required",
            )

        if speaker_audio:
            # Save uploaded file to temp location
            suffix = Path(speaker_audio.filename).suffix if speaker_audio.filename else ".wav"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
            cleanup_paths.append(temp_file.name)
            
            content = await speaker_audio.read()
            temp_file.write(content)
            temp_file.close()
            
            speaker_audio_path = temp_file.name
        else:
            # Use server path
            if not os.path.isabs(speaker_audio_path):
                speaker_audio_path = os.path.abspath(speaker_audio_path)
            
            if not os.path.exists(speaker_audio_path):
                 raise HTTPException(status_code=404, detail=f"Speaker audio file not found: {speaker_audio_path}")

        # Handle emotion audio input (if emotion_mode == 1)
        if emotion_mode == 1:
            if emotion_audio and emotion_audio_path:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot specify both emotion_audio and emotion_audio_path",
                )
            
            if not emotion_audio and not emotion_audio_path:
                 raise HTTPException(
                    status_code=400,
                    detail="emotion_audio or emotion_audio_path is required when emotion_mode=1",
                )

            if emotion_audio:
                suffix = Path(emotion_audio.filename).suffix if emotion_audio.filename else ".wav"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
                cleanup_paths.append(temp_file.name)
                
                content = await emotion_audio.read()
                temp_file.write(content)
                temp_file.close()
                
                emotion_audio_path = temp_file.name
            else:
                 if not os.path.isabs(emotion_audio_path):
                    emotion_audio_path = os.path.abspath(emotion_audio_path)
                 
                 if not os.path.exists(emotion_audio_path):
                    raise HTTPException(status_code=404, detail=f"Emotion audio file not found: {emotion_audio_path}")
        else:
            emotion_audio_path = None

        # Parse emotion_vector if provided
        parsed_emotion_vector = None
        if emotion_vector:
            try:
                import json

                parsed_emotion_vector = json.loads(emotion_vector)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid emotion_vector JSON format"
                )

        # Build request object
        request = TTSRequest(
            text=text,
            emotion_mode=emotion_mode,
            emotion_text=emotion_text,
            emotion_weight=emotion_weight,
            emotion_vector=parsed_emotion_vector,
            use_random=use_random,
            max_text_tokens=max_text_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
        )

        # Normalize text
        converter = app_state["converter"]
        normalized_text = normalize_text(text, converter)

        # Generate output path
        output_path = os.path.join(
            app_state["output_dir"],
            f"sync_{uuid.uuid4()}_{int(datetime.now().timestamp())}.wav",
        )

        # Perform TTS inference
        tts = app_state["tts"]
        result_path = infer_single(
            tts,
            normalized_text,
            speaker_audio_path,
            emotion_audio_path,
            output_path,
            request,
        )

        # Simple binary response logic
        if return_file:
            # Direct file download
            return FileResponse(
                result_path,
                media_type="audio/wav",
                filename=os.path.basename(result_path),
            )
        else:
            # Default: JSON with download link
            download_metadata = create_download_link_with_metadata("sync", result_path)
            return SyncTTSResponse(
                download=DownloadInfo(**download_metadata),
                created_at=start_time,
                completed_at=datetime.now()
            )

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sync TTS failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")
    finally:
        cleanup_temp_files(*cleanup_paths)


@app.post("/api/v2/tts/async", response_model=AsyncTaskResponse)
async def tts_async(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    speaker_audio: Optional[UploadFile] = File(None, description="Upload speaker audio file"),
    speaker_audio_path: Optional[str] = Form(None, description="Server-side audio file path"),
    emotion_audio: Optional[UploadFile] = File(None),
    emotion_audio_path: Optional[str] = Form(None),
    emotion_mode: int = Form(0),
    emotion_text: str = Form(""),
    emotion_weight: float = Form(0.65),
    emotion_vector: Optional[str] = Form(None),
    use_random: bool = Form(False),
    max_text_tokens: int = Form(120),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.8),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: Optional[int] = Form(None),
):
    """
    Asynchronous TTS task submission endpoint.

    Audio Input:
    - speaker_audio: Upload speaker audio file (WAV)
    - speaker_audio_path: Or provide server-side file path
      (Exactly one is required)

    - emotion_audio: Upload emotion reference audio (optional)
    - emotion_audio_path: Or provide server-side file path
      (Cannot specify both)

    Returns task_id for status tracking.
    """
    if app_state["tts"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleanup_paths = []

    try:
        # Handle speaker audio input
        if speaker_audio and speaker_audio_path:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both speaker_audio and speaker_audio_path",
            )
        
        if not speaker_audio and not speaker_audio_path:
            raise HTTPException(
                status_code=400,
                detail="Either speaker_audio or speaker_audio_path is required",
            )

        if speaker_audio:
            # Save uploaded file to temp location
            suffix = Path(speaker_audio.filename).suffix if speaker_audio.filename else ".wav"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
            cleanup_paths.append(temp_file.name)
            
            content = await speaker_audio.read()
            temp_file.write(content)
            temp_file.close()
            
            speaker_audio_path = temp_file.name
        else:
            # Use server path
            if not os.path.isabs(speaker_audio_path):
                speaker_audio_path = os.path.abspath(speaker_audio_path)
            
            if not os.path.exists(speaker_audio_path):
                 raise HTTPException(status_code=404, detail=f"Speaker audio file not found: {speaker_audio_path}")

        # Handle emotion audio input (if emotion_mode == 1)
        if emotion_mode == 1:
            if emotion_audio and emotion_audio_path:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot specify both emotion_audio and emotion_audio_path",
                )
            
            if not emotion_audio and not emotion_audio_path:
                 raise HTTPException(
                    status_code=400,
                    detail="emotion_audio or emotion_audio_path is required when emotion_mode=1",
                )

            if emotion_audio:
                suffix = Path(emotion_audio.filename).suffix if emotion_audio.filename else ".wav"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir())
                cleanup_paths.append(temp_file.name)
                
                content = await emotion_audio.read()
                temp_file.write(content)
                temp_file.close()
                
                emotion_audio_path = temp_file.name
            else:
                 if not os.path.isabs(emotion_audio_path):
                    emotion_audio_path = os.path.abspath(emotion_audio_path)
                 
                 if not os.path.exists(emotion_audio_path):
                    raise HTTPException(status_code=404, detail=f"Emotion audio file not found: {emotion_audio_path}")
        else:
            emotion_audio_path = None

        # Parse emotion_vector if provided
        parsed_emotion_vector = None
        if emotion_vector:
            try:
                import json

                parsed_emotion_vector = json.loads(emotion_vector)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid emotion_vector JSON format"
                )

        # Build request object
        request = TTSRequest(
            text=text,
            emotion_mode=emotion_mode,
            emotion_text=emotion_text,
            emotion_weight=emotion_weight,
            emotion_vector=parsed_emotion_vector,
            use_random=use_random,
            max_text_tokens=max_text_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
        )

        # Create task
        task_id = str(uuid.uuid4())
        app_state["tasks"][task_id] = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "created_at": datetime.now(),
            "completed_at": None,
            "result_path": None,
            "download_url": None,
            "error": None,
        }

        # Submit background task
        background_tasks.add_task(
            process_async_task,
            task_id,
            text,
            speaker_audio_path,
            emotion_audio_path,
            request,
            cleanup_paths,
        )

        return {"task_id": task_id, "status": TaskStatus.PENDING}

    except HTTPException:
        # Cleanup if task submission failed
        cleanup_temp_files(*cleanup_paths)
        raise
    except ValidationError as e:
        cleanup_temp_files(*cleanup_paths)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        cleanup_temp_files(*cleanup_paths)
        logger.error(f"Async task submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")


@app.get("/api/v2/tts/status/{task_id}", response_model=EnhancedTaskStatusResponse)
async def task_status(task_id: str):
    """Query task status with download information."""
    if task_id not in app_state["tasks"]:
        raise HTTPException(status_code=404, detail="Task not found")

    task = app_state["tasks"][task_id]
    response = {
        "task_id": task["task_id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "completed_at": task["completed_at"],
        "error": task["error"],
    }

    # Add download info if task is completed
    if task["status"] == TaskStatus.COMPLETED and task.get("link_id"):
        link_id = task["link_id"]
        if link_id in app_state["download_links"]:
            link_data = app_state["download_links"][link_id]
            response["download"] = DownloadInfo(
                download_url=task["download_url"],
                link_id=link_id,
                expires_at=link_data["expires_at"],
                file_size=link_data["file_size"],
                file_format=link_data.get("file_format", "wav"),
                duration=link_data.get("duration")
            )

    return response


@app.get("/api/v2/tts/result/{task_id}")
async def task_result(task_id: str):
    """Retrieve task result (audio file)."""
    if task_id not in app_state["tasks"]:
        raise HTTPException(status_code=404, detail="Task not found")

    task = app_state["tasks"][task_id]

    if task["status"] == TaskStatus.PENDING or task["status"] == TaskStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Task still processing")

    if task["status"] == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Task failed: {task['error']}")

    if not task["result_path"] or not os.path.exists(task["result_path"]):
        raise HTTPException(
            status_code=410, detail="Result has expired or been removed"
        )

    return FileResponse(
        task["result_path"],
        media_type="audio/wav",
        filename=os.path.basename(task["result_path"]),
    )


@app.get("/api/v2/download/{link_id}")
async def download_file(link_id: str):
    """Download file via temporary link."""
    if link_id not in app_state["download_links"]:
        raise HTTPException(status_code=404, detail="Download link not found")

    link_info = app_state["download_links"][link_id]

    # Check expiry
    if datetime.now() > link_info["expires_at"]:
        del app_state["download_links"][link_id]
        raise HTTPException(status_code=410, detail="Download link has expired")

    # Check file exists
    if not os.path.exists(link_info["file_path"]):
        raise HTTPException(status_code=410, detail="File has been removed")

    return FileResponse(
        link_info["file_path"],
        media_type="audio/wav",
        filename=os.path.basename(link_info["file_path"]),
    )


# ===== Exception Handlers =====
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": None,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": None,
            }
        },
    )


# ===== Main Entry Point =====
def ensure_model_files(model_dir: str, config_path: str):
    """Ensure required model files exist."""
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist")
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
            logger.error(f"Required file {file_path} does not exist")
            sys.exit(1)

    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist")
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    app_state["shutdown_event"].set()
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IndexTTS2 API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--link-expiry-days",
        type=int,
        default=7,
        help="Download link expiry in days",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host address"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
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

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Ensure model files exist
    ensure_model_files(args.model_dir, args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    app_state["output_dir"] = os.path.abspath(args.output_dir)
    app_state["link_expiry_days"] = args.link_expiry_days

    # Initialize OpenCC converter
    logger.info("Initializing OpenCC converter...")
    app_state["converter"] = OpenCC("t2s")

    # Load IndexTTS2 model
    logger.info("Loading IndexTTS2 model...")
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_fp16=args.fp16,
        device=args.device,
        use_cuda_kernel=args.cuda_kernel,
        use_deepspeed=args.deepspeed,
    )

    app_state["tts"] = tts
    logger.info("IndexTTS2 model loaded successfully")

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Output directory: {app_state['output_dir']}")
    logger.info(f"Download link expiry: {args.link_expiry_days} days")
    logger.info(
        "API documentation: http://{}:{}/docs".format(
            args.host if args.host != "0.0.0.0" else "localhost", args.port
        )
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
