#!/usr/bin/env python3
"""
IndexTTS2 API Test Script

Tests all API endpoints including synchronous/asynchronous operations,
audio input methods, emotion control modes, and generation parameters.

Usage:
    Terminal 1: uv run api_v2.py --fp16
    Terminal 2: uv run test_api_v2.py
"""

import json
import os
import sys
import time

import requests

# API configuration
API_BASE_URL = "http://localhost:8000"
TEST_TEXT = "你好，我是 IndexTTS2 語音合成系統。"
TEST_SPEAKER_AUDIO = "examples/voice_01.wav"
TEST_EMOTION_AUDIO = "examples/emo_sad.wav"
OUTPUT_DIR = "test_outputs"

# Test results tracking
test_results = {"passed": 0, "failed": 0, "skipped": 0, "tests": []}


# ===== Utility Functions =====
def print_section(title):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_test(name, status, message=""):
    """Print test result."""
    symbols = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}
    colors = {"PASS": "\033[92m", "FAIL": "\033[91m", "SKIP": "\033[93m"}
    reset = "\033[0m"

    symbol = symbols.get(status, "?")
    color = colors.get(status, "")

    print(f"{color}{symbol} {name}{reset}")
    if message:
        print(f"  {message}")

    test_results["tests"].append({"name": name, "status": status, "message": message})
    if status == "PASS":
        test_results["passed"] += 1
    elif status == "FAIL":
        test_results["failed"] += 1
    elif status == "SKIP":
        test_results["skipped"] += 1


def check_prerequisites():
    """Check if prerequisites are met."""
    print_section("Prerequisites Check")

    # Check API server
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_test("API server is running", "PASS")
        else:
            print_test(
                "API server health check", "FAIL", f"HTTP {response.status_code}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print_test("API server is running", "FAIL", f"Connection failed: {e}")
        print("\nPlease start the API server first:")
        print("  Terminal 1: uv run api_v2.py --fp16")
        return False

    # Check test audio files
    if not os.path.exists(TEST_SPEAKER_AUDIO):
        print_test(
            "Speaker audio file exists", "FAIL", f"Not found: {TEST_SPEAKER_AUDIO}"
        )
        return False
    else:
        print_test("Speaker audio file exists", "PASS")

    if not os.path.exists(TEST_EMOTION_AUDIO):
        print_test(
            "Emotion audio file exists", "SKIP", "Will skip emotion mode 1 tests"
        )
    else:
        print_test("Emotion audio file exists", "PASS")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_test("Output directory created", "PASS", OUTPUT_DIR)

    return True


# ===== Health & Info Tests =====
def test_health_check():
    """Test health check endpoint."""
    print_section("Health Check Tests")

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        data = response.json()

        if response.status_code == 200 and data.get("status") == "healthy":
            print_test(
                "GET /health", "PASS", f"Model loaded: {data.get('model_loaded')}"
            )
        else:
            print_test("GET /health", "FAIL", f"Unexpected response: {data}")
    except Exception as e:
        print_test("GET /health", "FAIL", str(e))


def test_api_info():
    """Test API info endpoint."""
    print_section("API Info Tests")

    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/info")
        data = response.json()

        if response.status_code == 200:
            print_test("GET /api/v2/info", "PASS")
            print(f"  Version: {data.get('version')}")
            print(f"  Emotion modes: {data.get('supported_emotion_modes')}")
            print(f"  Max text tokens: {data.get('max_text_tokens_limit')}")
            print(f"  Output dir: {data.get('output_directory')}")
        else:
            print_test("GET /api/v2/info", "FAIL", f"HTTP {response.status_code}")
    except Exception as e:
        print_test("GET /api/v2/info", "FAIL", str(e))


# ===== Synchronous Endpoint Tests =====
def test_sync_basic_upload():
    """Test basic synchronous TTS with file upload."""
    print_section("Synchronous TTS - Basic Upload")

    try:
        with open(TEST_SPEAKER_AUDIO, "rb") as f:
            files = {"speaker_audio_file": f}
            data = {"text": TEST_TEXT, "speaker_audio_type": "upload"}

            response = requests.post(
                f"{API_BASE_URL}/api/v2/tts/sync", files=files, data=data, timeout=120
            )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "sync_basic_upload.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test(
                "POST /api/v2/tts/sync (upload)", "PASS", f"Saved to {output_path}"
            )
        else:
            print_test(
                "POST /api/v2/tts/sync (upload)", "FAIL", f"HTTP {response.status_code}"
            )
    except Exception as e:
        print_test("POST /api/v2/tts/sync (upload)", "FAIL", str(e))


def test_sync_local_path():
    """Test synchronous TTS with local path."""
    print_section("Synchronous TTS - Local Path")

    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "sync_local_path.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test(
                "POST /api/v2/tts/sync (path)", "PASS", f"Saved to {output_path}"
            )
        else:
            print_test(
                "POST /api/v2/tts/sync (path)",
                "FAIL",
                f"HTTP {response.status_code}: {response.text}",
            )
    except Exception as e:
        print_test("POST /api/v2/tts/sync (path)", "FAIL", str(e))


def test_sync_return_path():
    """Test synchronous TTS returning server path."""
    print_section("Synchronous TTS - Return Path")

    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "return_path": "true",
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            audio_path = result.get("audio_path")
            if audio_path:
                print_test(
                    "POST /api/v2/tts/sync (return_path)", "PASS", f"Path: {audio_path}"
                )
            else:
                print_test(
                    "POST /api/v2/tts/sync (return_path)",
                    "FAIL",
                    "No audio_path in response",
                )
        else:
            print_test(
                "POST /api/v2/tts/sync (return_path)",
                "FAIL",
                f"HTTP {response.status_code}",
            )
    except Exception as e:
        print_test("POST /api/v2/tts/sync (return_path)", "FAIL", str(e))


# ===== Asynchronous Endpoint Tests =====
def test_async_workflow():
    """Test complete async workflow: submit -> status -> result."""
    print_section("Asynchronous TTS Workflow")

    task_id = None

    # Step 1: Submit task
    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
        }

        response = requests.post(f"{API_BASE_URL}/api/v2/tts/async", data=data)

        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print_test("POST /api/v2/tts/async (submit)", "PASS", f"Task ID: {task_id}")
        else:
            print_test(
                "POST /api/v2/tts/async (submit)",
                "FAIL",
                f"HTTP {response.status_code}",
            )
            return
    except Exception as e:
        print_test("POST /api/v2/tts/async (submit)", "FAIL", str(e))
        return

    # Step 2: Poll status
    max_wait = 120  # seconds
    poll_interval = 2
    waited = 0

    print(f"\nPolling task status (max {max_wait}s)...")

    while waited < max_wait:
        try:
            response = requests.get(f"{API_BASE_URL}/api/v2/tts/status/{task_id}")
            status_data = response.json()
            status = status_data.get("status")

            print(f"  [{waited}s] Status: {status}")

            if status == "completed":
                print_test("GET /api/v2/tts/status/{task_id}", "PASS", "Task completed")
                break
            elif status == "failed":
                error = status_data.get("error")
                print_test(
                    "GET /api/v2/tts/status/{task_id}", "FAIL", f"Task failed: {error}"
                )
                return
            elif status in ["pending", "processing"]:
                time.sleep(poll_interval)
                waited += poll_interval
            else:
                print_test(
                    "GET /api/v2/tts/status/{task_id}",
                    "FAIL",
                    f"Unknown status: {status}",
                )
                return

        except Exception as e:
            print_test("GET /api/v2/tts/status/{task_id}", "FAIL", str(e))
            return

    if waited >= max_wait:
        print_test("Task completion", "FAIL", "Timeout waiting for task completion")
        return

    # Step 3: Get result
    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/tts/result/{task_id}")

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "async_result.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test(
                "GET /api/v2/tts/result/{task_id}", "PASS", f"Saved to {output_path}"
            )
        else:
            print_test(
                "GET /api/v2/tts/result/{task_id}",
                "FAIL",
                f"HTTP {response.status_code}",
            )
    except Exception as e:
        print_test("GET /api/v2/tts/result/{task_id}", "FAIL", str(e))


# ===== Emotion Control Tests =====
def test_emotion_mode_0():
    """Test emotion mode 0 (use speaker audio)."""
    print_section("Emotion Control - Mode 0 (Speaker)")

    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "emotion_mode": "0",
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "emotion_mode_0.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test("Emotion mode 0", "PASS", f"Saved to {output_path}")
        else:
            print_test("Emotion mode 0", "FAIL", f"HTTP {response.status_code}")
    except Exception as e:
        print_test("Emotion mode 0", "FAIL", str(e))


def test_emotion_mode_1():
    """Test emotion mode 1 (reference audio)."""
    print_section("Emotion Control - Mode 1 (Reference Audio)")

    if not os.path.exists(TEST_EMOTION_AUDIO):
        print_test("Emotion mode 1", "SKIP", "Emotion audio file not found")
        return

    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "emotion_mode": "1",
            "emotion_audio_type": "path",
            "emotion_audio_value": TEST_EMOTION_AUDIO,
            "emotion_weight": "0.8",
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "emotion_mode_1.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test("Emotion mode 1", "PASS", f"Saved to {output_path}")
        else:
            print_test(
                "Emotion mode 1",
                "FAIL",
                f"HTTP {response.status_code}: {response.text}",
            )
    except Exception as e:
        print_test("Emotion mode 1", "FAIL", str(e))


def test_emotion_mode_2():
    """Test emotion mode 2 (emotion vector)."""
    print_section("Emotion Control - Mode 2 (Emotion Vector)")

    try:
        # Example emotion vector (8 values: HAPPY, ANGRY, SAD, AFRAID, DISGUSTED, MELANCHOLIC, SURPRISED, CALM)
        emotion_vector = [0.1, 0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.2]

        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "emotion_mode": "2",
            "emotion_vector": json.dumps(emotion_vector),
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "emotion_mode_2.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test("Emotion mode 2", "PASS", f"Saved to {output_path}")
        else:
            print_test(
                "Emotion mode 2",
                "FAIL",
                f"HTTP {response.status_code}: {response.text}",
            )
    except Exception as e:
        print_test("Emotion mode 2", "FAIL", str(e))


def test_emotion_mode_3():
    """Test emotion mode 3 (emotion text)."""
    print_section("Emotion Control - Mode 3 (Emotion Text)")

    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "emotion_mode": "3",
            "emotion_text": "委屈巴巴",
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "emotion_mode_3.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test("Emotion mode 3", "PASS", f"Saved to {output_path}")
        else:
            print_test("Emotion mode 3", "FAIL", f"HTTP {response.status_code}")
    except Exception as e:
        print_test("Emotion mode 3", "FAIL", str(e))


# ===== Generation Parameters Tests =====
def test_generation_params():
    """Test various generation parameters."""
    print_section("Generation Parameters Tests")

    # Test 1: Temperature variation
    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "temperature": "1.2",
            "top_p": "0.9",
            "top_k": "50",
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "params_temperature.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test(
                "Custom temperature/top_p/top_k", "PASS", f"Saved to {output_path}"
            )
        else:
            print_test(
                "Custom temperature/top_p/top_k", "FAIL", f"HTTP {response.status_code}"
            )
    except Exception as e:
        print_test("Custom temperature/top_p/top_k", "FAIL", str(e))

    # Test 2: Beam search
    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "do_sample": "false",
            "num_beams": "5",
        }

        response = requests.post(
            f"{API_BASE_URL}/api/v2/tts/sync", data=data, timeout=120
        )

        if response.status_code == 200:
            output_path = os.path.join(OUTPUT_DIR, "params_beam_search.wav")
            with open(output_path, "wb") as f:
                f.write(response.content)
            print_test("Beam search (no sampling)", "PASS", f"Saved to {output_path}")
        else:
            print_test(
                "Beam search (no sampling)", "FAIL", f"HTTP {response.status_code}"
            )
    except Exception as e:
        print_test("Beam search (no sampling)", "FAIL", str(e))


# ===== Error Handling Tests =====
def test_error_handling():
    """Test error scenarios."""
    print_section("Error Handling Tests")

    # Test 1: Invalid emotion vector (wrong size)
    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
            "speaker_audio_value": TEST_SPEAKER_AUDIO,
            "emotion_mode": "2",
            "emotion_vector": json.dumps([0.1, 0.2, 0.3]),  # Only 3 values instead of 8
        }

        response = requests.post(f"{API_BASE_URL}/api/v2/tts/sync", data=data)

        if response.status_code == 400:
            print_test("Invalid emotion vector", "PASS", "Correctly rejected")
        else:
            print_test(
                "Invalid emotion vector",
                "FAIL",
                f"Expected 400, got {response.status_code}",
            )
    except Exception as e:
        print_test("Invalid emotion vector", "FAIL", str(e))

    # Test 2: Missing speaker audio
    try:
        data = {
            "text": TEST_TEXT,
            "speaker_audio_type": "path",
        }  # Missing speaker_audio_value

        response = requests.post(f"{API_BASE_URL}/api/v2/tts/sync", data=data)

        if response.status_code == 400:
            print_test("Missing speaker audio", "PASS", "Correctly rejected")
        else:
            print_test(
                "Missing speaker audio",
                "FAIL",
                f"Expected 400, got {response.status_code}",
            )
    except Exception as e:
        print_test("Missing speaker audio", "FAIL", str(e))

    # Test 3: Non-existent task ID
    try:
        fake_task_id = "00000000-0000-0000-0000-000000000000"
        response = requests.get(f"{API_BASE_URL}/api/v2/tts/status/{fake_task_id}")

        if response.status_code == 404:
            print_test("Non-existent task ID", "PASS", "Correctly returned 404")
        else:
            print_test(
                "Non-existent task ID",
                "FAIL",
                f"Expected 404, got {response.status_code}",
            )
    except Exception as e:
        print_test("Non-existent task ID", "FAIL", str(e))


# ===== Test Report =====
def print_test_report():
    """Print final test report."""
    print_section("Test Report")

    total = test_results["passed"] + test_results["failed"] + test_results["skipped"]

    print(f"Total tests:   {total}")
    print(f"Passed:        {test_results['passed']} ✓")
    print(f"Failed:        {test_results['failed']} ✗")
    print(f"Skipped:       {test_results['skipped']} ○")

    if test_results["failed"] > 0:
        print("\nFailed tests:")
        for test in test_results["tests"]:
            if test["status"] == "FAIL":
                print(f"  - {test['name']}: {test['message']}")

    # Save report to JSON
    report_path = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDetailed report saved to: {report_path}")

    # Return exit code
    return 0 if test_results["failed"] == 0 else 1


# ===== Main =====
def main():
    """Main test runner."""
    print_section("IndexTTS2 API Test Suite")

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Run all tests
    test_health_check()
    test_api_info()
    test_sync_basic_upload()
    test_sync_local_path()
    test_sync_return_path()
    test_async_workflow()
    test_emotion_mode_0()
    test_emotion_mode_1()
    test_emotion_mode_2()
    test_emotion_mode_3()
    test_generation_params()
    test_error_handling()

    # Print report
    exit_code = print_test_report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
