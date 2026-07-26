import base64
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import wave
from unittest.mock import Mock, patch


def create_wav(path: Path, duration: float) -> None:
    frame_rate = 8000
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"\x00\x00" * int(frame_rate * duration))


def load_module():
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = Mock

    base_stub = types.ModuleType("app.whisper.base")
    base_stub.BaseWhisper = object
    base_stub.TranscribeOptions = dict
    base_stub.WhisperResult = dict
    base_stub.WhisperSegment = dict

    app_stub = types.ModuleType("app")
    app_stub.__path__ = []
    whisper_stub = types.ModuleType("app.whisper")
    whisper_stub.__path__ = []

    module_path = Path(__file__).parents[2] / "app" / "whisper" / "openai.py"
    spec = importlib.util.spec_from_file_location("app.whisper.openai", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "openai": openai_stub,
            "app": app_stub,
            "app.whisper": whisper_stub,
            "app.whisper.base": base_stub,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def test_loads_valid_wav_and_derives_name():
    module = load_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        valid = directory / "Rusty-Bucket.wav"
        create_wav(valid, 8.0)
        create_wav(directory / "Too-Short.wav", 1.0)
        valid_bytes = valid.read_bytes()

        with patch.dict(
            os.environ,
            {"OPENAI_KNOWN_SPEAKERS_DIR": temp_dir},
            clear=False,
        ):
            names, references = module.OpenAIApi.known_speakers()

    assert names == ["Rusty Bucket"]
    assert len(references) == 1
    assert references[0].startswith("data:audio/wav;base64,")
    assert base64.b64decode(references[0].split(",", 1)[1]) == valid_bytes


def test_diarization_request_includes_known_speakers():
    module = load_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        known_directory = directory / "known-speakers"
        known_directory.mkdir()
        create_wav(known_directory / "Rusty-Bucket.wav", 8.0)
        input_audio = directory / "input.wav"
        create_wav(input_audio, 5.0)

        response = Mock()
        response.model_dump.return_value = {
            "text": "hello",
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "hello",
                    "speaker": "Rusty Bucket",
                }
            ],
        }
        client = Mock()
        client.audio.transcriptions.create.return_value = response

        with patch.dict(
            os.environ,
            {"OPENAI_KNOWN_SPEAKERS_DIR": str(known_directory)},
            clear=False,
        ):
            implementation = module.OpenAIApi(
                "test-key",
                "gpt-4o-transcribe-diarize",
            )
            implementation.client = client
            result = implementation.transcribe(
                str(input_audio),
                {"initial_prompt": ""},
            )

    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["extra_body"]["known_speaker_names"] == ["Rusty Bucket"]
    assert len(kwargs["extra_body"]["known_speaker_references"]) == 1
    assert result["segments"][0]["speaker"] == "Rusty Bucket"


def test_short_known_speaker_falls_back_to_generic_label():
    module = load_module()
    result = {
        "text": "short known speaker and longer unknown speaker",
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "short known speaker",
                "speaker": "Rusty Bucket",
            },
            {
                "start": 2.0,
                "end": 7.0,
                "text": "longer unknown speaker",
                "speaker": "A",
            },
        ],
        "language": "en",
    }

    filtered = module.OpenAIApi.apply_known_speaker_minimum(
        result,
        ["Rusty Bucket"],
    )

    assert filtered["segments"][0]["speaker"] == "A"
    assert filtered["segments"][1]["speaker"] == "B"


def test_known_speaker_duration_is_summed_across_segments():
    module = load_module()
    result = {
        "text": "one two",
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "one",
                "speaker": "Sonic",
            },
            {
                "start": 3.0,
                "end": 5.1,
                "text": "two",
                "speaker": "Sonic",
            },
        ],
        "language": "en",
    }

    filtered = module.OpenAIApi.apply_known_speaker_minimum(result, ["Sonic"])

    assert all(segment["speaker"] == "Sonic" for segment in filtered["segments"])


def test_non_diarization_model_does_not_load_known_speakers():
    module = load_module()
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        known_directory = directory / "known-speakers"
        known_directory.mkdir()
        create_wav(known_directory / "Rusty-Bucket.wav", 3.0)
        input_audio = directory / "input.wav"
        create_wav(input_audio, 1.0)

        response = Mock()
        response.model_dump.return_value = {"text": "hello"}
        client = Mock()
        client.audio.transcriptions.create.return_value = response

        with patch.dict(
            os.environ,
            {"OPENAI_KNOWN_SPEAKERS_DIR": str(known_directory)},
            clear=False,
        ):
            implementation = module.OpenAIApi("test-key", "gpt-4o-transcribe")
            implementation.client = client
            implementation.transcribe(
                str(input_audio),
                {"initial_prompt": ""},
            )

    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert "extra_body" not in kwargs
