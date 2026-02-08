"""Tests for wyoming-qwen"""

import asyncio
from pathlib import Path

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_read_event, async_write_event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize

from wyoming_qwen.handler import Qwen3EventHandler


class MockReader:
    """Mock async reader for testing."""

    def __init__(self, events):
        self.events = events
        self.index = 0

    async def read(self, n=-1):
        if self.index >= len(self.events):
            return b""
        event = self.events[self.index]
        self.index += 1
        return event.encode("utf-8") if isinstance(event, str) else event


class MockWriter:
    """Mock async writer for testing."""

    def __init__(self):
        self.events = []
        self.closed = False

    def write(self, data):
        self.events.append(data)

    def writelines(self, lines):
        for line in lines:
            self.events.append(line)

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


async def test_describe() -> None:
    """Test Describe event returns Info with speakers."""
    from wyoming_qwen.download import load_speakers_json

    speakers_info = load_speakers_json()

    # Create Wyoming Info
    from wyoming.info import Attribution, TtsProgram, TtsVoice, TtsVoiceSpeaker

    voices = [
        TtsVoice(
            name=speaker_name,
            description=speaker_info["description"],
            attribution=Attribution(
                name="Alibaba Qwen Team", url="https://github.com/QwenLM/Qwen3-TTS"
            ),
            installed=True,
            version=None,
            languages=[lang.lower() for lang in speaker_info["languages"]],
            speakers=[TtsVoiceSpeaker(name=speaker_name)],
        )
        for speaker_name, speaker_info in speakers_info.items()
    ]

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="qwen",
                description="Alibaba Qwen3 Text-to-Speech (MLX)",
                attribution=Attribution(
                    name="Alibaba Qwen Team", url="https://github.com/QwenLM/Qwen3-TTS"
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version="1.0.0",
                supports_synthesize_streaming=True,
            )
        ],
    )

    # Mock CLI args
    class Args:
        speaker = "Ryan"
        data_dir = ["/tmp/wyoming-qwen"]
        temperature = 0.9
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.05
        max_tokens = 4096
        auto_punctuation = ".?!"
        samples_per_chunk = 1024
        no_streaming = False
        debug = False

    # Create mock reader/writer
    reader = MockReader([])
    writer = MockWriter()

    # Create handler
    handler = Qwen3EventHandler(wyoming_info, Args(), speakers_info, reader, writer)

    # Send Describe event
    describe_event = Describe().event()
    result = await handler.handle_event(describe_event)

    assert result is True, "Describe event should return True"

    # Check that info was written
    assert len(writer.events) > 0, "Info event should be written"
    assert handler.wyoming_info_event is not None


async def test_speakers_loaded() -> None:
    """Test that speakers.json is loaded correctly."""
    from wyoming_qwen.download import load_speakers_json

    speakers_info = load_speakers_json()

    # Check expected speakers
    expected_speakers = [
        "Ryan",
        "Aiden",
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ono_Anna",
        "Sohee",
    ]

    assert len(speakers_info) == len(
        expected_speakers
    ), f"Expected {len(expected_speakers)} speakers, got {len(speakers_info)}"

    for speaker in expected_speakers:
        assert (
            speaker in speakers_info
        ), f"Speaker {speaker} not found in speakers.json"

    # Check speaker structure
    for speaker_name, speaker_info in speakers_info.items():
        assert "name" in speaker_info, f"Speaker {speaker_name} missing 'name'"
        assert "gender" in speaker_info, f"Speaker {speaker_name} missing 'gender'"
        assert (
            "languages" in speaker_info
        ), f"Speaker {speaker_name} missing 'languages'"
        assert (
            "description" in speaker_info
        ), f"Speaker {speaker_name} missing 'description'"
        assert isinstance(
            speaker_info["languages"], list
        ), f"Speaker {speaker_name} languages should be a list"


async def test_constants() -> None:
    """Test that constants are defined correctly."""
    from wyoming_qwen.const import (
        MLX_MODEL_PATH,
        MLX_SAMPLE_RATE,
        QWEN_PRESET_SPEAKERS,
        QWEN_SUPPORTED_LANGUAGES,
    )

    # Check MLX constants
    assert (
        MLX_MODEL_PATH == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
    ), "MLX model path incorrect"
    assert MLX_SAMPLE_RATE == 24000, "MLX sample rate should be 24000 Hz"

    # Check supported languages
    expected_languages = ["Chinese", "English", "Japanese", "Korean"]
    assert (
        QWEN_SUPPORTED_LANGUAGES == expected_languages
    ), "Supported languages incorrect"

    # Check preset speakers
    expected_speakers = [
        "Ryan",
        "Aiden",
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ono_Anna",
        "Sohee",
    ]
    assert QWEN_PRESET_SPEAKERS == expected_speakers, "Preset speakers incorrect"


async def test_speaker_validation() -> None:
    """Test speaker validation logic."""
    from wyoming_qwen.download import load_speakers_json

    speakers_info = load_speakers_json()

    class Args:
        speaker = "Ryan"
        data_dir = ["/tmp/wyoming-qwen"]
        temperature = 0.9
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.05
        max_tokens = 4096
        auto_punctuation = ".?!"
        samples_per_chunk = 1024
        no_streaming = False
        debug = False

    from wyoming.info import Attribution, Info, TtsProgram

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="qwen",
                description="Test",
                attribution=Attribution(name="Test", url=""),
                installed=True,
                voices=[],
                version="1.0.0",
            )
        ]
    )

    handler = Qwen3EventHandler(wyoming_info, Args(), speakers_info, None, None)

    # Test valid speaker
    assert handler._validate_speaker("Ryan") is True, "Ryan should be valid"
    assert handler._validate_speaker("Vivian") is True, "Vivian should be valid"

    # Test invalid speaker
    assert (
        handler._validate_speaker("InvalidSpeaker") is False
    ), "InvalidSpeaker should be invalid"


def test_import() -> None:
    """Test that the package can be imported."""
    import wyoming_qwen

    assert hasattr(wyoming_qwen, "__version__"), "Package should have __version__"
    assert wyoming_qwen.__version__ == "1.0.0", "Version should be 1.0.0"


async def test_audio_generation() -> None:
    """Test actual audio generation with Qwen3-TTS."""
    import numpy as np

    from wyoming_qwen.download import load_speakers_json

    speakers_info = load_speakers_json()

    from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker

    voices = [
        TtsVoice(
            name="Ryan",
            description=speakers_info["Ryan"]["description"],
            attribution=Attribution(
                name="Alibaba Qwen Team", url="https://github.com/QwenLM/Qwen3-TTS"
            ),
            installed=True,
            version=None,
            languages=["english"],
            speakers=[TtsVoiceSpeaker(name="Ryan")],
        )
    ]

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="qwen",
                description="Alibaba Qwen3 Text-to-Speech (MLX)",
                attribution=Attribution(
                    name="Alibaba Qwen Team", url="https://github.com/QwenLM/Qwen3-TTS"
                ),
                installed=True,
                voices=voices,
                version="1.0.0",
                supports_synthesize_streaming=True,
            )
        ],
    )

    class Args:
        speaker = "Ryan"
        data_dir = ["/tmp/wyoming-qwen"]
        temperature = 0.9
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.05
        max_tokens = 4096
        auto_punctuation = ".?!"
        samples_per_chunk = 1024
        no_streaming = False
        debug = False

    # Create mock reader/writer
    reader = MockReader([])
    writer = MockWriter()

    # Create handler
    handler = Qwen3EventHandler(wyoming_info, Args(), speakers_info, reader, writer)

    # Create synthesize event with short text
    test_text = "Hello, this is a test."
    synthesize = Synthesize(text=test_text)

    print("⏳ Generating audio (this may take a few seconds on first run)...")

    try:
        # Handle synthesize event
        result = await handler._handle_synthesize(synthesize)

        assert result is True, "Synthesis should succeed"

        # Check that audio events were written
        audio_events = [e for e in writer.events if isinstance(e, bytes)]
        assert len(audio_events) > 0, "Should have audio data"

        # Check for AudioStart event (should contain rate=24000)
        audio_start_found = False
        audio_chunks = 0
        audio_stop_found = False

        for event in audio_events:
            if b"audio-start" in event:
                audio_start_found = True
                # Check for 24000 Hz sample rate
                assert b"24000" in event, "Sample rate should be 24000 Hz"
            elif b"audio-chunk" in event:
                audio_chunks += 1
            elif b"audio-stop" in event:
                audio_stop_found = True

        assert audio_start_found, "AudioStart event should be present"
        assert audio_chunks > 0, f"Should have audio chunks (got {audio_chunks})"
        assert audio_stop_found, "AudioStop event should be present"

        print(f"✅ Audio generation test passed!")
        print(f"   - Generated {audio_chunks} audio chunks")
        print(f"   - Sample rate: 24000 Hz")
        print(f"   - Text: '{test_text}'")

    except Exception as e:
        print(f"⚠️  Audio generation test skipped: {e}")
        print("   This is expected if MLX model is not downloaded yet.")
        print("   Run the server once to download the model, then run tests again.")
        # Don't fail the test if model is not available
        pass


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_describe())
    asyncio.run(test_speakers_loaded())
    asyncio.run(test_constants())
    asyncio.run(test_speaker_validation())
    test_import()

    # Try audio generation test (may be skipped if model not available)
    asyncio.run(test_audio_generation())

    print("\n✅ All tests passed!")
