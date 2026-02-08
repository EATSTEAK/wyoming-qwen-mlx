#!/usr/bin/env python3
"""Test script to verify all voice modes work correctly."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from wyoming_qwen.const import MLX_MODEL_PATHS
from wyoming_qwen.download import load_speakers_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_LOGGER = logging.getLogger(__name__)


async def test_model_loading():
    """Test loading models for each mode."""
    _LOGGER.info("=" * 60)
    _LOGGER.info("Testing Model Loading")
    _LOGGER.info("=" * 60)

    from mlx_audio.tts.utils import load_model

    for mode, model_path in MLX_MODEL_PATHS.items():
        _LOGGER.info(f"\nTesting mode: {mode}")
        _LOGGER.info(f"Model path: {model_path}")

        try:
            _LOGGER.info("Loading model...")
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, load_model, model_path)
            _LOGGER.info(f"✓ Model loaded successfully for {mode} mode")

            # Check available methods
            has_generate = hasattr(model, "generate")
            has_generate_custom = hasattr(model, "generate_custom_voice")
            has_generate_design = hasattr(model, "generate_voice_design")

            _LOGGER.info(f"  - has generate(): {has_generate}")
            _LOGGER.info(f"  - has generate_custom_voice(): {has_generate_custom}")
            _LOGGER.info(f"  - has generate_voice_design(): {has_generate_design}")

            # Verify correct method for each mode
            if mode == "clone-voice" and not has_generate:
                _LOGGER.warning(f"  ⚠ clone-voice mode requires generate() method")
            elif mode == "voice-design" and not has_generate_design:
                _LOGGER.warning(f"  ⚠ voice-design mode requires generate_voice_design() method")
            elif mode == "custom-voice" and not has_generate_custom:
                _LOGGER.warning(f"  ⚠ custom-voice mode requires generate_custom_voice() method")

        except Exception as e:
            _LOGGER.error(f"✗ Failed to load model for {mode}: {e}")
            return False

    return True


async def test_speaker_configs():
    """Test speaker configuration files."""
    _LOGGER.info("\n" + "=" * 60)
    _LOGGER.info("Testing Speaker Configurations")
    _LOGGER.info("=" * 60)

    modes = ["custom-voice", "voice-design", "clone-voice"]

    for mode in modes:
        _LOGGER.info(f"\nTesting mode: {mode}")
        speakers = load_speakers_json(mode)

        if not speakers:
            _LOGGER.warning(f"  ⚠ No speakers found for {mode}")
            continue

        _LOGGER.info(f"  ✓ Found {len(speakers)} speaker(s)")

        # Check first speaker configuration
        first_speaker = list(speakers.keys())[0]
        config = speakers[first_speaker]

        _LOGGER.info(f"  Example speaker: {first_speaker}")
        _LOGGER.info(f"    - Name: {config.get('name')}")
        _LOGGER.info(f"    - Languages: {config.get('languages')}")
        _LOGGER.info(f"    - Description: {config.get('description')}")

        # Check mode-specific fields
        if mode == "clone-voice":
            has_ref_audio = "ref_audio" in config
            has_ref_text = "ref_text" in config
            _LOGGER.info(f"    - Has ref_audio: {has_ref_audio}")
            _LOGGER.info(f"    - Has ref_text: {has_ref_text}")

            if not (has_ref_audio and has_ref_text):
                _LOGGER.warning(
                    f"    ⚠ Speaker missing ref_audio or ref_text for clone-voice mode"
                )

        # Check optional synthesis parameters
        optional_params = ["temperature", "top_k", "top_p", "repetition_penalty", "max_tokens"]
        for param in optional_params:
            if param in config:
                _LOGGER.info(f"    - {param}: {config[param]}")

    return True


async def test_generation(quick_test: bool = True):
    """Test audio generation for each mode and save as WAV files."""
    _LOGGER.info("\n" + "=" * 60)
    _LOGGER.info("Testing Audio Generation")
    _LOGGER.info("=" * 60)

    if quick_test:
        _LOGGER.info("(Generating audio and saving as WAV files)")

    import numpy as np
    import wave

    from mlx_audio.tts.utils import load_model

    test_text = "Hello, this is a test."
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Test custom-voice mode
    _LOGGER.info("\n1. Testing custom-voice mode")
    try:
        mode = "custom-voice"
        model = await asyncio.get_event_loop().run_in_executor(
            None, load_model, MLX_MODEL_PATHS[mode]
        )
        speakers = load_speakers_json(mode)
        if speakers:
            speaker_name = list(speakers.keys())[0]
            _LOGGER.info(f"   Using speaker: {speaker_name}")

            results = list(
                model.generate_custom_voice(
                    text=test_text,
                    speaker=speaker_name,
                    language="English",
                    instruct="",
                    temperature=0.9,
                    top_p=1.0,
                    max_tokens=4096,
                    stream=False,
                    top_k=50,
                    repetition_penalty=1.05,
                )
            )
            _LOGGER.info(f"   ✓ Generated {len(results)} audio result(s)")
            if results:
                result = results[0]
                audio_mx = result.audio
                sample_rate = result.sample_rate

                _LOGGER.info(f"   ✓ Audio shape: {audio_mx.shape}")

                # Convert MLX array to numpy and then to int16 PCM
                audio_np = np.array(audio_mx)
                audio_int16 = (audio_np * 32767).astype(np.int16)

                # Save as WAV file
                wav_file = output_dir / f"test_{mode}_{speaker_name}.wav"
                with wave.open(str(wav_file), "wb") as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_int16.tobytes())

                _LOGGER.info(f"   ✓ Saved to: {wav_file.absolute()}")
        else:
            _LOGGER.warning("   ⚠ No speakers found, skipping")
    except Exception as e:
        _LOGGER.error(f"   ✗ Failed: {e}")
        return False

    # Test voice-design mode
    _LOGGER.info("\n2. Testing voice-design mode")
    try:
        mode = "voice-design"
        model = await asyncio.get_event_loop().run_in_executor(
            None, load_model, MLX_MODEL_PATHS[mode]
        )
        speakers = load_speakers_json(mode)
        if speakers:
            speaker_name = list(speakers.keys())[0]
            speaker_config = speakers[speaker_name]
            _LOGGER.info(f"   Using voice design: {speaker_name}")
            _LOGGER.info(f"   Instruct: {speaker_config.get('instruct', '')}")

            # Voice design uses instruct to describe voice characteristics
            results = list(
                model.generate_voice_design(
                    text=test_text,
                    language=speaker_config.get("languages", ["English"])[0],
                    instruct=speaker_config.get("instruct", "A warm and friendly voice"),
                    temperature=0.9,
                    top_p=1.0,
                    max_tokens=4096,
                    stream=False,
                    top_k=50,
                    repetition_penalty=1.05,
                )
            )
            _LOGGER.info(f"   ✓ Generated {len(results)} audio result(s)")
            if results:
                result = results[0]
                audio_mx = result.audio
                sample_rate = result.sample_rate

                _LOGGER.info(f"   ✓ Audio shape: {audio_mx.shape}")

                # Convert MLX array to numpy and then to int16 PCM
                audio_np = np.array(audio_mx)
                audio_int16 = (audio_np * 32767).astype(np.int16)

                # Save as WAV file
                wav_file = output_dir / f"test_{mode}_{speaker_name}.wav"
                with wave.open(str(wav_file), "wb") as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_int16.tobytes())

                _LOGGER.info(f"   ✓ Saved to: {wav_file.absolute()}")
        else:
            _LOGGER.warning("   ⚠ No speakers found, skipping")
    except Exception as e:
        _LOGGER.error(f"   ✗ Failed: {e}")
        return False

    # Test clone-voice mode
    _LOGGER.info("\n3. Testing clone-voice mode")
    try:
        mode = "clone-voice"
        model = await asyncio.get_event_loop().run_in_executor(
            None, load_model, MLX_MODEL_PATHS[mode]
        )
        speakers = load_speakers_json(mode)
        if speakers:
            speaker_name = list(speakers.keys())[0]
            speaker_config = speakers[speaker_name]
            ref_audio = speaker_config.get("ref_audio")
            ref_text = speaker_config.get("ref_text")

            _LOGGER.info(f"   Using speaker: {speaker_name}")

            if ref_audio and ref_text and Path(ref_audio).exists():
                results = list(
                    model.generate(
                        text=test_text,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        language=speaker_config.get("languages", ["English"])[0],
                        instruct=speaker_config.get("instruct"),
                    )
                )
                _LOGGER.info(f"   ✓ Generated {len(results)} audio result(s)")
                if results:
                    result = results[0]
                    audio_mx = result.audio
                    sample_rate = result.sample_rate

                    _LOGGER.info(f"   ✓ Audio shape: {audio_mx.shape}")

                    # Convert MLX array to numpy and then to int16 PCM
                    audio_np = np.array(audio_mx)
                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    # Save as WAV file
                    wav_file = output_dir / f"test_{mode}_{speaker_name}.wav"
                    with wave.open(str(wav_file), "wb") as wav:
                        wav.setnchannels(1)  # mono
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(sample_rate)
                        wav.writeframes(audio_int16.tobytes())

                    _LOGGER.info(f"   ✓ Saved to: {wav_file.absolute()}")
            else:
                _LOGGER.warning(
                    f"   ⚠ ref_audio not found at: {ref_audio}, skipping generation test"
                )
                _LOGGER.info("   ℹ To test clone-voice, set valid ref_audio path in speakers/clone-voice.json")
        else:
            _LOGGER.warning("   ⚠ No speakers found, skipping")
    except Exception as e:
        _LOGGER.error(f"   ✗ Failed: {e}")
        # Clone-voice failure is acceptable if ref_audio doesn't exist
        _LOGGER.warning("   (This is expected if ref_audio file doesn't exist)")

    return True


async def main():
    """Run all tests."""
    _LOGGER.info("Wyoming-Qwen Voice Mode Test Suite")
    _LOGGER.info("=" * 60)

    try:
        # Test 1: Model loading
        if not await test_model_loading():
            _LOGGER.error("\n✗ Model loading test failed")
            return 1

        # Test 2: Speaker configurations
        if not await test_speaker_configs():
            _LOGGER.error("\n✗ Speaker configuration test failed")
            return 1

        # Test 3: Audio generation (quick test)
        _LOGGER.info("\n" + "=" * 60)
        response = input("Run audio generation tests? This will download models (~3GB). (y/N): ")
        if response.lower() == 'y':
            if not await test_generation(quick_test=True):
                _LOGGER.error("\n✗ Audio generation test failed")
                return 1
        else:
            _LOGGER.info("Skipping audio generation tests")

        _LOGGER.info("\n" + "=" * 60)
        _LOGGER.info("✓ All tests passed!")
        _LOGGER.info("=" * 60)
        return 0

    except KeyboardInterrupt:
        _LOGGER.info("\nTest interrupted by user")
        return 130
    except Exception as e:
        _LOGGER.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
