"""Event handler for clients of the server."""

import argparse
import asyncio
import logging
import math
from typing import Any, Dict, Optional

import numpy as np
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .const import MLX_MODEL_PATHS, MLX_SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

# Keep models loaded per voice mode (lazy loading)
_MODELS: Dict[str, Any] = {}
_MODEL_LOCK = asyncio.Lock()


class Qwen3EventHandler(AsyncEventHandler):
    """Wyoming event handler for Qwen3-TTS (MLX)."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        speakers_info: Dict[str, Any],
        voice_mode: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speakers_info = speakers_info
        self.voice_mode = voice_mode
        self.is_streaming: Optional[bool] = None
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Optional[Synthesize] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    # Ignore since this is only sent for compatibility reasons.
                    # For streaming, we expect:
                    # [synthesize-start] -> [synthesize-chunk]+ -> [synthesize]? -> [synthesize-stop]
                    return True

                # Sent outside a stream, so we must process it
                synthesize = Synthesize.from_event(event)
                self._synthesize = Synthesize(text="", voice=synthesize.voice)
                self.sbd = SentenceBoundaryDetector()
                start_sent = False
                for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
                    self._synthesize.text = sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(i == 0), send_stop=False
                    )
                    start_sent = True

                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Last sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(not start_sent), send_stop=True
                    )
                else:
                    # No final sentence
                    await self.write_event(AudioStop().event())

                return True

            if self.cli_args.no_streaming:
                # Streaming is not enabled
                return True

            if SynthesizeStart.is_type(event.type):
                # Start of a stream
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)

                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Final audio chunk(s)
                    await self._handle_synthesize(self._synthesize)

                # End of audio
                await self.write_event(SynthesizeStopped().event())

                _LOGGER.debug("Text stream stopped")
                return True

            if not Synthesize.is_type(event.type):
                return True

            synthesize = Synthesize.from_event(event)
            return await self._handle_synthesize(synthesize)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    def _validate_speaker(self, speaker_name: str) -> bool:
        """Check if speaker exists in speakers.json."""
        return speaker_name in self.speakers_info

    def _get_language_for_speaker(self, speaker_name: str) -> str:
        """Get primary language for speaker."""
        speaker_info = self.speakers_info.get(speaker_name, {})
        languages = speaker_info.get("languages", ["English"])
        return languages[0]  # Return first/primary language

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        global _MODELS

        _LOGGER.debug(synthesize)

        raw_text = synthesize.text

        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        if self.cli_args.auto_punctuation and text:
            # Add automatic punctuation (important for some voices)
            has_punctuation = False
            for punc_char in self.cli_args.auto_punctuation:
                if text[-1] == punc_char:
                    has_punctuation = True
                    break

            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        # Resolve speaker
        _LOGGER.debug("synthesize: raw_text=%s, text='%s'", raw_text, text)
        speaker_name: Optional[str] = None
        if synthesize.voice is not None:
            speaker_name = synthesize.voice.name

        if speaker_name is None:
            # No speaker specified - use first available speaker
            available_speakers = list(self.speakers_info.keys())
            if not available_speakers:
                raise ValueError("No speakers available")
            speaker_name = available_speakers[0]
            _LOGGER.debug("No speaker specified, using default: %s", speaker_name)

        assert speaker_name is not None

        # Validate speaker
        if not self._validate_speaker(speaker_name):
            raise ValueError(
                f"Unknown speaker: {speaker_name}. "
                f"Available speakers: {list(self.speakers_info.keys())}"
            )

        # Get speaker configuration
        speaker_config = self.speakers_info.get(speaker_name, {})

        # Get language for speaker
        language = self._get_language_for_speaker(speaker_name)

        # Get custom instruct for speaker
        speaker_instruct = speaker_config.get("instruct", "")
        instruct_value = speaker_instruct if speaker_instruct else None

        # Get speaker-specific synthesis parameters (with CLI fallbacks)
        temperature = speaker_config.get("temperature", self.cli_args.temperature)
        top_k = speaker_config.get("top_k", self.cli_args.top_k)
        top_p = speaker_config.get("top_p", self.cli_args.top_p)
        repetition_penalty = speaker_config.get(
            "repetition_penalty", self.cli_args.repetition_penalty
        )
        max_tokens = speaker_config.get("max_tokens", self.cli_args.max_tokens)

        # Load model for current voice mode (lazy loading)
        async with _MODEL_LOCK:
            if self.voice_mode not in _MODELS:
                model_path = MLX_MODEL_PATHS[self.voice_mode]
                _LOGGER.info("Loading Qwen3-TTS model for %s mode: %s", self.voice_mode, model_path)
                # Import here to avoid issues if mlx not available at module level
                from mlx_audio.tts.utils import load_model

                loop = asyncio.get_event_loop()
                _MODELS[self.voice_mode] = await loop.run_in_executor(None, load_model, model_path)
                _LOGGER.info("Model loaded successfully for %s mode", self.voice_mode)

        model = _MODELS[self.voice_mode]
        assert model is not None

        # Generate audio
        _LOGGER.debug(
            "Generating audio: mode=%s, speaker=%s, language=%s, text=%s",
            self.voice_mode,
            speaker_name,
            language,
            text,
        )

        # Prepare generation parameters
        gen_params = {
            "text": text,
            "language": language,
            "instruct": instruct_value,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        # Run synthesis in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()

        if self.voice_mode == "clone-voice":
            # Voice cloning mode: Base model with generate() function
            ref_audio = speaker_config.get("ref_audio")
            ref_text = speaker_config.get("ref_text")

            if not ref_audio or not ref_text:
                raise ValueError(
                    f"Speaker '{speaker_name}' in clone-voice mode requires "
                    f"'ref_audio' and 'ref_text' fields"
                )

            _LOGGER.debug(
                "Using Base model generate() with ref_audio=%s, ref_text=%s",
                ref_audio,
                ref_text,
            )

            results = await loop.run_in_executor(
                None,
                lambda: list(
                    model.generate(
                        text=gen_params["text"],
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        language=gen_params["language"],
                        instruct=gen_params["instruct"],
                    )
                ),
            )
        elif self.voice_mode == "voice-design":
            # Voice design mode: VoiceDesign model with generate_voice_design() function
            # Note: VoiceDesign uses instruct for voice characteristics description
            _LOGGER.debug(
                "Using VoiceDesign model generate_voice_design() with instruct=%s",
                gen_params["instruct"],
            )

            results = await loop.run_in_executor(
                None,
                lambda: list(
                    model.generate_voice_design(
                        text=gen_params["text"],
                        language=gen_params["language"],
                        instruct=gen_params["instruct"],
                        temperature=gen_params["temperature"],
                        top_p=gen_params["top_p"],
                        max_tokens=gen_params["max_tokens"],
                        stream=gen_params["stream"],
                        top_k=gen_params["top_k"],
                        repetition_penalty=gen_params["repetition_penalty"],
                    )
                ),
            )
        else:
            # Custom voice mode: CustomVoice model with generate_custom_voice()
            _LOGGER.debug(
                "Using CustomVoice model generate_custom_voice() with speaker=%s",
                speaker_name,
            )

            results = await loop.run_in_executor(
                None,
                lambda: list(
                    model.generate_custom_voice(
                        text=gen_params["text"],
                        speaker=speaker_name,
                        language=gen_params["language"],
                        instruct=gen_params["instruct"],
                        temperature=gen_params["temperature"],
                        top_p=gen_params["top_p"],
                        max_tokens=gen_params["max_tokens"],
                        stream=gen_params["stream"],
                        top_k=gen_params["top_k"],
                        repetition_penalty=gen_params["repetition_penalty"],
                    )
                ),
            )

        if not results:
            raise RuntimeError("No audio generated")

        result = results[0]
        audio_mx = result.audio  # mx.array
        sample_rate = result.sample_rate  # Should be 24000

        # Convert MLX array to numpy
        audio_np = np.array(audio_mx)

        # Convert float32 [-1, 1] to int16 PCM
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Send audio events
        if send_start:
            await self.write_event(
                AudioStart(
                    rate=sample_rate,
                    width=2,  # int16
                    channels=1,  # mono
                ).event(),
            )

        # Split into chunks
        bytes_per_sample = 2  # int16
        bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
        num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))

        for i in range(num_chunks):
            offset = i * bytes_per_chunk
            chunk = audio_bytes[offset : offset + bytes_per_chunk]

            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=2,
                    channels=1,
                ).event(),
            )

        if send_stop:
            await self.write_event(AudioStop().event())

        return True
