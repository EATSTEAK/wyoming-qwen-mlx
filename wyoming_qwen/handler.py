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

from .const import MLX_MODEL_PATH, MLX_SAMPLE_RATE

_LOGGER = logging.getLogger(__name__)

# Keep the model loaded (lazy loading)
_MODEL: Optional[Any] = None
_MODEL_LOCK = asyncio.Lock()


class Qwen3EventHandler(AsyncEventHandler):
    """Wyoming event handler for Qwen3-TTS (MLX)."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        speakers_info: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speakers_info = speakers_info
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
        global _MODEL

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
            # Default speaker
            speaker_name = self.cli_args.speaker

        assert speaker_name is not None

        # Validate speaker
        if not self._validate_speaker(speaker_name):
            raise ValueError(
                f"Unknown speaker: {speaker_name}. "
                f"Available speakers: {list(self.speakers_info.keys())}"
            )

        # Get language for speaker
        language = self._get_language_for_speaker(speaker_name)

        # Load model (lazy loading)
        async with _MODEL_LOCK:
            if _MODEL is None:
                _LOGGER.info("Loading Qwen3-TTS model: %s", MLX_MODEL_PATH)
                # Import here to avoid issues if mlx not available at module level
                from mlx_audio.tts import load

                loop = asyncio.get_event_loop()
                _MODEL = await loop.run_in_executor(None, load, MLX_MODEL_PATH)
                _LOGGER.info("Model loaded successfully")

        assert _MODEL is not None

        # Generate audio
        _LOGGER.debug(
            "Generating audio: speaker=%s, language=%s, text=%s",
            speaker_name,
            language,
            text,
        )

        # Run synthesis in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(
                _MODEL.generate_custom_voice(
                    text=text,
                    speaker=speaker_name,
                    language=language,
                    instruct="",
                    temperature=self.cli_args.temperature,
                    top_k=self.cli_args.top_k,
                    top_p=self.cli_args.top_p,
                    repetition_penalty=self.cli_args.repetition_penalty,
                    max_tokens=self.cli_args.max_tokens,
                    stream=False,
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
