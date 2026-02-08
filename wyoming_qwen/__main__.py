#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import signal
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .download import load_speakers_json
from .handler import Qwen3EventHandler

_LOGGER = logging.getLogger(__name__)

# Language name to BCP 47 language code mapping
LANGUAGE_CODE_MAP = {
    "English": "en-US",
    "Chinese": "zh-CN",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
}


async def main() -> None:
    """Main entry point."""
    # Platform check - MLX requires Apple Silicon
    if platform.machine() != "arm64":
        raise RuntimeError(
            f"wyoming-qwen requires Apple Silicon (M1/M2/M3/M4). "
            f"Detected platform: {platform.machine()}"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speaker",
        required=True,
        help="Default Qwen speaker (Ryan, Vivian, Aiden, etc.)",
    )
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="qwen",
        help="Enable discovery over zeroconf with optional name (default: qwen)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory for model cache",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )

    # Qwen synthesis parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Synthesis temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.05, help="Repetition penalty"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum tokens to generate"
    )

    # Audio streaming parameters
    parser.add_argument(
        "--auto-punctuation",
        default=".?!。？！．؟",
        help="Automatically add punctuation",
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable audio streaming on sentence boundaries",
    )

    # Logging
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Default to first data directory
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Load speaker info
    speakers_info = load_speakers_json()

    # Validate default speaker
    if args.speaker not in speakers_info:
        raise ValueError(
            f"Unknown speaker: {args.speaker}. "
            f"Available speakers: {list(speakers_info.keys())}"
        )

    # Create Wyoming Info
    voices = [
        TtsVoice(
            name=speaker_name,
            description=speaker_info["description"],
            attribution=Attribution(
                name="Alibaba Qwen Team", url="https://github.com/QwenLM/Qwen3-TTS"
            ),
            installed=True,
            version=None,
            languages=[
                LANGUAGE_CODE_MAP.get(lang, lang.lower())
                for lang in speaker_info["languages"]
            ],
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
                version=__version__,
                supports_synthesize_streaming=(not args.no_streaming),
            )
        ],
    )

    # Start server
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Ready")
    server_task = asyncio.create_task(
        server.run(
            partial(
                Qwen3EventHandler,
                wyoming_info,
                args,
                speakers_info,
            )
        )
    )
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, server_task.cancel)
    loop.add_signal_handler(signal.SIGTERM, server_task.cancel)

    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("Server stopped")


# -----------------------------------------------------------------------------


def run():
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
