"""Utility for loading Qwen3-TTS speaker configurations."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

_DIR = Path(__file__).parent


def load_speakers_json(mode: str = "custom-voice") -> Dict[str, Any]:
    """Load speakers JSON from package data based on mode.

    Args:
        mode: One of "custom-voice", "voice-design", or "clone-voice"

    Returns:
        Dictionary of speaker configurations
    """
    speakers_file = _DIR / "speakers" / f"{mode}.json"
    if not speakers_file.exists():
        return {}

    with open(speakers_file, encoding="utf-8") as f:
        return json.load(f)


def load_all_speakers() -> Dict[str, Dict[str, Any]]:
    """Load all speaker configurations from all mode files.

    Returns:
        Dictionary with mode as key and speaker configs as value
    """
    modes = ["custom-voice", "voice-design", "clone-voice"]
    all_speakers = {}

    for mode in modes:
        speakers = load_speakers_json(mode)
        if speakers:
            all_speakers[mode] = speakers

    return all_speakers
