"""Utility for loading Qwen3-TTS speaker configurations."""

import json
from pathlib import Path
from typing import Any, Dict

_DIR = Path(__file__).parent


def load_speakers_json() -> Dict[str, Any]:
    """Load speakers.json from package data."""
    speakers_file = _DIR / "speakers.json"
    with open(speakers_file, encoding="utf-8") as f:
        return json.load(f)
