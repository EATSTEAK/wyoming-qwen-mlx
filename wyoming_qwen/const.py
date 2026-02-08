# Qwen3-TTS supported languages (via mlx-audio)
QWEN_SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
]

# Qwen3-TTS preset speakers (mlx-audio CustomVoice 0.6B)
QWEN_PRESET_SPEAKERS = [
    "Ryan",      # English male
    "Aiden",     # English male
    "Vivian",    # Chinese/English female
    "Serena",    # Chinese female
    "Uncle_Fu",  # Chinese male
    "Dylan",     # Chinese male (Beijing)
    "Eric",      # Chinese male (Sichuan)
    "Ono_Anna",  # Japanese
    "Sohee",    # Korean
]

# MLX-specific constants
# Model paths for different voice modes
MLX_MODEL_PATHS = {
    "custom-voice": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",  # CustomVoice: speaker + emotion control
    "voice-design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",  # VoiceDesign: create any voice
    "clone-voice": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",          # Base: voice cloning with ref_audio
}
MLX_SAMPLE_RATE = 24000  # Hz
