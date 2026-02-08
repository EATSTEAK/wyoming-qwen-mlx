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
# Using mlx-community converted model for CustomVoice (supports emotion control)
MLX_MODEL_PATH = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
MLX_SAMPLE_RATE = 24000  # Hz
