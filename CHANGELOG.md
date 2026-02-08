# Changelog

## 1.0.0 (2025-02-09)

**Initial release of wyoming-qwen** - A complete rewrite using Qwen3-TTS with MLX for Apple Silicon.

### Features

- **Qwen3-TTS Integration**: Alibaba's state-of-the-art TTS model via mlx-audio
- **MLX Optimization**: Native Apple Silicon acceleration with Metal Performance Shaders
- **9 Preset Speakers**: Supporting English, Chinese, Japanese, and Korean
- **High-Quality Audio**: 24000 Hz sample rate output
- **Wyoming Protocol**: Full compatibility with Home Assistant
- **Real-time Streaming**: Sentence-level audio chunking
- **Fast Installation**: uv-based package management with 5-10x faster setup

### Technical Details

- Model: `Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`
- Sample Rate: 24000 Hz
- Languages: English, Chinese (Mandarin), Japanese, Korean
- Speakers: Ryan, Aiden, Vivian, Serena, Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee

### Requirements

- Apple Silicon (M1/M2/M3/M4)
- macOS 12.0 or later
- Python 3.9+
- 8GB+ RAM recommended

### Installation

```bash
git clone https://github.com/rhasspy/wyoming-qwen.git
cd wyoming-qwen
bash install.sh
```

### Usage

```bash
wyoming-qwen --speaker Ryan --data-dir /data --uri tcp://0.0.0.0:10200
```

See README.md for complete documentation.
