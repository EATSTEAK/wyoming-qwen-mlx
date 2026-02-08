# Wyoming Qwen

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech system optimized for Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

## Requirements

**⚠️ Apple Silicon Required**: This project requires Apple Silicon (M1/M2/M3/M4) Mac running macOS 12.0 or later. It will not work on Intel Macs, Linux, Windows, or Docker containers.

- macOS 12.0+
- Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- 8GB+ RAM recommended
- 2GB free disk space for model cache

## Features

- 🚀 **Fast inference** on Apple Silicon with MLX optimization
- 🎙️ **9 preset speakers** supporting English, Chinese, Japanese, and Korean
- 🌍 **Multilingual support**: English, Chinese (Mandarin), Japanese, Korean
- 🔌 **Wyoming protocol** compatible with Home Assistant
- ⚡ **Real-time streaming** with sentence-level chunking
- 📦 **Auto-downloading** models from HuggingFace

## Supported Speakers

| Speaker    | Gender | Language(s)       | Description                        |
|------------|--------|-------------------|------------------------------------|
| Ryan       | Male   | English           | Dynamic with strong rhythmic drive |
| Aiden      | Male   | English           | Sunny American, clear midrange     |
| Vivian     | Female | Chinese, English  | Bright, slightly edgy              |
| Serena     | Female | Chinese           | Warm, gentle                       |
| Uncle_Fu   | Male   | Chinese           | Seasoned, low mellow timbre        |
| Dylan      | Male   | Chinese (Beijing) | Youthful, clear and natural        |
| Eric       | Male   | Chinese (Sichuan) | Lively, slightly husky             |
| Ono_Anna   | Female | Japanese          | Playful, light and nimble          |
| Sohee      | Female | Korean            | Warm with rich emotion             |

## Installation

### Quick Install (Recommended)

```bash
git clone https://github.com/rhasspy/wyoming-qwen.git
cd wyoming-qwen
bash install.sh
```

The install script will:

1. Check for Apple Silicon
2. Optionally install [uv](https://github.com/astral-sh/uv) for faster installation
3. Create a virtual environment
4. Install all dependencies

### Manual Install

```bash
git clone https://github.com/rhasspy/wyoming-qwen.git
cd wyoming-qwen
bash script/setup
```

Or with uv for faster installation:

```bash
pip install uv
bash script/setup
```

## Usage

### Start the Server

```bash
source .venv/bin/activate
wyoming-qwen --speaker Ryan --data-dir /tmp/wyoming-qwen --uri tcp://0.0.0.0:10200
```

### Command-Line Options

```text
Required:
  --speaker SPEAKER       Default speaker (Ryan, Vivian, Aiden, etc.)
  --data-dir DIR          Directory for model cache
  --uri URI               Server URI (tcp://host:port or stdio://)

Optional:
  --temperature FLOAT     Synthesis temperature (default: 0.9)
  --top-k INT             Top-k sampling (default: 50)
  --top-p FLOAT           Top-p sampling (default: 1.0)
  --repetition-penalty FLOAT  Repetition penalty (default: 1.05)
  --max-tokens INT        Maximum tokens to generate (default: 4096)
  --samples-per-chunk INT Audio chunk size (default: 1024)
  --auto-punctuation STR  Auto-add punctuation (default: .?!。？！．؟)
  --no-streaming          Disable sentence-level streaming
  --debug                 Enable debug logging
```

### Examples

**English TTS:**

```bash
wyoming-qwen --speaker Ryan --data-dir /data --uri tcp://0.0.0.0:10200
```

**Chinese TTS:**

```bash
wyoming-qwen --speaker Vivian --data-dir /data --uri tcp://0.0.0.0:10200
```

**Custom parameters:**

```bash
wyoming-qwen --speaker Ryan --data-dir /data --uri tcp://0.0.0.0:10200 \
  --temperature 1.2 --top-k 30 --repetition-penalty 1.1
```

## Home Assistant Integration

**Note**: Docker-based Home Assistant addons are not supported due to MLX requiring native macOS. Use this on a Mac running Home Assistant Core or connect remotely via TCP.

### Configuration

Add to your `configuration.yaml`:

```yaml
wyoming:
  - uri: tcp://your-mac-ip:10200
```

Or use the Wyoming integration in the UI:

1. Settings → Devices & Services → Add Integration
2. Search for "Wyoming Protocol"
3. Enter: `tcp://your-mac-ip:10200`

### Using Different Speakers

```yaml
tts:
  - platform: wyoming
    name: "Ryan TTS"
    speaker: Ryan
  - platform: wyoming
    name: "Vivian TTS"
    speaker: Vivian
```

## Performance

First run will download the model (~1.2GB) which may take a few minutes. Subsequent runs are fast:

- **Model loading**: ~2-5 seconds
- **Synthesis latency**: <100ms for short sentences on M1/M2/M3
- **Streaming**: Real-time with sentence-level chunking

## Troubleshooting

### "Platform not supported" error

This means you're not running on Apple Silicon. Check:

```bash
uname -m  # Should output: arm64
```

### Model download fails

Ensure you have internet connection and enough disk space:

```bash
df -h /tmp  # Should have 2GB+ free
```

### Audio quality issues

Try adjusting synthesis parameters:

```bash
--temperature 0.8 --top-k 40 --repetition-penalty 1.2
```

## Development

### Running Tests

```bash
source .venv/bin/activate
pytest tests/
```

### Code Formatting

```bash
bash script/format
```

### Linting

```bash
bash script/lint
```

## Differences from wyoming-piper

This is a fork of [wyoming-piper](https://github.com/rhasspy/wyoming-piper) with the following changes:

- **TTS Engine**: Piper → Qwen3-TTS (via mlx-audio)
- **Platform**: Cross-platform → Apple Silicon only
- **Model Size**: Various sizes → 0.6B parameters
- **Voices**: 100+ voices → 9 preset speakers
- **Languages**: 40+ languages → 4 languages (English, Chinese, Japanese, Korean)
- **Sample Rate**: 22050 Hz → 24000 Hz
- **Installation**: pip → uv for faster setup

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) by Blaizzy
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [Wyoming Protocol](https://github.com/rhasspy/wyoming) by Rhasspy
- Original [wyoming-piper](https://github.com/rhasspy/wyoming-piper) by Michael Hansen

## License

MIT License - See LICENSE file for details
