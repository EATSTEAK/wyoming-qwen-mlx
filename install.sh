#!/usr/bin/env bash
set -e

# Install script for wyoming-qwen using uv
# Requires: Apple Silicon Mac (M1/M2/M3/M4)

echo "Installing wyoming-qwen with uv..."

# Check platform
if [[ $(uname -m) != "arm64" ]]; then
    echo "Error: This package requires Apple Silicon (M1/M2/M3/M4)"
    echo "Detected platform: $(uname -m)"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Create virtual environment and install
uv venv
source .venv/bin/activate

# Install dependencies with uv
uv pip install -e ".[zeroconf]"

echo ""
echo "Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run wyoming-qwen:"
echo "  wyoming-qwen --speaker Ryan --data-dir /tmp/wyoming-qwen --uri tcp://0.0.0.0:10200"
