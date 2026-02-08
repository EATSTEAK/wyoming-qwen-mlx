#!/usr/bin/env bash

# ⚠️ Docker is not supported for wyoming-qwen
#
# wyoming-qwen requires Apple Silicon and MLX framework,
# which are not available in Docker containers (Linux).
#
# Please use native macOS installation:
#   bash install.sh
#
# Then run directly:
#   source .venv/bin/activate
#   wyoming-qwen --speaker Ryan --data-dir /data --uri tcp://0.0.0.0:10200

echo "❌ ERROR: Docker is not supported for wyoming-qwen"
echo ""
echo "This package requires Apple Silicon (M1/M2/M3/M4) and macOS."
echo "Please install and run natively on macOS."
echo ""
echo "Installation:"
echo "  bash install.sh"
echo ""
echo "Usage:"
echo "  source .venv/bin/activate"
echo "  wyoming-qwen --speaker Ryan --data-dir /data --uri tcp://0.0.0.0:10200"
echo ""
exit 1
