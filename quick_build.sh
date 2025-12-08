#!/bin/bash
# Quick build script that runs in background and shows progress

set -e

cd "$(dirname "$0")/polars/py-polars/runtime/polars-runtime-32"

echo "Starting build in background..."
echo "This will take 8-15 minutes for release build"
echo ""

# Kill any existing builds
pkill -f "maturin build" 2>/dev/null || true
pkill -f "cargo.*fuzzy_join" 2>/dev/null || true
sleep 1

# Start build in background
nohup maturin build --release --features fuzzy_join > /tmp/maturin_build.log 2>&1 &
BUILD_PID=$!

echo "Build started (PID: $BUILD_PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f /tmp/maturin_build.log"
echo ""
echo "Check if still running:"
echo "  ps aux | grep $BUILD_PID"
echo ""
echo "When complete, install with:"
echo "  pip install polars/target/wheels/polars_runtime_32-*.whl --force-reinstall"
echo ""

# Show initial progress
sleep 3
echo "Initial build output:"
tail -10 /tmp/maturin_build.log 2>/dev/null || echo "Build log not ready yet"

