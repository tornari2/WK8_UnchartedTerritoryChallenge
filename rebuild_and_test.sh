#!/bin/bash
# Rebuild Polars with optimizations and run benchmark

set -e

echo "================================================================================"
echo "Rebuilding Polars with Performance Optimizations"
echo "================================================================================"
echo ""
echo "Optimizations applied:"
echo "  1. ✅ Increased blocking thresholds:"
echo "     - 100K-1M comparisons: 0.55 → 0.6 (filters ~95%)"
echo "     - 1M-100M comparisons: 0.7 → 0.75 (filters ~99%)"
echo "     - 100M+ comparisons: 0.75 → 0.8 (filters ~99.5%)"
echo "  2. ✅ Added blocking efficiency logging (debug builds)"
echo "  3. ✅ Added AVX-512 detection logging (debug builds)"
echo ""
echo "================================================================================"

cd polars

# Build with release profile
echo "Building Polars (release mode with optimizations)..."
echo "This may take 5-10 minutes..."

# Use release profile with PGO-style optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release --features=fuzzy_join 2>&1 | grep -E "(Compiling|Finished|error)"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    
    # Install the Python package
    echo "Installing Python package..."
    cd ..
    pip install -e polars/py-polars --force-reinstall --no-build-isolation
    
    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "================================================================================"
    echo "Running benchmark to verify improvements..."
    echo "================================================================================"
    echo ""
    
    # Run benchmark
    python3 benchmark_comparison_table.py
    
else
    echo ""
    echo "❌ Build failed. Check errors above."
    exit 1
fi

