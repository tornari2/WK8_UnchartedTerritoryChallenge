#!/usr/bin/env python3
"""
Quick test script to verify optimization fixes are working.
This runs a single small benchmark to quickly check if improvements are active.
"""

import time
import polars as pl

def generate_test_data(n=1000):
    """Generate simple test data."""
    import random
    import string
    random.seed(42)
    
    def random_string(length=15):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    left = pl.DataFrame({
        "id": list(range(n)),
        "name": [random_string() for _ in range(n)]
    })
    
    right = pl.DataFrame({
        "id": list(range(1000, 1000 + n)),
        "name": [random_string() for _ in range(n)]
    })
    
    return left, right


def test_fuzzy_join():
    """Quick test of fuzzy join performance."""
    print("=" * 80)
    print("Quick Fuzzy Join Test")
    print("=" * 80)
    
    print("\nGenerating test data (1K×1K)...")
    left, right = generate_test_data(1000)
    
    print("Running fuzzy join with Jaro-Winkler...")
    start = time.perf_counter()
    result = left.fuzzy_join(
        right,
        left_on="name",
        right_on="name",
        similarity="jaro_winkler",
        threshold=0.8,
        how="inner",
        keep="all"
    )
    elapsed = time.perf_counter() - start
    
    print(f"\n✅ Test completed successfully!")
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Matches found: {result.height}")
    print(f"   Throughput: {(1000 * 1000) / elapsed:,.0f} comparisons/sec")
    
    print("\nNote: For detailed logging, build in debug mode:")
    print("  cd polars && cargo build --features=fuzzy_join")
    print("  pip install -e polars/py-polars --force-reinstall")
    print("\nThen run again to see blocking and SIMD logs.")


if __name__ == "__main__":
    test_fuzzy_join()

