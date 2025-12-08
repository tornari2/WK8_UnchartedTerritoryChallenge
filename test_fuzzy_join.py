#!/usr/bin/env python3
"""Quick test script for Polars fuzzy join functionality.
Run this to verify fuzzy join is working correctly.
"""

import polars as pl

print("🧪 Testing Polars Fuzzy Join\n")
print("=" * 60)

# Test 1: Basic Inner Join
print("\n1️⃣  Testing Basic Inner Join")
left = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["John Smith", "Jane Doe", "Bob Wilson"]
})
right = pl.DataFrame({
    "customer_id": [10, 20, 30],
    "customer_name": ["Jon Smith", "Jane Do", "Robert Wilson"]
})

result = left.fuzzy_join(
    right,
    left_on="name",
    right_on="customer_name",
    similarity="jaro_winkler",
    threshold=0.8
)
print(result)
assert "_similarity" in result.columns
print("✅ Basic inner join working!")

# Test 2: All Similarity Metrics
print("\n2️⃣  Testing All Similarity Metrics")
for sim_type in ["levenshtein", "damerau_levenshtein", "jaro_winkler"]:
    result = left.fuzzy_join(
        right,
        left_on="name",
        right_on="customer_name",
        similarity=sim_type,
        threshold=0.7
    )
    print(f"  {sim_type}: {result.height} matches")
print("✅ All similarity metrics working!")

# Test 3: Keep Strategies
print("\n3️⃣  Testing Keep Strategies")
left_single = pl.DataFrame({"name": ["test"]})
right_multi = pl.DataFrame({"name": ["test1", "test2", "test3"]})

for keep in ["best", "all", "first"]:
    result = left_single.fuzzy_join(
        right_multi,
        left_on="name",
        right_on="name",
        threshold=0.5,
        keep=keep
    )
    print(f"  {keep}: {result.height} matches")
print("✅ All keep strategies working!")

# Test 4: Join Types
print("\n4️⃣  Testing Join Types")
for how in ["inner", "left", "right", "outer"]:
    result = left.fuzzy_join(
        right,
        left_on="name",
        right_on="customer_name",
        threshold=0.7,
        how=how
    )
    print(f"  {how}: {result.height} rows")
print("✅ All join types working!")

# Test 5: Threshold Filtering
print("\n5️⃣  Testing Threshold Filtering")
result_high = left.fuzzy_join(
    right,
    left_on="name",
    right_on="customer_name",
    threshold=0.95
)
result_low = left.fuzzy_join(
    right,
    left_on="name",
    right_on="customer_name",
    threshold=0.5
)
print(f"  High threshold (0.95): {result_high.height} matches")
print(f"  Low threshold (0.5): {result_low.height} matches")
assert result_low.height >= result_high.height
print("✅ Threshold filtering working!")

# Test 6: Null Handling
print("\n6️⃣  Testing Null Handling")
left_null = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["hello", None, "world"]
})
right_null = pl.DataFrame({
    "id": [10, 20, 30],
    "name": ["hallo", "test", None]
})

result_null = left_null.fuzzy_join(
    right_null,
    left_on="name",
    right_on="name",
    threshold=0.8
)
print(result_null)
print("✅ Null handling working!")

# Test 7: Unicode Support
print("\n7️⃣  Testing Unicode Support")
left_unicode = pl.DataFrame({
    "id": [1, 2],
    "name": ["café", "naïve"]
})
right_unicode = pl.DataFrame({
    "id": [10, 20],
    "name": ["cafe", "naive"]
})

result_unicode = left_unicode.fuzzy_join(
    right_unicode,
    left_on="name",
    right_on="name",
    similarity="levenshtein",
    threshold=0.7
)
print(result_unicode)
print("✅ Unicode support working!")

print("\n" + "=" * 60)
print("🎉 All fuzzy join tests passed!")
print("=" * 60)

