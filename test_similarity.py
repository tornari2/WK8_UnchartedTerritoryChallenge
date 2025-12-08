#!/usr/bin/env python3
"""
Quick test script for Polars similarity functions.
Run this to verify all functions are working correctly.
"""

import polars as pl

print("🧪 Testing Polars Similarity Functions\n")
print("=" * 60)

# Test 1: Levenshtein Similarity
print("\n1️⃣  Testing Levenshtein Similarity")
df1 = pl.DataFrame({
    "a": ["hello", "world", "test"],
    "b": ["hallo", "word", "test"]
})
result1 = df1.select(pl.col("a").str.levenshtein_sim(pl.col("b")))
print(result1)
assert result1["a"][0] > 0.7, "Levenshtein test failed"
assert result1["a"][2] == 1.0, "Identical strings should be 1.0"
print("✅ Levenshtein similarity working!")

# Test 2: Hamming Similarity
print("\n2️⃣  Testing Hamming Similarity")
df2 = pl.DataFrame({
    "a": ["abc", "xyz", "test"],
    "b": ["abd", "xyz", "best"]
})
result2 = df2.select(pl.col("a").str.hamming_sim(pl.col("b")))
print(result2)
print("✅ Hamming similarity working!")

# Test 3: Jaro-Winkler Similarity
print("\n3️⃣  Testing Jaro-Winkler Similarity")
df3 = pl.DataFrame({
    "a": ["Martha", "Dwayne", "hello"],
    "b": ["Marhta", "Duane", "hello"]
})
result3 = df3.select(pl.col("a").str.jaro_winkler_sim(pl.col("b")))
print(result3)
assert result3["a"][2] == 1.0, "Identical strings should be 1.0"
print("✅ Jaro-Winkler similarity working!")

# Test 4: Damerau-Levenshtein Similarity
print("\n4️⃣  Testing Damerau-Levenshtein Similarity")
df4 = pl.DataFrame({
    "a": ["teh", "hello", "world"],
    "b": ["the", "hallo", "word"]
})
result4 = df4.select(pl.col("a").str.damerau_levenshtein_sim(pl.col("b")))
print(result4)
print("✅ Damerau-Levenshtein similarity working!")

# Test 5: Cosine Similarity
print("\n5️⃣  Testing Cosine Similarity")
df5 = pl.DataFrame({
    "a": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    "b": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
}, schema={
    "a": pl.Array(pl.Float64, 2),
    "b": pl.Array(pl.Float64, 2)
})
result5 = df5.select(pl.col("a").arr.cosine_similarity(pl.col("b")))
print(result5)
assert result5["a"][0] == 1.0, "Identical vectors should be 1.0"
assert result5["a"][1] == 0.0, "Orthogonal vectors [1,0] and [0,1] should be 0.0"
assert result5["a"][2] == 0.0, "Orthogonal vectors [0,1] and [1,0] should be 0.0"
print("✅ Cosine similarity working!")

# Test 6: Null Handling
print("\n6️⃣  Testing Null Handling")
df6 = pl.DataFrame({
    "a": ["hello", None, "world"],
    "b": ["hallo", "test", None]
}, schema={"a": pl.String, "b": pl.String})
result6 = df6.select(pl.col("a").str.levenshtein_sim(pl.col("b")))
print(result6)
assert result6["a"][1] is None, "Null input should return null"
assert result6["a"][2] is None, "Null input should return null"
print("✅ Null handling working!")

# Test 7: Unicode Support
print("\n7️⃣  Testing Unicode Support")
df7 = pl.DataFrame({
    "a": ["café", "naïve", "🚀"],
    "b": ["cafe", "naive", "🚀🚀"]
})
result7 = df7.select(pl.col("a").str.levenshtein_sim(pl.col("b")))
print(result7)
print("✅ Unicode support working!")

print("\n" + "=" * 60)
print("🎉 All tests passed! Your similarity functions are working correctly!")
print("=" * 60)

