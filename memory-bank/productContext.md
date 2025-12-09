# Product Context

## Why This Project Exists
Polars currently lacks any built-in support for string similarity or fuzzy matching. Users who need such functionality must write slow Python UDFs, use external libraries and precompute scores, or implement their own Rust extensions. This project adds native, high-performance string similarity capabilities directly into Polars.

## Problems It Solves
- **Slow Python UDFs:** Current workarounds break lazy execution and are non-vectorized
- **Complex Workflows:** Users must precompute similarity scores outside Polars and join results back
- **Limited Capabilities:** No native fuzzy matching makes data cleaning, deduplication, and record linkage difficult
- **ML Feature Engineering:** No easy way to add similarity-based numeric features for machine learning pipelines
- **Future Extensibility:** Foundation needed for future fuzzy join operators

## How It Should Work

### User Journey
1. **Data Cleaning:** User has a DataFrame with customer names that may have typos or variations
   - Uses `df.select(pl.col("name").str.jaro_winkler_sim(pl.lit("canonical_name")))` to find similar names
   - Filters by similarity threshold to identify duplicates

2. **Record Linkage:** Analyst has two tables with company names that may be spelled differently
   - Computes similarity scores between columns from both tables
   - Uses scores to identify likely matches for manual review or automated joining

3. **ML Feature Engineering:** ML practitioner building entity matching model
   - Adds similarity-based features: `df.with_columns([pl.col("name1").str.levenshtein_sim(pl.col("name2")).alias("lev_sim")])`
   - Feeds similarity scores into external model training pipeline

4. **Embedding Search:** Developer stores text embeddings as Array<f32> columns
   - Computes cosine similarity between query vector and all rows
   - Ranks results by similarity for semantic search

5. **Fuzzy Joins:** Data engineer needs to link records across datasets with noisy keys
   - Uses `df.fuzzy_join(other, left_on="name", right_on="customer_name", similarity="jaro_winkler", threshold=0.85)`
   - Automatically finds best matches based on string similarity
   - Returns joined DataFrame with `_similarity` column showing match scores
   - **Advanced (Tasks 64-67):** LSH blocking for 95-99% comparison reduction, batch processing for datasets larger than RAM, progressive results for faster time-to-first-result

### Key Features
- **5 Similarity Metrics:** Levenshtein, Damerau-Levenshtein, Jaro-Winkler, Hamming, Cosine
- **Normalized Scores:** All metrics return 0.0-1.0 (except cosine which is -1.0 to 1.0)
- **Expression API:** Natural integration with Polars expression DSL
- **Fuzzy Join API:** DataFrame-level fuzzy join with multiple similarity metrics and join types
- **Blocking Strategies:** FirstNChars, NGram, Length, SortedNeighborhood, Multi-Column, LSH, SparseVector ✅ COMPLETE
- **Sparse Vector Blocking:** TF-IDF weighted n-gram vectors with cosine similarity ✅ COMPLETE
- **BK-Tree + Sparse Vector Hybrid:** 100% recall for high-threshold edit distance ✅ COMPLETE
- **Batch Processing:** Memory-efficient processing for datasets larger than RAM ✅ COMPLETE
- **Progressive Results:** Streaming results with early termination ✅ COMPLETE
- **Persistent Indices:** Reusable blocking indices for repeated joins ✅ COMPLETE
- **Null Safety:** Proper handling of null inputs and edge cases
- **Performance:** Native Rust implementation, significantly faster than Python UDFs
- **Streaming Compatible:** Works with Polars streaming engine

### Performance Optimizations ✅ ALL COMPLETE (Phases 1-17)

**Phase 2 Optimizations:**
- **ASCII Fast Path:** 2-5x faster for ASCII-only strings
- **Early Exit Optimizations:** 1.5-3x faster for mismatched strings
- **Parallel Processing:** 2-4x faster on multi-core systems
- **Buffer Reuse:** 10-20% reduced allocation overhead
- **SIMD Operations:** 3-5x faster vector operations for cosine similarity
- **Explicit SIMD:** Additional 2-4x speedup with std::simd
- **Inline Optimization:** 10-30% faster from reduced function call overhead

**Phase 17 RapidFuzz Parity Optimizations ✅ COMPLETE (2025-12-08):**
Based on analysis of RapidFuzz-cpp, these key algorithms close remaining performance gaps:
- ✅ **Common Prefix/Suffix Removal:** 10-50% speedup by reducing effective string length
- ✅ **mbleven2018 Algorithm:** O(1) lookup for edit distances ≤3 (2-5x speedup for similar strings)
- ✅ **Score Hint Doubling:** Iterative band widening avoids full matrix computation (2-10x speedup)
- ✅ **Small Band Diagonal Shifting:** Right-shift formulation for bands ≤64 (2-3x speedup)
- ✅ **Ukkonen Dynamic Band:** Adaptive first_block/last_block (10-30% additional speedup)
- ✅ **SIMD Batch Processing:** Process 4-16 string pairs simultaneously (4-8x for fuzzy joins)

**Task 141: Direct Array Processing ✅ COMPLETE (2025-12-09):**
- ✅ **Direct Array Access:** Bypass `get_as_series()` overhead for 7.7x cosine speedup
- ✅ **SIMD Wrapping Add Fix:** Use `+` operator for std::simd unsigned integer types

## User Experience Goals
- **Intuitive:** Functions feel natural in Polars expression API
- **Fast:** Users notice significant performance improvement over Python UDFs
- **Reliable:** Proper null handling and edge case behavior matches user expectations
- **Discoverable:** Functions appear in IDE autocomplete and documentation
- **Competitive:** Performance on par with or better than specialized libraries like RapidFuzz
- **RapidFuzz Parity:** ✅ ACHIEVED for most metrics - Phase 18 targets remaining Jaro-Winkler gap

## Target Audience
- **Data Engineers:** Cleaning and deduplicating messy text data
- **Data Analysts:** Linking records across datasets with noisy keys
- **ML Practitioners:** Engineering similarity-based features for models
- **Developers:** Building applications with fuzzy search and matching capabilities

## Performance Benchmarks

### Current Performance Results (2025-12-09)

**Element-wise Similarity vs RapidFuzz:**
| Metric | 1K Rows | 10K Rows | 100K Rows |
|--------|---------|----------|-----------|
| **Hamming** | ✅ 3.29x faster | ✅ 4.88x faster | ✅ 4.09x faster |
| **Levenshtein** | ≈ 1.03x | ✅ 1.46x faster | ✅ 1.62x faster |
| **Damerau-Lev** | ✅ 1.83x faster | ✅ 7.15x faster | ✅ 12.51x faster |
| **Jaro-Winkler** | ✅ 1.33x faster | ❌ 0.77x | ❌ **0.37x** ⬅️ Phase 18 Target |

**Vector Similarity vs NumPy (FIXED 2025-12-09):**
| Metric | 1K, dim=10 | 10K, dim=20 | 100K, dim=30 |
|--------|------------|-------------|--------------|
| **Cosine** | ❌ NumPy 3.4x faster | ✅ 1.69x faster | ✅ **5.10x faster** |

**Key Observations:**
- **Hamming:** Polars wins at all scales (3-5x faster)
- **Levenshtein:** Polars wins at medium/large scales (1.5-1.6x faster)
- **Damerau-Levenshtein:** Polars wins big (2-12x faster) - MAJOR SUCCESS
- **Jaro-Winkler:** Polars wins at small scale, **RapidFuzz wins at large scale** ⬅️ Phase 18 Target
- **Cosine:** ✅ **FIXED!** Polars now wins at medium/large scales (1.7-5.1x faster)

### Benchmark Summary (Pre-Phase 18)
| Result | Count | Percentage |
|--------|-------|------------|
| ✅ **Polars Wins** | **11** | **73%** |
| ❌ RapidFuzz/NumPy Wins | 4 | 27% |
| **Average Speedup** | **3.16x** | - |
| **Max Speedup** | **12.51x** | Damerau-Levenshtein 100K |

### Phase 18 Target
| Metric | Current | Target After Phase 18 |
|--------|---------|----------------------|
| Jaro-Winkler 10K | 0.77x | ≥1.0x |
| Jaro-Winkler 100K | 0.37x | ≥1.0x |
| Overall Win Rate | 73% (11/15) | 87%+ (13/15) |

### Performance vs pl-fuzzy-frame-match
**pl-fuzzy-frame-match version:** 0.4.0 (running in separate venv with ANN enabled)
**Custom Polars version:** 1.36.0-beta.1

| Algorithm | Avg Speedup | Best (at 100M comparisons) |
|-----------|------------|---------------------------|
| **Jaro-Winkler** | 4.13x faster | 7.73x faster |
| **Levenshtein** | 5.83x faster | 10.83x faster |
| **Damerau-Levenshtein** | 1.81x faster | 3.98x faster |

**Key Finding:** Polars scales significantly better than pl-fuzzy for large datasets (10K+ rows).

---

**Last Updated:** 2025-12-09
**Status:** 🔧 **ACTIVE DEVELOPMENT** - Phase 18 Jaro-Winkler Optimization
**Repository:** https://github.com/tornari2/WK8_UnchartedTerritoryChallenge
**Documentation:** README.md includes setup instructions, architecture overview, technical decisions, and performance benchmarks
- All 5 similarity metrics implemented and production-ready
- Fuzzy join functionality fully available via `df.fuzzy_join()` method
- 177+ tests passing (120 similarity + 14 fuzzy join + 43 batch/LSH/index tests)
- **Phase 18:** 7 new tasks to close Jaro-Winkler performance gap
