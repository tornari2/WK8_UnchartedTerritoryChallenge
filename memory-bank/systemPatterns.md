# System Patterns

> **📚 Comprehensive Architecture Reference:** See [`polarsArchitecture.md`](./polarsArchitecture.md) for complete Polars architecture discovery including crate hierarchy, execution flow, optimization passes, and integration points.

## Architecture Overview
The implementation follows Polars' existing architecture for adding new expression functions:

```
User Code (Python/Rust)
    ↓
Expression DSL (.str.levenshtein_sim())
    ↓
Logical Plan (FunctionExpr::StringSimilarity)
    ↓
Physical Plan (Physical Expression Builder)
    ↓
Compute Kernel (Rust implementation)
    ↓
Arrow Arrays (ChunkedArray with null bitmaps)
```

## Implementation Path (Confirmed from Task 1)

The complete integration path for similarity functions has been confirmed through codebase review:

1. **Kernel Implementation:**
   - `polars-ops/src/chunked_array/strings/similarity.rs` - String similarity kernels
   - `polars-ops/src/chunked_array/array/similarity.rs` - Vector similarity kernels
   - Pattern: Functions take `&StringChunked`/`&ArrayChunked`, return `Float32Chunked`
   - Use `broadcast_binary_elementwise()` for column-to-column operations
   - Handle null bitmaps via `unary_elementwise()` / `binary_elementwise()` helpers

2. **FunctionExpr Enum:**
   - `polars-plan/src/dsl/function_expr/strings.rs` - Add variants to `StringFunction` enum
   - `polars-plan/src/dsl/function_expr/array.rs` - Add variant to `ArrayFunction` enum
   - Must implement `Display` trait for error messages
   - Must implement `From<T> for FunctionExpr` conversion

3. **IR FunctionExpr:**
   - `polars-plan/src/plans/aexpr/function_expr/strings.rs` - Add `IRStringFunction` variants
   - `polars-plan/src/plans/aexpr/function_expr/array.rs` - Add `IRArrayFunction` variant
   - Used in optimized query plans

4. **DSL Methods:**
   - `polars-plan/src/dsl/string.rs` - Add methods to `StringNameSpace` impl
   - `polars-plan/src/dsl/array.rs` - Add method to `ArrayNameSpace` impl
   - Use `self.0.map_binary()` pattern for binary operations
   - Methods return `Expr::Function` with appropriate `FunctionExpr` variant

5. **Physical Execution:**
   - `polars-expr/src/dispatch/strings.rs` - Route `IRStringFunction` to kernels
   - `polars-expr/src/dispatch/array.rs` - Route `IRArrayFunction` to kernels
   - Use `map_as_slice!()` macro for column-to-column operations
   - Handle column-to-literal via `map!()` macro

6. **Python Bindings:**
   - `polars-python/src/expr/string.rs` - Add `str_*_sim()` methods to `PyExpr`
   - `polars-python/src/expr/array.rs` - Add `arr_cosine_similarity()` method
   - Wrap DSL methods, return `PyResult<PyExpr>`
   - Python side: `py-polars/src/polars/expr/string.py` - Add to `ExprStringNameSpace`

## Key Technical Decisions

### Decision: Polars Repository Integration
- **Context:** Polars repository was cloned for modification within this project.
- **Decision:** Removed nested `.git` directory from `polars/` subdirectory to integrate it directly into the main repository.
- **Rationale:** Allows all modifications to be tracked in the main WK8_UnchartedTerritoryChallenge repository.

### Decision: Unicode Handling - Codepoint-Level Operations
- **Context:** Strings can be measured at byte, codepoint, or grapheme level
- **Decision:** Operate on Unicode codepoints using Rust's `.chars()` iterator
- **Rationale:** Matches user intuition and reference implementations (RapidFuzz, jellyfish)

### Decision: Normalized Similarity Scores
- **Context:** Raw distances vs normalized similarity scores
- **Decision:** Return normalized similarity scores (0.0-1.0) for all metrics
- **Rationale:** More useful for ML feature engineering and threshold-based filtering

### Decision: ASCII Fast Path Optimization
- **Context:** Need to improve performance for common ASCII strings
- **Decision:** Implement byte-level operations for ASCII-only strings with fallback to Unicode
- **Rationale:** ASCII strings are common, byte operations are 2-5x faster

### Decision: Thread-Local Buffer Pools
- **Context:** Dynamic programming algorithms allocate Vec buffers repeatedly
- **Decision:** Use `thread_local!` storage for buffer pools
- **Rationale:** Avoids allocation overhead in hot loops, thread-safe by design

### Decision: SIMD Implementation (Tasks 28-30)
- **Context:** Performance targets exceeded, but explicit SIMD could provide additional speedup
- **Decision:** Implement explicit SIMD using std::simd (portable_simd) with feature gating
- **Rationale:** Provides additional 2-4x speedup potential when enabled

## Task 141: Direct Array Processing Pattern ✅ CRITICAL (2025-12-09)

**Problem:** Cosine similarity was slower than NumPy for small vector dimensions because `get_as_series()` per-row overhead dominated computation time.

**Solution:** Bypass Series extraction by directly accessing the flat underlying arrays.

```rust
/// Try to get contiguous f64 values from an ArrayChunked.
/// Returns None if the array has multiple chunks, nulls, or is not f64.
fn try_get_flat_f64_values(ca: &ArrayChunked) -> Option<&[f64]> {
    // Must be single chunk for contiguous access
    if ca.chunks().len() != 1 {
        return None;
    }
    
    // Get the underlying FixedSizeListArray
    let arr = ca.downcast_iter().next()?;
    
    // Check for nulls - fast path only works without nulls
    if arr.validity().is_some() {
        return None;
    }
    
    // Get the flat values array
    let values = arr.values();
    
    // Try to downcast to f64
    let f64_arr = values.as_any().downcast_ref::<Float64Array>()?;
    
    // Must have no nulls in the values
    if f64_arr.validity().is_some() {
        return None;
    }
    
    Some(f64_arr.values())
}

/// Process rows directly from flat arrays - no Series extraction overhead
fn process_rows_direct_f64(
    a_flat: &[f64],
    b_flat: &[f64],
    n_rows: usize,
    width: usize,
) -> Vec<Option<f64>> {
    if n_rows >= PARALLEL_THRESHOLD {
        (0..n_rows)
            .into_par_iter()
            .map(|row| cosine_similarity_from_flat_f64(a_flat, b_flat, row, width))
            .collect()
    } else {
        // Sequential processing for smaller datasets
        (0..n_rows)
            .map(|row| cosine_similarity_from_flat_f64(a_flat, b_flat, row, width))
            .collect()
    }
}

/// Fast cosine similarity from flat slices
#[inline(always)]
fn cosine_similarity_from_flat_f64(a_flat: &[f64], b_flat: &[f64], row: usize, width: usize) -> Option<f64> {
    let start = row * width;
    let end = start + width;
    let a = &a_flat[start..end];
    let b = &b_flat[start..end];
    
    // Use SIMD for dot product and norms
    let (dot, norm_a, norm_b) = dot_product_and_norms_explicit_simd(a, b);
    if norm_a < EPSILON || norm_b < EPSILON {
        return None;
    }
    Some(dot / (norm_a * norm_b))
}
```

**Impact:** 7.7x speedup (7.5M → 57.7M pairs/s), now **5.1x faster than NumPy** on 100K dim=30.

## Phase 17: RapidFuzz Parity Patterns ✅ COMPLETE (2025-12-08)

### Common Prefix/Suffix Removal Pattern (Task 134)
```rust
/// Strip matching prefix/suffix before expensive edit distance computation
#[inline(always)]
fn remove_common_affix<'a>(s1: &'a [u8], s2: &'a [u8]) -> (&'a [u8], &'a [u8], usize, usize) {
    let prefix_len = find_common_prefix_simd(s1, s2);
    let s1_trimmed = &s1[prefix_len..];
    let s2_trimmed = &s2[prefix_len..];
    let suffix_len = find_common_suffix_simd(s1_trimmed, s2_trimmed);
    (&s1_trimmed[..s1_trimmed.len() - suffix_len], 
     &s2_trimmed[..s2_trimmed.len() - suffix_len],
     prefix_len, suffix_len)
}

// SIMD-accelerated prefix detection using u8x32
fn find_common_prefix_simd(s1: &[u8], s2: &[u8]) -> usize {
    // Process 32 bytes at a time for strings > 64 bytes
    // Fallback to scalar for remainder
}
```

### MBLEVEN2018 Algorithm Pattern (Tasks 135, 140)
```rust
/// O(1) lookup for edit distances ≤3 using precomputed table
const MBLEVEN2018_MATRIX: &[u16] = &[/* encoded edit sequences */];

fn levenshtein_mbleven2018(s1: &[u8], s2: &[u8], max_dist: usize) -> Option<usize> {
    if max_dist > 3 { return None; }
    let len_diff = (s1.len() as isize - s2.len() as isize).abs() as usize;
    if len_diff > max_dist { return None; }
    
    // Try each valid edit sequence from lookup table
    for &ops in &MBLEVEN2018_MATRIX[index] {
        if check_edit_sequence(s1, s2, ops) {
            return Some(count_ops(ops));
        }
    }
    None
}
```

### Score Hint Doubling Pattern (Task 136)
```rust
/// Iterative band widening - start small, double on miss
fn levenshtein_distance_iterative_band(s1: &[u8], s2: &[u8], score_cutoff: usize) -> usize {
    let mut score_hint = 31.min(score_cutoff);
    loop {
        if let Some(dist) = levenshtein_distance_banded_with_max(s1, s2, score_hint) {
            return dist;
        }
        if score_hint >= score_cutoff {
            return score_cutoff + 1;
        }
        score_hint = score_hint.saturating_mul(2).min(score_cutoff);
    }
}
```

### Small Band Diagonal Shifting Pattern (Task 137)
```rust
/// Right-shift formulation for bands ≤64 (fits in single u64)
fn levenshtein_small_band_diagonal(s1: &[u8], s2: &[u8], max_distance: usize) -> Option<usize> {
    let mut vp: u64 = !0u64 >> (64 - max_distance - 1);
    let mut vn: u64 = 0;
    let mut score = max_distance;
    
    for &c in s2 {
        let pm = pattern_match_vector[c as usize];
        let d0 = ((pm & vp) + vp) ^ vp | pm | vn;  // Use + instead of wrapping_add
        let hp = vn | !((d0 >> 1) | vp);  // Right-shift formulation
        let hn = (d0 >> 1) & vp;
        
        score += ((hp >> (max_distance - 1)) & 1) as usize;
        score -= ((hn >> (max_distance - 1)) & 1) as usize;
        
        vp = hn | !(d0 | hp);
        vn = d0 & hp;
    }
    Some(score)
}
```

### Ukkonen Dynamic Band Pattern (Task 138)
```rust
/// Dynamic adjustment of first_block and last_block
fn levenshtein_distance_ukkonen(s1: &[u8], s2: &[u8], max_distance: usize) -> Option<usize> {
    let words = (s1.len() + 63) / 64;
    let mut scores = vec![0usize; words];
    let mut first_block = 0;
    let mut last_block = (max_distance / 64).min(words - 1);
    
    for (i, &c) in s2.iter().enumerate() {
        // Process only active blocks
        for block in first_block..=last_block {
            // Myers' algorithm per block...
        }
        
        // Shrink band from start
        while first_block < last_block && scores[first_block] > max_distance {
            first_block += 1;
        }
        
        // Expand band at end if needed
        // ...
    }
}
```

### SIMD Batch Processing Pattern (Task 139)
```rust
/// Process multiple string pairs simultaneously
struct BatchPatternMatchVector {
    patterns: Vec<PatternMatchVector>,
    count: usize,
}

/// Process 4 pairs with AVX2 - use + operator for SIMD addition
fn levenshtein_batch_simd_u64(
    patterns: &[&[u8]],
    texts: &[&[u8]],
    max_distances: &[usize],
) -> Vec<Option<usize>> {
    type SimdU64 = Simd<u64, 4>;
    
    let mut vp = SimdU64::splat(!0u64);
    let mut vn = SimdU64::splat(0);
    let mut scores = SimdU64::splat(0);
    
    // Process all pairs in parallel using SIMD
    // Note: Use `eq_vp + vp` instead of `eq_vp.wrapping_add(vp)` for std::simd
}
```

## Design Patterns Summary

### Kernel Pattern
- Functions take ChunkedArray inputs, return ChunkedArray output
- Handle null bitmaps and multi-chunk columns
- Use `broadcast_binary_elementwise()` for column-to-column operations

### Expression DSL Pattern
- Functions exposed via namespace methods (`.str.xxx_sim()`, `.arr.cosine_similarity()`)
- Methods construct `Expr::Function` with appropriate `FunctionExpr` variant

### Physical Plan Pattern
- Physical expression builder routes `FunctionExpr` variants to concrete kernels
- Handles column-to-column and column-to-literal cases

### ASCII Fast Path Pattern
```rust
if is_ascii_only(a) && is_ascii_only(b) { 
    byte_impl() 
} else { 
    unicode_impl() 
}
```

### Buffer Pool Pattern
```rust
thread_local! {
    static BUFFER_POOL: RefCell<Option<Vec<T>>> = RefCell::new(None);
}
```

### SIMD Pattern
```rust
#[cfg(feature = "simd")]
fn simd_operation(data: &[f64]) -> f64 {
    let chunks = data.chunks_exact(4);
    let mut acc = f64x4::splat(0.0);
    for chunk in chunks {
        let vec = f64x4::from_slice(chunk);
        acc += vec;
    }
    acc.reduce_sum() + chunks.remainder().iter().sum::<f64>()
}
```

### Direct Array Access Pattern (NEW - Task 141)
```rust
// Bypass get_as_series() overhead for contiguous arrays
fn try_process_direct(ca: &ArrayChunked, other: &ArrayChunked, n: usize, width: usize) -> Option<Vec<Option<f64>>> {
    let a_flat = try_get_flat_f64_values(ca)?;
    let b_flat = try_get_flat_f64_values(other)?;
    Some(process_rows_direct_f64(a_flat, b_flat, n, width))
}
```

## Phase 18: Jaro-Winkler Optimization Patterns ✅ COMPLETE (2025-12-09)

### Position-Based Character Matching Pattern (Task 141)
```rust
/// O(1) character position lookup using packed u64 representation
struct CharacterPositionIndex {
    /// Each entry: lower 6 bits = count, remaining bits = packed positions
    positions: [u64; 256],
}

impl CharacterPositionIndex {
    fn new(s: &[u8]) -> Self {
        let mut positions = [0u64; 256];
        for (i, &c) in s.iter().enumerate().take(64) {
            let idx = c as usize;
            let current = positions[idx];
            let count = (current & 0x3F) as usize;
            if count < 10 {
                // Pack position: 6 bits per position after count
                positions[idx] = current | ((i as u64) << (6 + count * 6)) | ((count + 1) as u64);
            }
        }
        Self { positions }
    }
    
    fn find_unmatched(&self, c: u8, start: usize, end: usize, matched_mask: u64) -> Option<usize> {
        let entry = self.positions[c as usize];
        let count = (entry & 0x3F) as usize;
        for i in 0..count {
            let pos = ((entry >> (6 + i * 6)) & 0x3F) as usize;
            if pos >= start && pos < end && (matched_mask & (1u64 << pos)) == 0 {
                return Some(pos);
            }
        }
        None
    }
}
```

### AVX2 SIMD Match Finding Pattern (Task 142)
```rust
/// Find matches using 256-bit SIMD vectors (32 chars at once)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn find_matches_avx2(haystack: &[u8], needle: u8, start: usize, end: usize, matched_mask: u64) -> Option<usize> {
    use std::simd::prelude::*;
    const SIMD_WIDTH: usize = 32;
    type SimdU8 = Simd<u8, SIMD_WIDTH>;
    
    let needle_vec = SimdU8::splat(needle);
    let chunks = (end - start) / SIMD_WIDTH;
    
    for chunk_idx in 0..chunks {
        let offset = start + chunk_idx * SIMD_WIDTH;
        let chunk = SimdU8::from_slice(&haystack[offset..offset + SIMD_WIDTH]);
        let eq_mask = chunk.simd_eq(needle_vec);
        let mut bitmask = eq_mask.to_bitmask() as u64;
        
        // Mask out already-matched positions
        let window_matched = (matched_mask >> offset) & ((1u64 << SIMD_WIDTH) - 1);
        bitmask &= !window_matched;
        
        if bitmask != 0 {
            return Some(offset + bitmask.trailing_zeros() as usize);
        }
    }
    // Scalar fallback for remainder...
    None
}
```

### Parallel Batch Processing Pattern (Task 143)
```rust
/// Rayon-based parallel processing for large datasets
const JARO_WINKLER_PARALLEL_THRESHOLD: usize = 10_000;

pub fn jaro_winkler_parallel(left: &[&[u8]], right: &[&[u8]]) -> Vec<f32> {
    if left.len() < JARO_WINKLER_PARALLEL_THRESHOLD {
        // Sequential for small datasets
        return left.iter().zip(right.iter())
            .map(|(l, r)| jaro_winkler_similarity_bytes_direct(l, r))
            .collect();
    }
    // Parallel for large datasets
    left.par_iter().zip(right.par_iter())
        .map(|(l, r)| jaro_winkler_similarity_bytes_direct(l, r))
        .collect()
}
```

### Early Exit with Length-Based Upper Bound Pattern (Task 144)
```rust
/// Calculate maximum possible Jaro similarity from lengths alone
fn jaro_max_possible(len1: usize, len2: usize) -> f32 {
    let min_len = len1.min(len2);
    let max_len = len1.max(len2);
    let max_matches = min_len as f32;
    
    // Best case: all chars match, no transpositions
    let best_jaro = (max_matches / len1 as f32 + max_matches / len2 as f32 + 1.0) / 3.0;
    let len_bound = (2.0 * (min_len as f32 / max_len as f32) + 1.0) / 3.0;
    
    best_jaro.min(len_bound)
}

/// Early exit if max possible < threshold
pub fn jaro_winkler_with_threshold_early_exit(s1: &[u8], s2: &[u8], threshold: f32) -> Option<f32> {
    let max_possible = jaro_winkler_max_possible(s1.len(), s2.len(), common_prefix);
    if max_possible < threshold {
        return None;  // Early exit - impossible to reach threshold
    }
    // Compute actual similarity...
}
```

### Cache-Optimized Batch Processing Pattern (Task 145)
```rust
/// Process in cache-friendly batches (L2 cache ~1024 pairs)
const CACHE_OPTIMAL_BATCH_SIZE: usize = 1024;

pub fn jaro_winkler_batch_cache_optimized(left: &[&[u8]], right: &[&[u8]], results: &mut [f32]) {
    let num_batches = (left.len() + CACHE_OPTIMAL_BATCH_SIZE - 1) / CACHE_OPTIMAL_BATCH_SIZE;
    
    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * CACHE_OPTIMAL_BATCH_SIZE;
        let batch_end = (batch_start + CACHE_OPTIMAL_BATCH_SIZE).min(left.len());
        
        // Prefetch next batch
        if batch_idx + 1 < num_batches {
            let next_start = (batch_idx + 1) * CACHE_OPTIMAL_BATCH_SIZE;
            for i in next_start..next_start.saturating_add(8).min(left.len()) {
                let _ = left.get(i).map(|s| s.first());
            }
        }
        
        // Process current batch
        for i in batch_start..batch_end {
            results[i] = jaro_winkler_similarity_bytes_direct(left[i], right[i]);
        }
    }
}
```

### u128 Bitmask Pattern for Long Strings (Task 146)
```rust
/// Jaro similarity for 65-128 char strings using u128 bitmasks
fn jaro_similarity_u128(s1: &[u8], s2: &[u8]) -> f32 {
    let mut s1_matches: u128 = 0;
    let mut s2_matches: u128 = 0;
    let mut matches = 0u32;
    
    for i in 0..len1 {
        for j in start..end {
            if (s2_matches & (1u128 << j)) == 0 && s2[j] == needle {
                s1_matches |= 1u128 << i;
                s2_matches |= 1u128 << j;
                matches += 1;
                break;
            }
        }
    }
    
    // Fast transposition counting with u128 bit manipulation
    let mut transpositions = 0u32;
    while s1_remaining != 0 {
        let s1_pos = s1_remaining.trailing_zeros() as usize;
        let s2_pos = s2_remaining.trailing_zeros() as usize;
        if s1[s1_pos] != s2[s2_pos] { transpositions += 1; }
        s1_remaining &= s1_remaining - 1;  // Kernighan's algorithm
        s2_remaining &= s2_remaining - 1;
    }
    // ...
}
```

### Unified Dispatcher Pattern (Task 147)
```rust
/// O(1) path selection via enum-based jump table
enum JaroLengthCategory { Tiny, Short, Medium, Long, VeryLong, Huge }

fn get_length_category(len: usize) -> JaroLengthCategory {
    if len <= 8 { JaroLengthCategory::Tiny }
    else if len <= 20 { JaroLengthCategory::Short }
    else if len <= 64 { JaroLengthCategory::Medium }
    else if len <= 128 { JaroLengthCategory::Long }
    else if len <= 256 { JaroLengthCategory::VeryLong }
    else { JaroLengthCategory::Huge }
}

fn jaro_similarity_bytes_optimized(s1: &[u8], s2: &[u8]) -> f32 {
    match get_length_category(s1.len().max(s2.len())) {
        JaroLengthCategory::Tiny => jaro_similarity_tiny(s1, s2),
        JaroLengthCategory::Short => jaro_similarity_position_based(s1, s2),
        JaroLengthCategory::Medium => jaro_similarity_simd_avx2(s1, s2),
        JaroLengthCategory::Long => jaro_similarity_u128(s1, s2),
        JaroLengthCategory::VeryLong => jaro_similarity_blocked_v2(s1, s2),
        JaroLengthCategory::Huge => jaro_similarity_blocked(s1, s2),
    }
}
```

### DataFrame-Level Batch Integration Pattern
```rust
/// Integrate parallel batch processing into main entry point
pub fn jaro_winkler_similarity(ca: &StringChunked, other: &StringChunked) -> Float32Chunked {
    let meta1 = scan_column_metadata(ca);
    let meta2 = scan_column_metadata(other);
    
    // Use parallel batch for large ASCII datasets
    if meta1.is_all_ascii && meta2.is_all_ascii && ca.len() >= PARALLEL_THRESHOLD {
        // Extract byte slices
        let left_bytes: Vec<&[u8]> = ca.iter().filter_map(|s| s.map(|s| s.as_bytes())).collect();
        let right_bytes: Vec<&[u8]> = other.iter().filter_map(|s| s.map(|s| s.as_bytes())).collect();
        
        // Use parallel processing
        let results = jaro_winkler_parallel(&left_bytes, &right_bytes);
        return Float32Chunked::from_vec(ca.name().clone(), results);
    }
    
    // Fall back to element-wise
    broadcast_binary_elementwise(ca, other, jaro_winkler_similarity_impl)
}
```

---

**Last Updated:** 2025-12-09
**Status:** ✅ **PROJECT COMPLETE** - All 18 Phases Finished
**Recent Updates:**
- ✅ Phase 18: Jaro-Winkler Optimization (7 tasks, 43 subtasks)
- ✅ Position-based character matching (O(1) lookup)
- ✅ AVX2 SIMD parallel match finding
- ✅ Rayon parallel batch processing
- ✅ Length-based early exit for thresholds
- ✅ Cache-optimized batching with prefetching
- ✅ u128 bitmasks for 65-128 char strings
- ✅ Unified dispatcher with enum-based jump table
- ✅ DataFrame-level batch integration
- **Result:** Jaro-Winkler now 1.71x faster than RapidFuzz (was 0.37x)
