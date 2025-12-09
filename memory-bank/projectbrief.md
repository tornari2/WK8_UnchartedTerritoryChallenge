# Project Brief

## Overview
*This document serves as the foundation for all project decisions and documentation. Update this when core requirements or goals change.*

## Project Name
WK8_UnchartedTerritoryChallenge - Polars Native String Similarity Expression Kernels

## Core Requirements
Extend Polars (the high-performance Rust DataFrame library) by adding native string similarity kernels implemented directly in Rust and fully integrated into Polars' expression engine, lazy optimizer, and compute pipeline.

**Key Requirements:**
1. Implement 5 similarity metrics as native Rust kernels:
   - Levenshtein similarity (normalized, 0.0-1.0)
   - Damerau-Levenshtein similarity (OSA variant, normalized)
   - Jaro-Winkler similarity (0.0-1.0)
   - Hamming similarity (normalized for equal-length strings)
   - Cosine similarity for Array<f32> or List<f32> vectors
2. Full integration into Polars expression system (DSL, logical plan, physical execution)
3. Support for eager execution, lazy queries, and streaming
4. Python bindings under `.str` and `.arr` namespaces
5. Comprehensive testing validated against reference implementations (RapidFuzz, NumPy)

## Project Goals
1. **Primary Goal:** Add first-class fuzzy text analytics capabilities to Polars without external Python dependencies
2. **Secondary Goal:** Provide foundation for future fuzzy join operators
3. **Learning Goal:** Work in Rust and contribute to a substantial existing codebase (satisfies Uncharted Territory Challenge requirement)

## Scope

### In Scope
- All 5 similarity metrics as native Rust kernels
- Arrow-aware implementation with proper null handling
- Expression DSL integration (FunctionExpr variants)
- Physical plan integration
- Python bindings
- Comprehensive test suite
- Documentation and examples

### Out of Scope (Phase 1)
- SIMD intrinsics or low-level assembly optimizations (moved to Phase 2)
- Tokenization and text preprocessing
- Fuzzy join operator implementation (future extension) ✅ **NOW COMPLETE in Phase 5**
- Contributing upstream to Polars repository (standalone fork)

### Phase 2: Performance Optimization ✅ COMPLETE
**All 16 Optimization Tasks Complete (Tasks 15-30):**
- SIMD optimizations for character comparison and cosine similarity (auto-vectorization)
- ASCII fast path for common text processing
- Parallel chunk processing with Rayon
- Memory pool optimizations
- Early exit and threshold-based optimizations
- Myers' bit-parallel algorithm for small strings
- Inner loop optimizations and cache-friendly access patterns
- **Task 27: Diagonal Band Optimization for Levenshtein** ✅ CRITICAL SUCCESS

**Explicit SIMD Tasks (28-30):** ✅ COMPLETE
- Task 28: SIMD for Diagonal Band Computation ✅ COMPLETE
- Task 29: Explicit SIMD for Character Comparison ✅ COMPLETE
- Task 30: Explicit SIMD for Cosine Similarity Enhancement ✅ COMPLETE

## Success Criteria
1. All 5 metrics implemented and exposed via Polars expression API
2. Functions work in eager mode, lazy queries, and streaming
3. Null handling and edge cases properly handled
4. Performance significantly better than Python UDF implementations
5. Tests validated against RapidFuzz (strings) and NumPy (cosine)
6. Documentation and examples provided

## Constraints
- **Timeline:** Hard deadline (7-day implementation window)
- **Language:** Must be implemented in Rust (new language for this challenge)
- **Codebase:** Must work within existing Polars architecture
- **Standalone:** This is a fork, not an upstream contribution

## Stakeholders
- Developer: Implementing the feature
- Future users: Data engineers, ML practitioners, analysts who need fuzzy matching in Polars

---

### Phase 3: Final Performance Gap Closure ✅ COMPLETE
**3 Optimization Tasks Completed (Tasks 35-37):**
- ✅ Task 35: Hamming Similarity Small Dataset Optimization (COMPLETE)
- ✅ Task 36: Jaro-Winkler Large Dataset Optimization (COMPLETE)
- ✅ Task 37: General Column-Level Optimizations (COMPLETE)

### Phase 4: Additional Jaro-Winkler Optimizations (Tasks 38-43) ✅ COMPLETE
**6 Optimization Tasks Completed**

### Phase 8: Sparse Vector Blocking (2025-12-05) ✅ COMPLETE
**8 Tasks Completed (Tasks 73-80)**

### Phase 17: RapidFuzz Parity Optimizations ✅ COMPLETE (2025-12-08)
**7 Tasks Completed Based on RapidFuzz-cpp Analysis:**

| Task | Title | Priority | Status | Expected Impact |
|------|-------|----------|--------|-----------------|
| 134 | Common Prefix/Suffix Removal | High | ✅ DONE | 10-50% speedup |
| 135 | MBLEVEN2018 Algorithm | High | ✅ DONE | 2-5x for edit dist ≤3 |
| 136 | Score Hint Doubling | High | ✅ DONE | 2-10x speedup |
| 137 | Small Band Diagonal Shifting | Medium | ✅ DONE | 2-3x speedup |
| 138 | Ukkonen Dynamic Band | Medium | ✅ DONE | 10-30% speedup |
| 139 | SIMD Batch Processing (cdist) | Low | ✅ DONE | 4-8x for fuzzy joins |
| 140 | mbleven2018 Algorithm | High | ✅ DONE | 2-5x for edit dist ≤3 |

---

## Phase 18: Jaro-Winkler Performance Optimization 🔧 ACTIVE (2025-12-09)

### Objective
Close the remaining Jaro-Winkler performance gap vs RapidFuzz:
- **Current:** 0.37x on 100K pairs (2.7x slower than RapidFuzz)
- **Target:** ≥1.0x (match or exceed RapidFuzz)

### Root Cause Analysis
1. **O(n×m) match-finding complexity** - Scanning through match window for each character
2. **Cache pressure at scale** - Thread-local buffers don't scale well
3. **Missing RapidFuzz optimizations** - Position-based preprocessing, SIMD matching

### Phase 18 Tasks (7 Tasks, 43 Subtasks)

| Task | Title | Priority | Expected Speedup |
|------|-------|----------|------------------|
| **141** | Position-Based Character Matching | **HIGH** | 2-3x |
| **142** | AVX2 SIMD Parallel Match Finding | **HIGH** | 1.3-1.5x |
| **143** | Parallel Batch Processing (Rayon) | Medium | 1.5-2x |
| **144** | Early Exit Length-Based Upper Bound | Medium | 10-30% |
| **145** | Cache-Optimized Batch Processing | Medium | 10-20% |
| **146** | Jaro-Winkler for Long Strings (>64) | Medium | 2-3x for long strings |
| **147** | Unified Dispatcher Optimization | Medium | 5-10% |

### Key Techniques

**Task 141: Position-Based Character Matching**
- Replace O(window_size) scanning with O(1) character position lookup
- Pre-build position list mapping character values to positions in shorter string
- Expected 2-3x speedup for medium strings (20-64 chars)

**Task 142: AVX2 SIMD Parallel Match Finding**
- Use `_mm256_cmpeq_epi8` for 32-character comparisons
- Runtime CPU feature detection with SSE2 fallback
- Expected 1.3-1.5x additional speedup

**Task 143: Rayon Parallel Processing**
- Parallelize across CPU cores for ≥10K pairs
- Chunk-based processing for cache locality
- Expected 1.5-2x speedup for 100K+ pairs

### Implementation Order
1. **Task 141** - Highest impact, start here
2. **Task 142** - Can parallelize with Task 141
3. **Task 144** - Quick win for threshold operations
4. **Task 143** - After core optimizations
5. **Task 145** - Cache optimization
6. **Task 146** - Long string support
7. **Task 147** - Final cleanup

---

**Last Updated:** 2025-12-09
**Status:** 🔧 **ACTIVE DEVELOPMENT** - Phase 18 Jaro-Winkler Optimization
**Repository:** https://github.com/tornari2/WK8_UnchartedTerritoryChallenge

**Phase Summary:**
- Phase 1-17: ✅ COMPLETE (140 tasks)
- **Phase 18: 🔧 ACTIVE (7 tasks, 43 subtasks)**
- **Total: 147 tasks**

**Current Performance:**
| Metric | Status | Best Result |
|--------|--------|-------------|
| Damerau-Levenshtein | ✅ Polars wins | 12.51x faster |
| Cosine (100K, dim=30) | ✅ Polars wins | 5.1x faster |
| Levenshtein | ✅ Polars wins | 1.62x faster |
| Hamming | ✅ Polars wins | 4.88x faster |
| **Jaro-Winkler** | ⚠️ **Target for Phase 18** | Currently 0.37x |
