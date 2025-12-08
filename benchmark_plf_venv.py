#!/usr/bin/env python3
"""
Benchmark pl-fuzzy-frame-match with ANN enabled in separate venv.
This script should be run from the plf_venv virtual environment.

Outputs results to a JSON file for combination with Polars benchmarks.
Now also returns matched pairs for precision/recall calculation.
"""

import time
import statistics
import sys
import json
import random
import string
from typing import List, Tuple, Dict, Optional, Set

import polars as pl
import pl_fuzzy_frame_match as plf


def generate_test_data(
    left_rows: int,
    right_rows: int,
    avg_string_length: int = 15,
    match_rate: float = 0.3,
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Generate test DataFrames with controlled similarity levels."""
    random.seed(seed)
    
    def random_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def create_similar_string(original: str, target_similarity: float) -> str:
        if target_similarity >= 0.99:
            return original
        
        chars = list(original)
        if len(chars) == 0:
            return original
        
        string_length = len(chars)
        estimated_edits = int(string_length * (1.0 - target_similarity))
        estimated_edits = max(1, min(estimated_edits, string_length - 1))
        
        modified = chars.copy()
        edits_made = 0
        
        edit_types = ['substitute'] * 3 + ['delete'] * 2
        
        while edits_made < estimated_edits and len(modified) > 0:
            edit_type = random.choice(edit_types)
            
            if edit_type == 'substitute' and len(modified) > 0:
                idx = random.randint(0, len(modified) - 1)
                modified[idx] = random.choice(string.ascii_letters + string.digits)
                edits_made += 1
            elif edit_type == 'delete' and len(modified) > 1:
                idx = random.randint(0, len(modified) - 1)
                modified.pop(idx)
                edits_made += 1
            else:
                break
        
        result = ''.join(modified)
        return result if len(result) > 0 else original
    
    # Generate left DataFrame
    left_names = [random_string(avg_string_length) for _ in range(left_rows)]
    left_df = pl.DataFrame({
        "id": list(range(left_rows)),
        "name": left_names
    })
    
    # Generate right DataFrame with controlled similarity levels
    right_names = []
    right_ids = []
    
    num_matches = int(right_rows * match_rate)
    
    for i in range(num_matches):
        if i < len(left_names):
            similarity_distribution = (i % 10) / 10.0
            
            if similarity_distribution < 0.3:
                target_sim = 0.90 + random.random() * 0.10
            elif similarity_distribution < 0.6:
                target_sim = 0.80 + random.random() * 0.10
            elif similarity_distribution < 0.8:
                target_sim = 0.75 + random.random() * 0.10
            else:
                target_sim = 0.60 + random.random() * 0.15
            
            modified = create_similar_string(left_names[i], target_sim)
            right_names.append(modified)
            right_id = 1000 + i
            right_ids.append(right_id)
    
    num_false_positives = int(right_rows * 0.1)
    for i in range(num_false_positives):
        if i < len(left_names):
            target_sim = 0.70 + random.random() * 0.09
            modified = create_similar_string(left_names[i], target_sim)
            right_names.append(modified)
            right_id = 1000 + num_matches + i
            right_ids.append(right_id)
    
    remaining = right_rows - len(right_names)
    for i in range(remaining):
        right_names.append(random_string(avg_string_length))
        right_id = 1000 + num_matches + num_false_positives + i
        right_ids.append(right_id)
    
    right_df = pl.DataFrame({
        "id": right_ids,
        "name": right_names
    })
    
    return left_df, right_df


def extract_match_pairs(result_df) -> List[Tuple[int, int]]:
    """Extract (left_id, right_id) pairs from a fuzzy match result."""
    if result_df is None or result_df.height == 0:
        return []
    
    # pl-fuzzy-frame-match returns columns like: id, name, id_right, name_right, score
    try:
        left_ids = result_df["id"].to_list()
        right_ids = result_df["id_right"].to_list()
        return list(zip(left_ids, right_ids))
    except Exception as e:
        print(f"    Warning: Could not extract pairs: {e}", file=sys.stderr)
        return []


def benchmark_pl_fuzzy_frame_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    similarity: str = "jaro_winkler",
    threshold: float = 0.8,
    iterations: int = 3,
    use_ann: bool = True
) -> Tuple[Dict[str, float], List[Tuple[int, int]]]:
    """Benchmark pl-fuzzy-frame-match with optional ANN. Returns metrics and matched pairs."""
    times = []
    result_count = 0
    matched_pairs = []
    
    similarity_map = {
        "jaro_winkler": "jaro_winkler",
        "levenshtein": "levenshtein",
        "damerau_levenshtein": "damerau_levenshtein",
    }
    
    plf_similarity = similarity_map.get(similarity, "jaro_winkler")
    threshold_100 = threshold * 100.0
    
    mapping = plf.FuzzyMapping(
        left_col="name",
        right_col="name",
        threshold_score=threshold_100,
        fuzzy_type=plf_similarity
    )
    
    comparisons = left.height * right.height
    
    for i in range(iterations):
        start = time.perf_counter()
        try:
            if use_ann:
                result = plf.fuzzy_match_dfs(
                    left.lazy(),
                    right.lazy(),
                    [mapping],
                )
            else:
                result = plf.fuzzy_match_dfs(
                    left.lazy(),
                    right.lazy(),
                    [mapping],
                    use_appr_nearest_neighbor_for_new_matches=False,
                )
            result_height = result.height
        except Exception as e:
            print(f"    Warning: pl-fuzzy-frame-match failed: {e}", file=sys.stderr)
            result_height = 0
            result = None
        end = time.perf_counter()
        times.append(end - start)
        
        if i == 0:
            # Extract pairs from first iteration
            if result is not None:
                matched_pairs = extract_match_pairs(result)
        
        if result_count == 0:
            result_count = result_height
    
    mean_time = statistics.mean(times)
    
    metrics = {
        "mean": mean_time,
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "result_count": result_count,
        "comparisons": comparisons,
        "throughput": comparisons / mean_time if mean_time > 0 else 0,
        "ann_enabled": use_ann
    }
    
    return metrics, matched_pairs


def main():
    print("=" * 80, file=sys.stderr)
    print("pl-fuzzy-frame-match Benchmark (with ANN enabled)", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Polars version: {pl.__version__}", file=sys.stderr)
    print(f"pl-fuzzy-frame-match version: {plf.__version__}", file=sys.stderr)
    
    # Test configurations - matching main benchmark
    test_configs = [
        (100, 100, "Tiny (10K)"),
        (1000, 1000, "Small (1M)"),
        (2000, 2000, "Medium (4M)"),
        (4000, 4000, "Large (16M)"),
        (10000, 10000, "XLarge (100M)"),
    ]
    
    similarities = ["jaro_winkler", "levenshtein", "damerau_levenshtein"]
    
    thresholds = {
        "jaro_winkler": 0.8,
        "levenshtein": 0.7,
        "damerau_levenshtein": 0.7
    }
    
    all_results = []
    
    for left_rows, right_rows, description in test_configs:
        print(f"\nTesting {description}: {left_rows:,} × {right_rows:,}", file=sys.stderr)
        
        for similarity in similarities:
            threshold = thresholds[similarity]
            
            print(f"  {similarity} (threshold={threshold})...", file=sys.stderr)
            
            # Generate test data with same seed as main benchmark
            left_df, right_df = generate_test_data(
                left_rows, right_rows, 
                avg_string_length=15, 
                match_rate=0.3,
                seed=42
            )
            
            # Benchmark with ANN enabled (default)
            result, matched_pairs = benchmark_pl_fuzzy_frame_match(
                left_df, right_df,
                similarity=similarity,
                threshold=threshold,
                iterations=3,
                use_ann=True
            )
            
            # Only include pairs for smaller datasets to avoid huge JSON
            include_pairs = (left_rows * right_rows) <= 4_000_000
            
            all_results.append({
                "similarity": similarity,
                "left_rows": left_rows,
                "right_rows": right_rows,
                "description": description,
                "comparisons": left_rows * right_rows,
                "plf": result,
                "matched_pairs": matched_pairs if include_pairs else [],
                "polars_version": pl.__version__,
                "plf_version": plf.__version__
            })
            
            print(f"    Time: {result['mean']:.4f}s, Results: {result['result_count']:,}", file=sys.stderr)
    
    # Output results as JSON to stdout
    print(json.dumps(all_results, indent=2))
    
    print("\nBenchmark complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
