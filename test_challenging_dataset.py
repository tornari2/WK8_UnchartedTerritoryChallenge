#!/usr/bin/env python3
"""
Test with a more challenging dataset that better reflects real-world scenarios:
- Multiple matches per left row (some above threshold, some below)
- Ambiguous cases (multiple similar strings)
- False positive opportunities
"""

import sys
import random
import string
from typing import Dict, Tuple, List

try:
    import polars as pl
except ImportError:
    print("Error: polars not installed")
    sys.exit(1)

try:
    import pl_fuzzy_frame_match as plf
    PLF_AVAILABLE = True
except ImportError:
    PLF_AVAILABLE = False
    print("Warning: pl-fuzzy-frame-match not available")


def generate_challenging_test_data(
    left_rows: int = 100,
    right_rows: int = 200,  # More right rows to create ambiguity
    avg_string_length: int = 15,
    similarity_threshold: float = 0.8
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[Tuple[int, int], bool]]:
    """
    Generate a challenging test dataset with:
    - Multiple potential matches per left row
    - Some matches just above threshold, some just below
    - False positive opportunities
    """
    random.seed(42)
    
    def random_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def create_variant(original: str, target_sim: float) -> str:
        """Create a string variant with approximately target similarity."""
        if target_sim >= 0.99:
            return original
        
        chars = list(original)
        if len(chars) == 0:
            return original
        
        # Estimate number of edits needed
        edits = max(1, int(len(chars) * (1.0 - target_sim)))
        edits = min(edits, len(chars) - 1)
        
        modified = chars.copy()
        for _ in range(edits):
            if len(modified) <= 1:
                break
            if random.random() < 0.7:  # 70% substitution, 30% deletion
                idx = random.randint(0, len(modified) - 1)
                modified[idx] = random.choice(string.ascii_letters + string.digits)
            else:
                idx = random.randint(0, len(modified) - 1)
                modified.pop(idx)
        
        return ''.join(modified) if len(modified) > 0 else original
    
    # Generate left DataFrame
    left_names = [random_string(avg_string_length) for _ in range(left_rows)]
    left_df = pl.DataFrame({
        "id": list(range(left_rows)),
        "name": left_names
    })
    
    # Generate right DataFrame with multiple matches per left row
    right_names = []
    right_ids = []
    all_potential_matches = {}  # left_id -> [(right_id, target_sim, actual_sim)]
    
    right_id_counter = 1000
    
    # For each left row, create multiple potential matches
    for left_id in range(left_rows):
        original = left_names[left_id]
        
        # Create 2-3 matches per left row with different similarity levels
        num_matches = random.randint(2, 3)
        for match_idx in range(num_matches):
            # Vary similarity: some above threshold, some below
            if match_idx == 0:
                # Best match: well above threshold
                target_sim = 0.85 + random.random() * 0.10  # 0.85-0.95
            elif match_idx == 1:
                # Second match: borderline (might be above or below)
                target_sim = 0.75 + random.random() * 0.10  # 0.75-0.85
            else:
                # Third match: below threshold
                target_sim = 0.60 + random.random() * 0.15  # 0.60-0.75
            
            variant = create_variant(original, target_sim)
            right_names.append(variant)
            right_ids.append(right_id_counter)
            
            # We'll calculate actual similarity later using Polars
            if left_id not in all_potential_matches:
                all_potential_matches[left_id] = []
            all_potential_matches[left_id].append((right_id_counter, target_sim))
            
            right_id_counter += 1
    
    # Add some random strings (noise)
    remaining = right_rows - len(right_names)
    for _ in range(remaining):
        right_names.append(random_string(avg_string_length))
        right_ids.append(right_id_counter)
        right_id_counter += 1
    
    right_df = pl.DataFrame({
        "id": right_ids,
        "name": right_names
    })
    
    # Calculate ACTUAL similarity using Polars to build ground truth
    print("Calculating actual similarities using Polars...")
    result_all = left_df.fuzzy_join(
        right_df,
        left_on="name",
        right_on="name",
        similarity="jaro_winkler",
        threshold=0.0,  # Very low to get all scores
        how="inner",
        keep="all"
    )
    
    # Build ground truth: ALL matches above threshold (not just one per left row)
    ground_truth = {}
    matches_per_left = {}
    
    for row in result_all.iter_rows(named=True):
        left_id = row["id"]
        right_id = row["id_right"]
        actual_sim = row.get("_similarity", 0.0)
        
        if actual_sim >= similarity_threshold:
            ground_truth[(left_id, right_id)] = True
        
        if left_id not in matches_per_left:
            matches_per_left[left_id] = []
        matches_per_left[left_id].append((right_id, actual_sim))
    
    print(f"Ground truth: {len(ground_truth)} matches above threshold={similarity_threshold}")
    print(f"Left rows with matches: {len(matches_per_left)}")
    
    # Show distribution
    left_rows_with_multiple = sum(1 for matches in matches_per_left.values() 
                                   if len([m for m in matches if m[1] >= similarity_threshold]) > 1)
    print(f"Left rows with multiple matches above threshold: {left_rows_with_multiple}")
    
    return left_df, right_df, ground_truth


def test_with_challenging_data():
    """Test with challenging dataset."""
    print("=" * 80)
    print("TESTING WITH CHALLENGING DATASET")
    print("=" * 80)
    
    threshold = 0.8
    
    print("\nGenerating challenging test data...")
    left_df, right_df, ground_truth = generate_challenging_test_data(
        left_rows=100,
        right_rows=200,
        avg_string_length=15,
        similarity_threshold=threshold
    )
    
    print(f"\nDataset: {left_df.height} left rows × {right_df.height} right rows")
    print(f"Ground truth: {len(ground_truth)} matches (ALL matches above threshold)")
    
    # Test Polars
    print("\n" + "-" * 80)
    print("Testing Polars...")
    polars_result = left_df.fuzzy_join(
        right_df,
        left_on="name",
        right_on="name",
        similarity="jaro_winkler",
        threshold=threshold,
        how="inner",
        keep="all"
    )
    
    polars_pairs = set()
    for row in polars_result.iter_rows(named=True):
        polars_pairs.add((row["id"], row["id_right"]))
    
    polars_tp = sum(1 for pair in polars_pairs if ground_truth.get(pair, False))
    polars_fp = len(polars_pairs) - polars_tp
    polars_fn = sum(1 for pair, is_match in ground_truth.items() 
                    if is_match and pair not in polars_pairs)
    
    polars_precision = polars_tp / (polars_tp + polars_fp) if (polars_tp + polars_fp) > 0 else 0.0
    polars_recall = polars_tp / (polars_tp + polars_fn) if (polars_tp + polars_fn) > 0 else 0.0
    
    print(f"Polars found: {len(polars_pairs)} matches")
    print(f"  True Positives: {polars_tp}")
    print(f"  False Positives: {polars_fp}")
    print(f"  False Negatives: {polars_fn}")
    print(f"  Precision: {polars_precision:.4f}")
    print(f"  Recall: {polars_recall:.4f}")
    
    # Test pl-fuzzy if available
    if PLF_AVAILABLE:
        print("\n" + "-" * 80)
        print("Testing pl-fuzzy...")
        threshold_100 = threshold * 100.0
        
        mapping = plf.FuzzyMapping(
            left_col="name",
            right_col="name",
            threshold_score=threshold_100,
            fuzzy_type="jaro_winkler"
        )
        
        plf_result = plf.fuzzy_match_dfs(
            left_df.lazy(),
            right_df.lazy(),
            [mapping],
            use_appr_nearest_neighbor_for_new_matches=False,
        )
        
        plf_pairs = set()
        for row in plf_result.iter_rows(named=True):
            plf_pairs.add((row["id"], row["id_right"]))
        
        plf_tp = sum(1 for pair in plf_pairs if ground_truth.get(pair, False))
        plf_fp = len(plf_pairs) - plf_tp
        plf_fn = sum(1 for pair, is_match in ground_truth.items() 
                     if is_match and pair not in plf_pairs)
        
        plf_precision = plf_tp / (plf_tp + plf_fp) if (plf_tp + plf_fp) > 0 else 0.0
        plf_recall = plf_tp / (plf_tp + plf_fn) if (plf_tp + plf_fn) > 0 else 0.0
        
        print(f"pl-fuzzy found: {len(plf_pairs)} matches")
        print(f"  True Positives: {plf_tp}")
        print(f"  False Positives: {plf_fp}")
        print(f"  False Negatives: {plf_fn}")
        print(f"  Precision: {plf_precision:.4f}")
        print(f"  Recall: {plf_recall:.4f}")
        
        # Compare
        print("\n" + "-" * 80)
        print("COMPARISON:")
        print(f"  Precision: Polars={polars_precision:.4f}, pl-fuzzy={plf_precision:.4f}")
        print(f"  Recall: Polars={polars_recall:.4f}, pl-fuzzy={plf_recall:.4f}")
        
        common = polars_pairs & plf_pairs
        only_polars = polars_pairs - plf_pairs
        only_plf = plf_pairs - polars_pairs
        
        print(f"\n  Common matches: {len(common)}")
        print(f"  Only Polars: {len(only_polars)}")
        print(f"  Only pl-fuzzy: {len(only_plf)}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("If Polars has perfect precision/recall while pl-fuzzy doesn't,")
    print("it might be because:")
    print("1. Ground truth is based on Polars' similarity calculation")
    print("2. Test dataset is too simple (one match per left row)")
    print("3. pl-fuzzy might have slightly different similarity implementation")
    print("=" * 80)


if __name__ == "__main__":
    test_with_challenging_data()

