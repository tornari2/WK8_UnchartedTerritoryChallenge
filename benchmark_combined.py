#!/usr/bin/env python3
"""
Combined benchmark: Custom Polars fuzzy_join vs pl-fuzzy-frame-match (from separate venv).

This script:
1. Runs Polars fuzzy_join benchmarks (custom build with keep='all')
2. Loads pl-fuzzy-frame-match results from plf_venv_results.json (run in separate venv with standard Polars)
3. Combines results into a comparison table
"""

import time
import statistics
import sys
import os
import json
import random
import string
from typing import List, Tuple, Dict, Optional

import polars as pl


def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Calculate Jaro-Winkler similarity between two strings."""
    jaro = jaro_similarity(s1, s2)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity (normalized edit distance)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    max_len = max(len1, len2)
    return 1.0 - dp[len1][len2] / max_len


def calculate_actual_similarity(s1: str, s2: str, similarity_type: str) -> float:
    if similarity_type == "jaro_winkler":
        return jaro_winkler_similarity(s1, s2)
    elif similarity_type == "levenshtein":
        return levenshtein_similarity(s1, s2)
    elif similarity_type == "damerau_levenshtein":
        return levenshtein_similarity(s1, s2)
    else:
        return jaro_winkler_similarity(s1, s2)


def generate_test_data(
    left_rows: int,
    right_rows: int,
    avg_string_length: int = 15,
    match_rate: float = 0.3,
    similarity_threshold: float = 0.8,
    similarity_type: str = "jaro_winkler",
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[Tuple[int, int], bool]]:
    """Generate test DataFrames with controlled similarity levels."""
    random.seed(seed)
    
    def random_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def create_similar_string(original: str, target_similarity: float, similarity_type: str) -> str:
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
    
    left_names = [random_string(avg_string_length) for _ in range(left_rows)]
    left_df = pl.DataFrame({
        "id": list(range(left_rows)),
        "name": left_names
    })
    
    right_names = []
    right_ids = []
    all_matches = {}
    
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
            
            modified = create_similar_string(left_names[i], target_sim, similarity_type)
            right_names.append(modified)
            right_id = 1000 + i
            right_ids.append(right_id)
            
            actual_sim = calculate_actual_similarity(left_names[i], modified, similarity_type)
            if actual_sim >= similarity_threshold:
                if i not in all_matches:
                    all_matches[i] = []
                all_matches[i].append((right_id, actual_sim))
    
    num_false_positives = int(right_rows * 0.1)
    for i in range(num_false_positives):
        if i < len(left_names):
            target_sim = 0.70 + random.random() * 0.09
            modified = create_similar_string(left_names[i], target_sim, similarity_type)
            right_names.append(modified)
            right_id = 1000 + num_matches + i
            right_ids.append(right_id)
            actual_sim = calculate_actual_similarity(left_names[i], modified, similarity_type)
            if actual_sim >= similarity_threshold:
                if i not in all_matches:
                    all_matches[i] = []
                all_matches[i].append((right_id, actual_sim))
    
    remaining = right_rows - len(right_names)
    for i in range(remaining):
        right_names.append(random_string(avg_string_length))
        right_id = 1000 + num_matches + num_false_positives + i
        right_ids.append(right_id)
    
    right_df = pl.DataFrame({
        "id": right_ids,
        "name": right_names
    })
    
    ground_truth = {}
    for left_id, matches in all_matches.items():
        if matches:
            best_right_id, best_sim = max(matches, key=lambda x: x[1])
            ground_truth[(left_id, best_right_id)] = True
    
    return left_df, right_df, ground_truth


def benchmark_polars_fuzzy_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    similarity: str = "jaro_winkler",
    threshold: float = 0.8,
    iterations: int = 3,
    keep: str = "all"
) -> Tuple[Dict[str, float], Optional[pl.DataFrame]]:
    """Benchmark native Polars fuzzy join."""
    times = []
    result_count = 0
    result_df = None
    
    def do_join():
        return left.fuzzy_join(
            right,
            left_on="name",
            right_on="name",
            similarity=similarity,
            threshold=threshold,
            how="inner",
            keep=keep
        )
    
    for i in range(iterations):
        start = time.perf_counter()
        result = do_join()
        result_height = result.height
        end = time.perf_counter()
        times.append(end - start)
        if i == 0:
            result_df = result
        if result_count == 0:
            result_count = result_height
    
    mean_time = statistics.mean(times)
    comparisons = left.height * right.height
    
    metrics = {
        "mean": mean_time,
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "result_count": result_count,
        "comparisons": comparisons,
        "throughput": comparisons / mean_time if mean_time > 0 else 0
    }
    
    return metrics, result_df


def deduplicate_to_best(result_df: pl.DataFrame, left_id_col: str, score_col: str) -> pl.DataFrame:
    if result_df is None or result_df.height == 0:
        return result_df
    return (
        result_df
        .sort(score_col, descending=True)
        .unique(subset=[left_id_col], keep="first")
    )


def calculate_precision_recall(
    result_df: Optional[pl.DataFrame],
    ground_truth: Dict[Tuple[int, int], bool],
    left_id_col: str = "id",
    right_id_col: str = "id_right",
    deduplicate: bool = True
) -> Tuple[float, float]:
    if result_df is None or result_df.height == 0:
        return 0.0, 0.0
    
    try:
        left_id_col_actual = None
        right_id_col_actual = None
        
        if left_id_col in result_df.columns:
            left_id_col_actual = left_id_col
        if right_id_col in result_df.columns:
            right_id_col_actual = right_id_col
        
        if not left_id_col_actual:
            for col in result_df.columns:
                if col.lower() == "id" or (col.lower().startswith("id") and not col.lower().endswith("_right")):
                    left_id_col_actual = col
                    break
        
        if not right_id_col_actual:
            for col in result_df.columns:
                if col.lower().endswith("_right") and "id" in col.lower():
                    right_id_col_actual = col
                    break
            if not right_id_col_actual:
                id_cols = [col for col in result_df.columns if "id" in col.lower()]
                if len(id_cols) >= 2:
                    right_id_col_actual = id_cols[1]
        
        if not left_id_col_actual or not right_id_col_actual:
            return 0.0, 0.0
        
        score_col_actual = None
        for col in result_df.columns:
            if 'score' in col.lower() or 'similarity' in col.lower() or 'jaro' in col.lower() or 'levenshtein' in col.lower():
                score_col_actual = col
                break
        
        if deduplicate and score_col_actual:
            result_df = deduplicate_to_best(result_df, left_id_col_actual, score_col_actual)
        
        predicted_pairs = set()
        for row in result_df.iter_rows(named=True):
            left_id = row.get(left_id_col_actual)
            right_id = row.get(right_id_col_actual)
            if left_id is not None and right_id is not None:
                try:
                    predicted_pairs.add((int(left_id), int(right_id)))
                except (ValueError, TypeError):
                    continue
        
        true_positives = sum(1 for pair in predicted_pairs if ground_truth.get(pair, False))
        false_positives = len(predicted_pairs) - true_positives
        false_negatives = sum(1 for pair, is_match in ground_truth.items() if is_match and pair not in predicted_pairs)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        return precision, recall
    except Exception as e:
        return 0.0, 0.0


def create_html_table(results: List[Dict], plf_polars_version: str, plf_version: str) -> str:
    """Create an HTML table with benchmark results."""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Polars vs pl-fuzzy-frame-match Benchmark Comparison</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 10px;
        }}
        .note {{
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-bottom: 20px;
            line-height: 1.6;
        }}
        .env-note {{
            background-color: #e3f2fd;
            border: 1px solid #90caf9;
            border-radius: 4px;
            padding: 10px;
            margin: 10px auto;
            max-width: 900px;
            font-size: 11px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px auto;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f5e9;
        }}
        .number {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        .polars {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .plf {{
            color: #f44336;
        }}
        .speedup {{
            color: #FF9800;
            font-weight: bold;
        }}
        .dataset {{
            font-weight: bold;
            color: #673AB7;
        }}
        .section-header {{
            background-color: #34495e !important;
            color: white !important;
            font-weight: bold;
        }}
        .candidate-selection {{
            font-size: 10px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>Polars vs pl-fuzzy-frame-match Benchmark Comparison</h1>
    <div class="subtitle">Performance comparison for Jaro-Winkler, Levenshtein, and Damerau-Levenshtein algorithms</div>
    <div class="note">
        Both libraries use the SAME data, SAME threshold, and return ALL matches above threshold.<br/>
        Results are deduplicated to best match per row for precision/recall calculation.
    </div>
    <div class="env-note">
        <strong>Environment Notes:</strong><br/>
        • <span class="polars">Polars</span>: Custom build v{pl.__version__} with native Rust fuzzy_join (this environment)<br/>
        • <span class="plf">pl-fuzzy-frame-match</span>: v{plf_version} running with standard Polars v{plf_polars_version} (separate venv with ANN enabled)
    </div>
    <table>
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Dataset Size</th>
                <th>Comparisons</th>
                <th>Library</th>
                <th>Candidate Selection</th>
                <th class="number">Time (s)</th>
                <th class="number">Throughput (comp/s)</th>
                <th class="number">Raw Results</th>
                <th class="number">Precision</th>
                <th class="number">Recall</th>
                <th class="number">Speedup</th>
            </tr>
        </thead>
        <tbody>
"""
    
    similarities = ["jaro_winkler", "levenshtein", "damerau_levenshtein"]
    
    for similarity in similarities:
        metric_results = [r for r in results if r["similarity"] == similarity]
        
        if not metric_results:
            continue
        
        similarity_name = similarity.replace("_", "-").title()
        html += f'            <tr class="section-header"><td colspan="11">{similarity_name}</td></tr>\n'
        
        metric_results.sort(key=lambda x: (x["left_rows"], x["right_rows"]))
        
        for result in metric_results:
            dataset_size = f"{result['left_rows']:,} × {result['right_rows']:,}"
            comparisons = f"{result['comparisons']:,}"
            
            polars_time = result["polars"]["mean"]
            polars_throughput = result["polars"]["throughput"]
            polars_results = result["polars"]["result_count"]
            polars_precision = result.get("polars_precision", 0.0)
            polars_recall = result.get("polars_recall", 0.0)
            
            plf_time = result["plf"]["mean"]
            plf_throughput = result["plf"]["throughput"]
            plf_results = result["plf"]["result_count"]
            plf_precision = result.get("plf_precision", 0.0)
            plf_recall = result.get("plf_recall", 0.0)
            
            speedup = result['speedup']
            if speedup > 1.0:
                speedup_text = f"Polars {speedup:.2f}x faster"
                speedup_class = "polars"
            elif speedup < 1.0:
                speedup_text = f"pl-fuzzy {1.0/speedup:.2f}x faster"
                speedup_class = "plf"
            else:
                speedup_text = "Equal"
                speedup_class = ""
            
            # Polars row - TF-IDF threshold-based candidate selection
            html += f"""            <tr>
                <td></td>
                <td class="dataset">{dataset_size}</td>
                <td class="number">{comparisons}</td>
                <td class="polars">Polars (custom)</td>
                <td class="candidate-selection">TF-IDF Threshold</td>
                <td class="number">{polars_time:.4f}</td>
                <td class="number">{polars_throughput:,.0f}</td>
                <td class="number">{polars_results:,}</td>
                <td class="number">{polars_precision:.3f}</td>
                <td class="number">{polars_recall:.3f}</td>
                <td class="number speedup {speedup_class}">{speedup_text}</td>
            </tr>
"""
            
            # pl-fuzzy row - ANN + Top-K candidate selection
            plf_precision_str = f"{plf_precision:.3f}" if plf_precision > 0 else "-"
            plf_recall_str = f"{plf_recall:.3f}" if plf_recall > 0 else "-"
            
            html += f"""            <tr>
                <td></td>
                <td></td>
                <td></td>
                <td class="plf">pl-fuzzy (std venv)</td>
                <td class="candidate-selection">ANN + Top-K</td>
                <td class="number">{plf_time:.4f}</td>
                <td class="number">{plf_throughput:,.0f}</td>
                <td class="number">{plf_results:,}</td>
                <td class="number">{plf_precision_str}</td>
                <td class="number">{plf_recall_str}</td>
                <td></td>
            </tr>
"""
    
    html += """        </tbody>
    </table>
</body>
</html>
"""
    return html


def main():
    print("=" * 80)
    print("Combined Benchmark: Custom Polars vs pl-fuzzy-frame-match (separate venv)")
    print("=" * 80)
    print(f"\nPolars version (custom build): {pl.__version__}")
    
    # Load pl-fuzzy results from separate venv
    plf_results_file = "plf_venv_results.json"
    if not os.path.exists(plf_results_file):
        print(f"\nERROR: {plf_results_file} not found!")
        print("Run: source plf_venv/bin/activate && python benchmark_plf_venv.py > plf_venv_results.json")
        sys.exit(1)
    
    with open(plf_results_file, 'r') as f:
        plf_results = json.load(f)
    
    plf_polars_version = plf_results[0].get("polars_version", "unknown")
    plf_version = plf_results[0].get("plf_version", "unknown")
    print(f"pl-fuzzy-frame-match version: {plf_version} (with Polars {plf_polars_version})")
    
    # Create lookup for plf results (including precision/recall)
    plf_lookup = {}
    for r in plf_results:
        key = (r["similarity"], r["left_rows"], r["right_rows"])
        plf_lookup[key] = {
            "plf": r["plf"],
            "precision": r.get("plf_precision", 0.0),
            "recall": r.get("plf_recall", 0.0)
        }
    
    test_configs = [
        (100, 100, "XSmall (10K)"),
        (1000, 1000, "Medium (1M)"),
        (2000, 2000, "Large (4M)"),
        (5000, 5000, "XLarge (25M)"),
        (10000, 10000, "XXLarge (100M)"),
    ]
    
    similarities = ["jaro_winkler", "levenshtein", "damerau_levenshtein"]
    
    thresholds = {
        "jaro_winkler": 0.8,
        "levenshtein": 0.7,
        "damerau_levenshtein": 0.7
    }
    
    all_results = []
    
    print("\nRunning Polars benchmarks...")
    
    for left_rows, right_rows, description in test_configs:
        print(f"\n{'='*80}")
        print(f"Testing {description}: {left_rows:,} × {right_rows:,}")
        print(f"{'='*80}")
        
        for similarity in similarities:
            threshold = thresholds[similarity]
            similarity_name = similarity.replace("_", "-").title()
            
            print(f"\n  {similarity_name} (threshold={threshold})...")
            
            # Generate test data (same seed as plf benchmark)
            left_df, right_df, ground_truth = generate_test_data(
                left_rows, right_rows, 
                avg_string_length=15, 
                match_rate=0.3,
                similarity_threshold=threshold,
                similarity_type=similarity,
                seed=42
            )
            
            # Benchmark Polars
            print("    Benchmarking Polars...")
            polars_result, polars_result_df = benchmark_polars_fuzzy_join(
                left_df, right_df,
                similarity=similarity,
                threshold=threshold,
                iterations=3,
                keep="all"
            )
            
            # Calculate precision/recall for Polars
            polars_precision, polars_recall = calculate_precision_recall(
                polars_result_df, ground_truth, deduplicate=True
            )
            
            # Get pl-fuzzy results from pre-computed file
            key = (similarity, left_rows, right_rows)
            plf_data = plf_lookup.get(key, {
                "plf": {"mean": float('inf'), "throughput": 0, "result_count": 0},
                "precision": 0.0,
                "recall": 0.0
            })
            plf_result = plf_data["plf"]
            plf_precision = plf_data.get("precision", 0.0)
            plf_recall = plf_data.get("recall", 0.0)
            
            # Calculate speedup
            if polars_result["mean"] > 0 and plf_result["mean"] > 0 and plf_result["mean"] != float('inf'):
                speedup = plf_result["mean"] / polars_result["mean"]
            else:
                speedup = 0.0
            
            all_results.append({
                "similarity": similarity,
                "left_rows": left_rows,
                "right_rows": right_rows,
                "description": description,
                "comparisons": left_rows * right_rows,
                "polars": polars_result,
                "plf": plf_result,
                "speedup": speedup,
                "polars_precision": polars_precision,
                "polars_recall": polars_recall,
                "plf_precision": plf_precision,
                "plf_recall": plf_recall,
            })
            
            print(f"    Polars: {polars_result['mean']:.4f}s, "
                  f"Precision: {polars_precision:.3f}, Recall: {polars_recall:.3f}")
            print(f"    pl-fuzzy (venv): {plf_result['mean']:.4f}s, "
                  f"Precision: {plf_precision:.3f}, Recall: {plf_recall:.3f}")
            print(f"    Speedup: {speedup:.2f}x")
    
    # Generate HTML table
    print("\n" + "=" * 80)
    print("Generating outputs...")
    html_content = create_html_table(all_results, plf_polars_version, plf_version)
    
    html_file = "benchmark_comparison_table.html"
    with open(html_file, "w") as f:
        f.write(html_content)
    print(f"✓ HTML table saved to: {html_file}")
    
    json_file = "benchmark_comparison_data.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ JSON data saved to: {json_file}")
    
    # Convert to PNG
    print("\nConverting HTML to PNG image...")
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            with open(html_file, 'r') as f:
                html_content = f.read()
            
            page.set_viewport_size({"width": 1600, "height": 3000})
            page.set_content(html_content)
            page.wait_for_timeout(500)
            
            png_file = "benchmark_comparison_table.png"
            page.screenshot(path=png_file, full_page=True)
            browser.close()
        
        print(f"✓ PNG image saved to: {png_file}")
    except Exception as e:
        print(f"⚠️  Could not convert to PNG: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    for similarity in similarities:
        metric_results = [r for r in all_results if r["similarity"] == similarity]
        if not metric_results:
            continue
        
        similarity_name = similarity.replace("_", "-").title()
        print(f"\n{similarity_name}:")
        
        valid_speedups = [r["speedup"] for r in metric_results if r["speedup"] > 0]
        if valid_speedups:
            avg_speedup = statistics.mean(valid_speedups)
            print(f"  Average speedup: {avg_speedup:.2f}x")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

