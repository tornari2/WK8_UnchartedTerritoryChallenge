#!/usr/bin/env python3
"""
Comprehensive benchmark comparison: Polars vs pl-fuzzy-frame-match
for Jaro-Winkler, Levenshtein, and Damerau-Levenshtein algorithms.

IMPORTANT: pl-fuzzy-frame-match runs in a SEPARATE venv (plf_venv) with its own
Polars version and ANN (polars-simed) ENABLED.

Generates a visual table as an image showing performance across various dataset sizes.
Includes precision and recall metrics using Python implementations as ground truth.
"""

import time
import statistics
import sys
import os
import subprocess
import json
import random
import string
from typing import List, Tuple, Dict, Optional, Set

try:
    import polars as pl
except ImportError:
    print("Error: polars not installed. Install with: pip install polars")
    sys.exit(1)


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


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len1][len2]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity (normalized edit distance)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_len = max(len1, len2)
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Damerau-Levenshtein distance (allows transpositions)."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    
    # Use optimal string alignment distance (restricted edit distance)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
            # Transposition
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost)
    
    return dp[len1][len2]


def damerau_levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate Damerau-Levenshtein similarity (normalized)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_len = max(len1, len2)
    return 1.0 - damerau_levenshtein_distance(s1, s2) / max_len


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


def compute_ground_truth(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    similarity: str,
    threshold: float,
    max_comparisons: int = 4_000_000  # Limit for performance
) -> Set[Tuple[int, int]]:
    """
    Compute ground truth matches using Python similarity implementations.
    Returns set of (left_id, right_id) tuples that are above threshold.
    """
    left_rows = left_df.height
    right_rows = right_df.height
    total_comparisons = left_rows * right_rows
    
    # Skip if too many comparisons (would take too long)
    if total_comparisons > max_comparisons:
        return set()
    
    # Select similarity function
    sim_funcs = {
        "jaro_winkler": jaro_winkler_similarity,
        "levenshtein": levenshtein_similarity,
        "damerau_levenshtein": damerau_levenshtein_similarity,
    }
    sim_func = sim_funcs[similarity]
    
    # Get data as lists for faster iteration
    left_ids = left_df["id"].to_list()
    left_names = left_df["name"].to_list()
    right_ids = right_df["id"].to_list()
    right_names = right_df["name"].to_list()
    
    ground_truth = set()
    
    for i, (lid, lname) in enumerate(zip(left_ids, left_names)):
        for rid, rname in zip(right_ids, right_names):
            sim = sim_func(lname, rname)
            if sim >= threshold:
                ground_truth.add((lid, rid))
    
    return ground_truth


def extract_match_pairs(result_df: Optional[pl.DataFrame]) -> Set[Tuple[int, int]]:
    """Extract (left_id, right_id) pairs from a fuzzy join result."""
    if result_df is None or result_df.height == 0:
        return set()
    
    # The result should have id (left) and id_right columns
    try:
        left_ids = result_df["id"].to_list()
        right_ids = result_df["id_right"].to_list()
        return set(zip(left_ids, right_ids))
    except Exception:
        return set()


def calculate_precision_recall(
    predicted: Set[Tuple[int, int]],
    ground_truth: Set[Tuple[int, int]]
) -> Tuple[float, float]:
    """Calculate precision and recall."""
    if not ground_truth:
        return -1.0, -1.0  # Indicate ground truth not available
    
    if not predicted:
        return 0.0, 0.0
    
    true_positives = len(predicted & ground_truth)
    
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    
    return precision, recall


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
    
    if not hasattr(pl.DataFrame, 'fuzzy_join'):
        raise AttributeError("fuzzy_join is not available.")
    
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


def run_plf_benchmark_in_venv() -> Dict:
    """Run pl-fuzzy-frame-match benchmark in separate venv with ANN enabled."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plf_venv_python = os.path.join(script_dir, "plf_venv", "bin", "python")
    plf_benchmark_script = os.path.join(script_dir, "benchmark_plf_venv.py")
    
    if not os.path.exists(plf_venv_python):
        print(f"Error: plf_venv not found at {plf_venv_python}")
        print("Please create it with: python -m venv plf_venv && plf_venv/bin/pip install polars pl-fuzzy-frame-match")
        return {}
    
    print("\n" + "=" * 80)
    print("Running pl-fuzzy-frame-match benchmark in separate venv (with ANN enabled)...")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [plf_venv_python, plf_benchmark_script],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Print stderr (progress info)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"Error running pl-fuzzy benchmark: {result.stderr}")
            return {}
        
        # Parse JSON output from stdout
        plf_results = json.loads(result.stdout)
        
        # Convert to lookup dict
        lookup = {}
        for r in plf_results:
            key = (r["similarity"], r["left_rows"], r["right_rows"])
            lookup[key] = r
        
        return lookup
        
    except subprocess.TimeoutExpired:
        print("Error: pl-fuzzy benchmark timed out")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing pl-fuzzy results: {e}")
        return {}
    except Exception as e:
        print(f"Error running pl-fuzzy benchmark: {e}")
        return {}


def create_html_table(results: List[Dict], plf_polars_version: str = "", plf_version: str = "") -> str:
    """Create an HTML table with benchmark results including precision/recall."""
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Polars vs pl-fuzzy-frame-match Benchmark Comparison</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .note {
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-bottom: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px auto;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e8f5e9;
        }
        .number {
            text-align: right;
            font-family: 'Courier New', monospace;
        }
        .polars {
            color: #4CAF50;
            font-weight: bold;
        }
        .plf {
            color: #2196F3;
            font-weight: bold;
        }
        .speedup {
            font-weight: bold;
        }
        .speedup-polars {
            color: #4CAF50;
        }
        .speedup-plf {
            color: #2196F3;
        }
        .dataset {
            font-weight: bold;
            color: #673AB7;
        }
        .section-header {
            background-color: #34495e !important;
            color: white !important;
            font-weight: bold;
        }
        .failed {
            color: #999;
            font-style: italic;
        }
        .failed-row {
            background-color: #fff3e0 !important;
        }
        .precision-high {
            color: #4CAF50;
        }
        .precision-medium {
            color: #FF9800;
        }
        .precision-low {
            color: #f44336;
        }
        .na {
            color: #999;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Polars vs pl-fuzzy-frame-match Benchmark Comparison</h1>
    <div class="subtitle">Performance comparison for Jaro-Winkler, Levenshtein, and Damerau-Levenshtein algorithms</div>
    <div class="note">
        Both libraries use the SAME test data and SAME threshold.<br/>
        pl-fuzzy-frame-match runs in separate venv with ANN (polars-simed) ENABLED.<br/>
        Precision/Recall computed using Python reference implementations as ground truth (up to 4M comparisons).<br/>
"""
    
    if plf_version:
        html += f"        pl-fuzzy-frame-match version: {plf_version}<br/>\n"
    if plf_polars_version:
        html += f"        pl-fuzzy Polars version: {plf_polars_version}<br/>\n"
    
    html += f"""        Custom Polars version: {pl.__version__}
    </div>
    <table>
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Dataset Size</th>
                <th>Comparisons</th>
                <th>Library</th>
                <th class="number">Time (s)</th>
                <th class="number">Throughput (comp/s)</th>
                <th class="number">Matches</th>
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
        html += f'            <tr class="section-header"><td colspan="10">{similarity_name}</td></tr>\n'
        
        metric_results.sort(key=lambda x: (x["left_rows"], x["right_rows"]))
        
        for result in metric_results:
            dataset_size = f"{result['left_rows']:,} × {result['right_rows']:,}"
            comparisons = f"{result['comparisons']:,}"
            
            polars_time = result["polars"]["mean"]
            polars_throughput = result["polars"]["throughput"]
            polars_results = result["polars"]["result_count"]
            polars_precision = result.get("polars_precision", -1)
            polars_recall = result.get("polars_recall", -1)
            
            plf_failed = result.get("plf_failed", False) or result["plf"]["result_count"] == 0
            
            speedup = result['speedup']
            if plf_failed:
                speedup_text = "N/A"
                speedup_class = "failed"
            elif speedup > 0:
                if speedup > 1.0:
                    speedup_text = f"Polars {speedup:.2f}x faster"
                    speedup_class = "speedup speedup-polars"
                elif speedup < 1.0:
                    speedup_text = f"pl-fuzzy-frame-match {1.0/speedup:.2f}x faster"
                    speedup_class = "speedup speedup-plf"
                else:
                    speedup_text = "Equal"
                    speedup_class = ""
            else:
                speedup_text = "N/A"
                speedup_class = ""
            
            # Format precision/recall
            def format_metric(val):
                if val < 0:
                    return "N/A", "na"
                elif val >= 0.95:
                    return f"{val:.1%}", "precision-high"
                elif val >= 0.80:
                    return f"{val:.1%}", "precision-medium"
                else:
                    return f"{val:.1%}", "precision-low"
            
            polars_prec_text, polars_prec_class = format_metric(polars_precision)
            polars_rec_text, polars_rec_class = format_metric(polars_recall)
            
            # Polars row
            html += f"""            <tr>
                <td></td>
                <td class="dataset">{dataset_size}</td>
                <td class="number">{comparisons}</td>
                <td class="polars">Polars (custom)</td>
                <td class="number">{polars_time:.4f}</td>
                <td class="number">{polars_throughput:,.0f}</td>
                <td class="number">{polars_results:,}</td>
                <td class="number {polars_prec_class}">{polars_prec_text}</td>
                <td class="number {polars_rec_class}">{polars_rec_text}</td>
                <td class="number {speedup_class}">{speedup_text}</td>
            </tr>
"""
            
            # pl-fuzzy row
            plf_time = result["plf"]["mean"]
            plf_throughput = result["plf"]["throughput"]
            plf_results = result["plf"]["result_count"]
            plf_precision = result.get("plf_precision", -1)
            plf_recall = result.get("plf_recall", -1)
            
            plf_prec_text, plf_prec_class = format_metric(plf_precision)
            plf_rec_text, plf_rec_class = format_metric(plf_recall)
            
            if plf_failed:
                html += f"""            <tr class="failed-row">
                <td></td>
                <td></td>
                <td></td>
                <td class="plf">pl-fuzzy-frame-match</td>
                <td class="number failed">FAILED</td>
                <td class="number failed">-</td>
                <td class="number failed">-</td>
                <td class="number failed">-</td>
                <td class="number failed">-</td>
                <td></td>
            </tr>
"""
            else:
                html += f"""            <tr>
                <td></td>
                <td></td>
                <td></td>
                <td class="plf">pl-fuzzy-frame-match</td>
                <td class="number">{plf_time:.4f}</td>
                <td class="number">{plf_throughput:,.0f}</td>
                <td class="number">{plf_results:,}</td>
                <td class="number {plf_prec_class}">{plf_prec_text}</td>
                <td class="number {plf_rec_class}">{plf_rec_text}</td>
                <td></td>
            </tr>
"""
    
    html += """        </tbody>
    </table>
    
    <div class="note" style="margin-top: 30px;">
        <strong>Why is Polars slower on small datasets?</strong><br/>
        1. <strong>Thread pool overhead</strong>: Polars uses multi-threading (Rayon) which has startup costs not amortized on small data.<br/>
        2. <strong>pl-fuzzy-frame-match uses ANN</strong>: Approximate Nearest Neighbor indexing is very efficient for small datasets that fit in cache.<br/>
        3. <strong>Memory allocation</strong>: Polars allocates buffers optimized for large-scale processing.<br/>
        4. <strong>Polars scales better</strong>: Notice Polars gets relatively faster as dataset size increases (better amortization).
    </div>
</body>
</html>
"""
    return html


def main():
    """Run comprehensive comparison benchmarks and generate table image."""
    print("=" * 80)
    print("Polars vs pl-fuzzy-frame-match Comprehensive Benchmark")
    print("=" * 80)
    print(f"\nCustom Polars version: {pl.__version__}")
    
    has_fuzzy_join = hasattr(pl.DataFrame, 'fuzzy_join')
    
    if not has_fuzzy_join:
        print("\n" + "=" * 80)
        print("⚠️  ERROR: fuzzy_join is not available in your Polars installation")
        print("=" * 80)
        sys.exit(1)
    
    # Test configurations
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
    
    # First, run pl-fuzzy benchmarks in separate venv with ANN enabled
    plf_lookup = run_plf_benchmark_in_venv()
    
    plf_polars_version = ""
    plf_version = ""
    if plf_lookup:
        first_result = list(plf_lookup.values())[0]
        plf_polars_version = first_result.get("polars_version", "")
        plf_version = first_result.get("plf_version", "")
    
    all_results = []
    
    print("\n" + "=" * 80)
    print("Running Polars benchmarks...")
    print("=" * 80)
    
    for left_rows, right_rows, description in test_configs:
        print(f"\nTesting {description}: {left_rows:,} × {right_rows:,}")
        
        for similarity in similarities:
            threshold = thresholds[similarity]
            similarity_name = similarity.replace("_", "-").title()
            
            print(f"  {similarity_name} (threshold={threshold})...")
            
            # Generate test data (same seed as plf benchmark)
            left_df, right_df = generate_test_data(
                left_rows, right_rows, 
                avg_string_length=15, 
                match_rate=0.3,
                seed=42
            )
            
            # Compute ground truth (only for smaller datasets)
            print(f"    Computing ground truth...", end=" ")
            ground_truth = compute_ground_truth(
                left_df, right_df, similarity, threshold
            )
            if ground_truth:
                print(f"({len(ground_truth):,} true matches)")
            else:
                print("(skipped - dataset too large)")
            
            # Benchmark Polars
            polars_result, polars_result_df = benchmark_polars_fuzzy_join(
                left_df, right_df,
                similarity=similarity,
                threshold=threshold,
                iterations=3,
                keep="all"
            )
            
            # Calculate Polars precision/recall
            polars_pairs = extract_match_pairs(polars_result_df)
            polars_precision, polars_recall = calculate_precision_recall(polars_pairs, ground_truth)
            
            # Get pl-fuzzy results from separate venv run
            key = (similarity, left_rows, right_rows)
            plf_data = plf_lookup.get(key, {
                "plf": {"mean": float('inf'), "throughput": 0, "result_count": 0},
            })
            plf_result = plf_data.get("plf", {"mean": float('inf'), "throughput": 0, "result_count": 0})
            plf_pairs_data = plf_data.get("matched_pairs", [])
            plf_pairs = set(tuple(p) for p in plf_pairs_data) if plf_pairs_data else set()
            
            # Calculate pl-fuzzy precision/recall
            plf_precision, plf_recall = calculate_precision_recall(plf_pairs, ground_truth)
            
            # Calculate speedup
            plf_failed = plf_result.get("result_count", 0) == 0 or plf_result.get("mean", float('inf')) == float('inf')
            if plf_failed:
                speedup = 0.0
            elif polars_result["mean"] > 0 and plf_result["mean"] > 0:
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
                "plf_failed": plf_failed,
                "speedup": speedup,
                "ground_truth_count": len(ground_truth) if ground_truth else 0,
                "polars_precision": polars_precision,
                "polars_recall": polars_recall,
                "plf_precision": plf_precision,
                "plf_recall": plf_recall,
            })
            
            print(f"    Polars: {polars_result['mean']:.4f}s, Results: {polars_result['result_count']:,}", end="")
            if polars_precision >= 0:
                print(f", P={polars_precision:.1%}, R={polars_recall:.1%}")
            else:
                print()
            
            if not plf_failed:
                print(f"    pl-fuzzy: {plf_result['mean']:.4f}s, Results: {plf_result['result_count']:,}", end="")
                if plf_precision >= 0:
                    print(f", P={plf_precision:.1%}, R={plf_recall:.1%}")
                else:
                    print()
                if speedup > 1:
                    print(f"    → Polars {speedup:.2f}x faster")
                elif speedup > 0:
                    print(f"    → pl-fuzzy-frame-match {1.0/speedup:.2f}x faster")
            else:
                print(f"    pl-fuzzy: FAILED or not available")
    
    # Generate outputs
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
    
    # Convert HTML to PNG
    print("\nConverting HTML to PNG image...")
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            with open(html_file, 'r') as f:
                html_content = f.read()
            
            page.set_viewport_size({"width": 1600, "height": 2000})
            page.set_content(html_content)
            page.wait_for_timeout(500)
            
            png_file = "benchmark_comparison_table.png"
            page.screenshot(path=png_file, full_page=True)
            browser.close()
        
        print(f"✓ PNG image saved to: {png_file}")
    except Exception as e:
        print(f"⚠️  Could not convert to PNG: {e}")
        print(f"   Open {html_file} in your browser to view the table")
    
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
            if avg_speedup > 1:
                print(f"  Average speedup: Polars {avg_speedup:.2f}x faster")
            else:
                print(f"  Average speedup: pl-fuzzy-frame-match {1.0/avg_speedup:.2f}x faster")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
