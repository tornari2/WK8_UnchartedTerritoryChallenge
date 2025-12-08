#!/usr/bin/env python3
"""
Comprehensive benchmark: Polars String Similarity Functions vs RapidFuzz

Benchmarks all 5 similarity metrics:
1. Levenshtein (string similarity)
2. Damerau-Levenshtein (string similarity)
3. Jaro-Winkler (string similarity)
4. Hamming (string similarity)
5. Cosine (vector similarity - Array<Float64>) - LARGE 100k dataset, len=30

Compares Polars native implementations against RapidFuzz.
Generates benchmark_results.png with comprehensive visualization.
"""

import time
import statistics
import random
import string
import json
from typing import List, Tuple, Dict
import polars as pl

# Try to import RapidFuzz
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein, DamerauLevenshtein, Hamming, JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    print("⚠️  RapidFuzz not installed. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "rapidfuzz"], check=True)
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein, DamerauLevenshtein, Hamming, JaroWinkler
    HAS_RAPIDFUZZ = True


def generate_string_pairs(
    num_pairs: int,
    avg_string_length: int = 15,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Generate pairs of strings for similarity benchmarks."""
    random.seed(seed)
    
    def random_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def create_similar_string(original: str, similarity: float) -> str:
        if similarity >= 0.99:
            return original
        chars = list(original)
        if len(chars) == 0:
            return original
        num_changes = max(1, int(len(chars) * (1.0 - similarity)))
        modified = chars.copy()
        for _ in range(num_changes):
            if len(modified) > 1:
                idx = random.randint(0, len(modified) - 1)
                if random.random() < 0.7:
                    modified[idx] = random.choice(string.ascii_letters)
                else:
                    modified.pop(idx)
        return ''.join(modified)
    
    strings_a = []
    strings_b = []
    
    for i in range(num_pairs):
        s1 = random_string(avg_string_length)
        # Mix of similar and different strings
        if i % 3 == 0:
            s2 = create_similar_string(s1, 0.85 + random.random() * 0.15)
        elif i % 3 == 1:
            s2 = create_similar_string(s1, 0.6 + random.random() * 0.2)
        else:
            s2 = random_string(avg_string_length)
        strings_a.append(s1)
        strings_b.append(s2)
    
    return strings_a, strings_b


def generate_vector_pairs(
    num_pairs: int,
    vector_dim: int = 30,
    seed: int = 42
) -> Tuple[List[List[float]], List[List[float]]]:
    """Generate pairs of vectors for cosine similarity benchmarks."""
    random.seed(seed)
    
    def random_unit_vector(dim: int) -> List[float]:
        vec = [random.gauss(0, 1) for _ in range(dim)]
        magnitude = sum(x**2 for x in vec) ** 0.5
        if magnitude > 0:
            vec = [x / magnitude for x in vec]
        return vec
    
    vectors_a = []
    vectors_b = []
    
    for i in range(num_pairs):
        v1 = random_unit_vector(vector_dim)
        if i % 3 == 0:
            # Similar vector (add small noise)
            noise = [random.gauss(0, 0.1) for _ in range(vector_dim)]
            v2 = [a + n for a, n in zip(v1, noise)]
            mag = sum(x**2 for x in v2) ** 0.5
            v2 = [x / mag for x in v2] if mag > 0 else v2
        else:
            v2 = random_unit_vector(vector_dim)
        vectors_a.append(v1)
        vectors_b.append(v2)
    
    return vectors_a, vectors_b


def benchmark_polars_string_similarity(
    strings_a: List[str],
    strings_b: List[str],
    metric: str,
    iterations: int = 5
) -> Dict:
    """Benchmark Polars string similarity functions."""
    df = pl.DataFrame({"a": strings_a, "b": strings_b})
    times = []
    
    # Warmup run
    if metric == "levenshtein":
        _ = df.select(pl.col("a").str.levenshtein_sim(pl.col("b"))).to_series().to_list()
    elif metric == "damerau_levenshtein":
        _ = df.select(pl.col("a").str.damerau_levenshtein_sim(pl.col("b"))).to_series().to_list()
    elif metric == "jaro_winkler":
        _ = df.select(pl.col("a").str.jaro_winkler_sim(pl.col("b"))).to_series().to_list()
    elif metric == "hamming":
        _ = df.select(pl.col("a").str.hamming_sim(pl.col("b"))).to_series().to_list()
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        if metric == "levenshtein":
            result = df.select(pl.col("a").str.levenshtein_sim(pl.col("b")))
        elif metric == "damerau_levenshtein":
            result = df.select(pl.col("a").str.damerau_levenshtein_sim(pl.col("b")))
        elif metric == "jaro_winkler":
            result = df.select(pl.col("a").str.jaro_winkler_sim(pl.col("b")))
        elif metric == "hamming":
            result = df.select(pl.col("a").str.hamming_sim(pl.col("b")))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Force evaluation
        _ = result.to_series().to_list()
        
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = statistics.mean(times)
    num_pairs = len(strings_a)
    
    return {
        "mean": mean_time,
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "num_pairs": num_pairs,
        "throughput": num_pairs / mean_time if mean_time > 0 else 0
    }


def benchmark_rapidfuzz_similarity(
    strings_a: List[str],
    strings_b: List[str],
    metric: str,
    iterations: int = 5
) -> Dict:
    """Benchmark RapidFuzz string similarity functions."""
    times = []
    
    # Warmup run
    if metric == "levenshtein":
        _ = [Levenshtein.normalized_similarity(a, b) for a, b in zip(strings_a[:100], strings_b[:100])]
    elif metric == "damerau_levenshtein":
        _ = [DamerauLevenshtein.normalized_similarity(a, b) for a, b in zip(strings_a[:100], strings_b[:100])]
    elif metric == "jaro_winkler":
        _ = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(strings_a[:100], strings_b[:100])]
    elif metric == "hamming":
        for a, b in zip(strings_a[:100], strings_b[:100]):
            max_len = max(len(a), len(b))
            _ = Hamming.normalized_similarity(a.ljust(max_len), b.ljust(max_len))
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        if metric == "levenshtein":
            results = [Levenshtein.normalized_similarity(a, b) for a, b in zip(strings_a, strings_b)]
        elif metric == "damerau_levenshtein":
            results = [DamerauLevenshtein.normalized_similarity(a, b) for a, b in zip(strings_a, strings_b)]
        elif metric == "jaro_winkler":
            results = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(strings_a, strings_b)]
        elif metric == "hamming":
            # Hamming requires equal length - pad shorter strings
            results = []
            for a, b in zip(strings_a, strings_b):
                max_len = max(len(a), len(b))
                a_padded = a.ljust(max_len)
                b_padded = b.ljust(max_len)
                results.append(Hamming.normalized_similarity(a_padded, b_padded))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = statistics.mean(times)
    num_pairs = len(strings_a)
    
    return {
        "mean": mean_time,
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "num_pairs": num_pairs,
        "throughput": num_pairs / mean_time if mean_time > 0 else 0
    }


def benchmark_polars_cosine_similarity(
    vectors_a: List[List[float]],
    vectors_b: List[List[float]],
    vector_dim: int,
    iterations: int = 5
) -> Dict:
    """Benchmark Polars cosine similarity for vectors."""
    # Pre-build DataFrame outside timing
    df = pl.DataFrame({"a": vectors_a, "b": vectors_b}).with_columns([
        pl.col("a").cast(pl.Array(pl.Float64, vector_dim)),
        pl.col("b").cast(pl.Array(pl.Float64, vector_dim))
    ])
    
    times = []
    
    # Warmup run
    _ = df.select(pl.col("a").arr.cosine_similarity(pl.col("b"))).to_series().to_list()
    
    for _ in range(iterations):
        start = time.perf_counter()
        result = df.select(pl.col("a").arr.cosine_similarity(pl.col("b")))
        # Force evaluation
        _ = result.to_series().to_list()
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = statistics.mean(times)
    num_pairs = len(vectors_a)
    
    return {
        "mean": mean_time,
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "num_pairs": num_pairs,
        "throughput": num_pairs / mean_time if mean_time > 0 else 0
    }


def benchmark_numpy_cosine_similarity(
    vectors_a: List[List[float]],
    vectors_b: List[List[float]],
    iterations: int = 5
) -> Dict:
    """Benchmark NumPy cosine similarity for comparison."""
    import numpy as np
    
    arr_a = np.array(vectors_a)
    arr_b = np.array(vectors_b)
    
    times = []
    
    # Warmup run
    dot_products = np.sum(arr_a * arr_b, axis=1)
    norms_a = np.linalg.norm(arr_a, axis=1)
    norms_b = np.linalg.norm(arr_b, axis=1)
    _ = (dot_products / (norms_a * norms_b)).tolist()
    
    for _ in range(iterations):
        start = time.perf_counter()
        # Compute cosine similarity: dot(a, b) / (||a|| * ||b||)
        dot_products = np.sum(arr_a * arr_b, axis=1)
        norms_a = np.linalg.norm(arr_a, axis=1)
        norms_b = np.linalg.norm(arr_b, axis=1)
        similarities = dot_products / (norms_a * norms_b)
        # Force evaluation
        _ = similarities.tolist()
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = statistics.mean(times)
    num_pairs = len(vectors_a)
    
    return {
        "mean": mean_time,
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "num_pairs": num_pairs,
        "throughput": num_pairs / mean_time if mean_time > 0 else 0
    }


def create_html_report(results: List[Dict]) -> str:
    """Create an HTML report with benchmark results."""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Polars vs RapidFuzz Similarity Benchmark</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        h1 {{
            color: #00d9ff;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        }}
        .subtitle {{
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .version-info {{
            text-align: center;
            color: #888;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .metric-section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid;
        }}
        .metric-levenshtein {{ border-color: #4ecdc4; color: #4ecdc4; }}
        .metric-damerau_levenshtein {{ border-color: #ff6b6b; color: #ff6b6b; }}
        .metric-jaro_winkler {{ border-color: #feca57; color: #feca57; }}
        .metric-hamming {{ border-color: #a29bfe; color: #a29bfe; }}
        .metric-cosine {{ border-color: #00d9ff; color: #00d9ff; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th {{
            background: rgba(0, 217, 255, 0.2);
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #00d9ff;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        .number {{
            text-align: right;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }}
        .polars {{
            color: #00ff88;
            font-weight: bold;
        }}
        .rapidfuzz {{
            color: #ff6b6b;
        }}
        .numpy {{
            color: #feca57;
        }}
        .speedup {{
            font-weight: bold;
        }}
        .speedup-win {{
            color: #00ff88;
        }}
        .speedup-lose {{
            color: #ff6b6b;
        }}
        .dataset-label {{
            background: rgba(255, 255, 255, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .summary {{
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(0, 255, 136, 0.1));
            border-radius: 12px;
            padding: 20px;
            margin: 30px 0;
            border: 1px solid rgba(0, 217, 255, 0.3);
        }}
        .summary h2 {{
            color: #00d9ff;
            margin-bottom: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-card {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #00ff88;
        }}
        .summary-card .label {{
            color: #aaa;
            font-size: 0.9em;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        .legend-polars {{ background: #00ff88; }}
        .legend-rapidfuzz {{ background: #ff6b6b; }}
        .legend-numpy {{ background: #feca57; }}
    </style>
</head>
<body>
    <h1>🚀 Polars vs RapidFuzz Similarity Benchmark</h1>
    <div class="subtitle">Direct comparison of string similarity functions + Cosine vector similarity</div>
    <div class="version-info">Polars version: {pl.__version__}</div>
    
    <div class="legend">
        <div class="legend-item"><div class="legend-color legend-polars"></div><span>Polars</span></div>
        <div class="legend-item"><div class="legend-color legend-rapidfuzz"></div><span>RapidFuzz</span></div>
        <div class="legend-item"><div class="legend-color legend-numpy"></div><span>NumPy</span></div>
    </div>
"""
    
    # Group results by metric
    metrics = ["levenshtein", "damerau_levenshtein", "jaro_winkler", "hamming", "cosine"]
    metric_names = {
        "levenshtein": "Levenshtein",
        "damerau_levenshtein": "Damerau-Levenshtein",
        "jaro_winkler": "Jaro-Winkler",
        "hamming": "Hamming",
        "cosine": "Cosine (Vector, dim=30)"
    }
    
    # Calculate summary stats
    polars_wins = sum(1 for r in results if r.get("speedup", 0) > 1.0)
    total_tests = len([r for r in results if "speedup" in r])
    avg_speedup = statistics.mean([r.get("speedup", 1.0) for r in results if "speedup" in r]) if total_tests > 0 else 1.0
    max_speedup = max([r.get("speedup", 1.0) for r in results if "speedup" in r], default=1.0)
    
    # Summary section
    html += f"""
    <div class="summary">
        <h2>📊 Overall Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{len(results)}</div>
                <div class="label">Benchmark Tests</div>
            </div>
            <div class="summary-card">
                <div class="value">{polars_wins}/{total_tests}</div>
                <div class="label">Polars Wins</div>
            </div>
            <div class="summary-card">
                <div class="value">{avg_speedup:.2f}x</div>
                <div class="label">Avg Speedup</div>
            </div>
            <div class="summary-card">
                <div class="value">{max_speedup:.2f}x</div>
                <div class="label">Max Speedup</div>
            </div>
        </div>
    </div>
"""
    
    for metric in metrics:
        metric_results = [r for r in results if r.get("metric") == metric]
        if not metric_results:
            continue
        
        html += f"""
    <div class="metric-section">
        <div class="metric-title metric-{metric}">{metric_names.get(metric, metric)}</div>
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th class="number">Pairs</th>
                    <th>Library</th>
                    <th class="number">Time (s)</th>
                    <th class="number">Throughput (pairs/s)</th>
                    <th class="number">Speedup</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for r in sorted(metric_results, key=lambda x: x.get("num_pairs", 0)):
            polars_data = r.get("polars", {})
            competitor_data = r.get("competitor", {})
            competitor_name = r.get("competitor_name", "RapidFuzz")
            speedup = r.get("speedup", 1.0)
            
            if speedup > 1.0:
                speedup_text = f"Polars {speedup:.2f}x faster"
                speedup_class = "speedup speedup-win"
            elif speedup < 1.0:
                speedup_text = f"{competitor_name} {1.0/speedup:.2f}x faster"
                speedup_class = "speedup speedup-lose"
            else:
                speedup_text = "Equal"
                speedup_class = ""
            
            # Polars row
            html += f"""
                <tr>
                    <td><span class="dataset-label">{r.get('description', '')}</span></td>
                    <td class="number">{r.get('num_pairs', 0):,}</td>
                    <td class="polars">Polars</td>
                    <td class="number">{polars_data.get('mean', 0):.6f}</td>
                    <td class="number">{polars_data.get('throughput', 0):,.0f}</td>
                    <td class="number {speedup_class}">{speedup_text}</td>
                </tr>
"""
            
            # Competitor row
            comp_class = "rapidfuzz" if competitor_name == "RapidFuzz" else "numpy"
            html += f"""
                <tr>
                    <td></td>
                    <td></td>
                    <td class="{comp_class}">{competitor_name}</td>
                    <td class="number">{competitor_data.get('mean', 0):.6f}</td>
                    <td class="number">{competitor_data.get('throughput', 0):,.0f}</td>
                    <td></td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
"""
    
    html += """
</body>
</html>
"""
    return html


def main():
    print("=" * 80)
    print("Polars vs RapidFuzz - String Similarity Functions Benchmark")
    print("=" * 80)
    print(f"\nPolars version: {pl.__version__}")
    print(f"RapidFuzz available: {HAS_RAPIDFUZZ}")
    
    all_results = []
    
    # String similarity configurations
    # Using string length=30 to match original benchmark methodology
    string_configs = [
        (1_000, 10, "1K pairs, len=10"),
        (10_000, 20, "10K pairs, len=20"),
        (100_000, 30, "100K pairs, len=30 (LARGE)"),
    ]
    
    string_metrics = ["levenshtein", "damerau_levenshtein", "jaro_winkler", "hamming"]
    
    print("\n" + "=" * 80)
    print("String Similarity Benchmarks (Polars vs RapidFuzz)")
    print("=" * 80)
    
    for num_pairs, string_len, description in string_configs:
        print(f"\n📊 {description}: {num_pairs:,} string pairs")
        
        # Generate test data
        strings_a, strings_b = generate_string_pairs(num_pairs, avg_string_length=string_len, seed=42)
        
        for metric in string_metrics:
            metric_name = metric.replace('_', '-').title()
            print(f"   {metric_name}...", end=" ", flush=True)
            
            try:
                # Benchmark Polars
                polars_result = benchmark_polars_string_similarity(
                    strings_a, strings_b, metric, iterations=5
                )
                
                # Benchmark RapidFuzz
                rapidfuzz_result = benchmark_rapidfuzz_similarity(
                    strings_a, strings_b, metric, iterations=5
                )
                
                # Calculate speedup
                speedup = rapidfuzz_result["mean"] / polars_result["mean"] if polars_result["mean"] > 0 else 0
                
                all_results.append({
                    "metric": metric,
                    "description": description,
                    "num_pairs": num_pairs,
                    "polars": polars_result,
                    "competitor": rapidfuzz_result,
                    "competitor_name": "RapidFuzz",
                    "speedup": speedup
                })
                
                winner = "Polars" if speedup > 1 else "RapidFuzz"
                ratio = speedup if speedup > 1 else 1/speedup
                print(f"✅ {winner} {ratio:.2f}x faster | Polars: {polars_result['throughput']:,.0f}/s, RapidFuzz: {rapidfuzz_result['throughput']:,.0f}/s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Cosine similarity configurations - including LARGE 100k dataset with len=30
    print("\n" + "=" * 80)
    print("Cosine Similarity Benchmarks (Polars vs NumPy)")
    print("=" * 80)
    
    # Cosine similarity - matching original benchmark methodology
    cosine_configs = [
        (1_000, 10, "1K pairs, dim=10"),
        (10_000, 20, "10K pairs, dim=20"),
        (100_000, 30, "100K pairs, dim=30 (LARGE)"),
    ]
    
    for num_pairs, vector_dim, description in cosine_configs:
        print(f"\n📊 {description}: {num_pairs:,} vector pairs")
        print(f"   Cosine (dim={vector_dim})...", end=" ", flush=True)
        
        try:
            vectors_a, vectors_b = generate_vector_pairs(num_pairs, vector_dim=vector_dim, seed=42)
            
            # Benchmark Polars
            polars_result = benchmark_polars_cosine_similarity(
                vectors_a, vectors_b, vector_dim, iterations=5
            )
            
            # Benchmark NumPy
            numpy_result = benchmark_numpy_cosine_similarity(
                vectors_a, vectors_b, iterations=5
            )
            
            # Calculate speedup
            speedup = numpy_result["mean"] / polars_result["mean"] if polars_result["mean"] > 0 else 0
            
            all_results.append({
                "metric": "cosine",
                "description": description,
                "num_pairs": num_pairs,
                "vector_dim": vector_dim,
                "polars": polars_result,
                "competitor": numpy_result,
                "competitor_name": "NumPy",
                "speedup": speedup
            })
            
            winner = "Polars" if speedup > 1 else "NumPy"
            ratio = speedup if speedup > 1 else 1/speedup
            print(f"✅ {winner} {ratio:.2f}x faster | Polars: {polars_result['throughput']:,.0f}/s, NumPy: {numpy_result['throughput']:,.0f}/s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Generate HTML report
    print("\n" + "=" * 80)
    print("Generating outputs...")
    print("=" * 80)
    
    html_content = create_html_report(all_results)
    
    html_file = "benchmark_results.html"
    with open(html_file, "w") as f:
        f.write(html_content)
    print(f"✓ HTML report saved to: {html_file}")
    
    # Save JSON data
    json_file = "benchmark_results.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ JSON data saved to: {json_file}")
    
    # Convert to PNG
    print("\nConverting HTML to PNG...")
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            page.set_viewport_size({"width": 1400, "height": 3000})
            page.set_content(html_content)
            page.wait_for_timeout(500)
            
            png_file = "benchmark_results.png"
            page.screenshot(path=png_file, full_page=True)
            browser.close()
        
        print(f"✓ PNG image saved to: {png_file}")
    except Exception as e:
        print(f"⚠️  Could not convert to PNG: {e}")
        print(f"   Open {html_file} in your browser to view the report")
    
    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    for metric in ["levenshtein", "damerau_levenshtein", "jaro_winkler", "hamming", "cosine"]:
        metric_results = [r for r in all_results if r.get("metric") == metric]
        if not metric_results:
            continue
        
        metric_name = metric.replace("_", "-").title()
        avg_speedup = statistics.mean([r.get("speedup", 1.0) for r in metric_results])
        max_pairs = max([r.get("num_pairs", 0) for r in metric_results])
        
        if avg_speedup > 1:
            winner = "Polars"
            ratio = avg_speedup
        else:
            winner = "RapidFuzz/NumPy"
            ratio = 1/avg_speedup
        
        print(f"\n{metric_name}:")
        print(f"  Tests: {len(metric_results)}")
        print(f"  Max pairs: {max_pairs:,}")
        print(f"  Avg speedup: {winner} {ratio:.2f}x faster")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
