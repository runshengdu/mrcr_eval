import pandas as pd
from pathlib import Path
import os
import re

START_TOKEN = 4000
MAX_TOKEN = 128000*0.9

bins = [
    (START_TOKEN, 5000),
    (7000, 8000),
    (10000, 15000),
    (30000, 35000),
    (60000, 65000),
    (100000, MAX_TOKEN),
]

def extract_model_name(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"^([^_]+_[^_]+)", base)
    if not m:
        raise ValueError(f"Cannot extract model name from filename: {filename}")
    return m.group(1)

def accuracy_by_bins(df: pd.DataFrame, bins: list[tuple[int, int]]):
    if 'grade' not in df.columns or 'token_count' not in df.columns:
        raise ValueError("CSV must contain 'grade' and 'token_count' columns")
    df = df.copy()
    df['grade'] = pd.to_numeric(df['grade'], errors='coerce')
    df['token_count'] = pd.to_numeric(df['token_count'], errors='coerce')
    df = df[(df['token_count'] >= START_TOKEN) & (df['token_count'] < MAX_TOKEN)]
    out = {}
    for start, end in bins:
        bin_df = df[(df['token_count'] >= start) & (df['token_count'] < end)]
        total = len(bin_df)
        sum_grade = float(bin_df['grade'].sum())
        avg = (sum_grade / total) if total > 0 else 0.0
        out[(start, end)] = {
            'avg': avg,
            'count': int(total)
        }
    return out

def main():
    results_dir = Path('result')
    if not results_dir.exists():
        raise FileNotFoundError("'result' directory not found")

    files = sorted(results_dir.glob('*.csv'))
    if not files:
        raise FileNotFoundError("No CSV files found in 'result' directory")

    # 使用手动创建的 bins
    # bins = build_bins(START_TOKEN, BIN_WIDTH, MAX_TOKEN)
    per_model = {}
    for f in files:
        model = extract_model_name(f.name)
        df = pd.read_csv(f)
        try:
            per_model[model] = accuracy_by_bins(df, bins)
        except ValueError as e:
            print(f"Skip {f}: {e}")

    # Compose compact output: two columns [bin_label, model_accuracies_with_counts]
    rows = []
    for start, end in bins:
        label = f"[{start}, {end})"
        pairs = []
        for model in sorted(per_model.keys()):
            metric = per_model[model].get((start, end), {'avg': 0.0, 'count': 0})
            acc = metric['avg']
            n = metric['count']
            pairs.append(f"{model}={acc:.6f}|n={n}")
        rows.append({
            'bin': label,
            'models': ", ".join(pairs)
        })

    out_df = pd.DataFrame(rows)

    out_dir = Path('sorted_data')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'accuracy.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    # Plotting moved to plot_mrcr_results.py

if __name__ == '__main__':
    main()