import pandas as pd
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt

START_TOKEN = 4000
MAX_TOKEN = 128000*0.9

bins = [
    (START_TOKEN, 5000),
    (8000,9000),
    (10000, 12000),
    (20000, 22000),
    (30000, 35000),
    (40000, 45000),
    (50000, 55000),
    (60000, 65000),
    (70000, 75000),
    (80000, 85000),
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

    all_files = sorted(results_dir.glob('*.csv'))
    # 仅选择文件名包含 ds-3.1 或 ds-3.2 的 CSV 文件
    files = [f for f in all_files if ("ds-3.1" in f.name or "ds-3.2" in f.name or "minimax" in f.name)]
    if not files:
        raise FileNotFoundError("未在 'result' 目录中找到包含 ds-3.1 或 ds-3.2 的 CSV 文件")

    # Use manually defined bins
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

    # Plot line chart directly from per_model and bins
    x_positions_plot = [start for (start, end) in bins]
    x_labels_plot = [f"[{start}, {end})" for (start, end) in bins]

    plt.figure(figsize=(20, 10))
    for model in sorted(per_model.keys()):
        ys_plot = [per_model[model].get((start, end), {'avg': 0.0})['avg'] for (start, end) in bins]
        plt.plot(x_positions_plot, ys_plot, marker="o", linewidth=1.5, label=model)

    plt.title("MRCR Average Grade by Token Bins")
    plt.xlabel("Token Bin")
    plt.ylabel("Average grade")
    plt.xticks(x_positions_plot, x_labels_plot, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.yticks([i / 20 for i in range(0, 21)])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_dir = Path("sorted_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "accuracy.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved line chart to {fig_path}")

if __name__ == '__main__':
    main()