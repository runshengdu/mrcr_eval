import argparse
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
from datetime import datetime

time = datetime.now().strftime("%Y%m%d%H%M%S")
needle = "8needle"

thresholds = [8000, 16000, 32000, 64000, 128000, 256000, 512000, 900000]

def parse_args():
    parser = argparse.ArgumentParser(description="Sort MRCR result data by token thresholds.")
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=True,
        help="Model/file keywords to include",
    )
    return parser.parse_args()

def extract_model_name(filename: str) -> str:
    base = Path(filename).name
    m = re.match(r"^(.+)(?=_[^_]+_[^_]+$)", base)

    if not m:
        raise ValueError(f"Cannot extract model name from filename: {filename}")
    return m.group(1)

def format_metric(model: str, metric: dict) -> str:
    acc = metric['avg']
    n = metric['count']
    if acc is None or n is None:
        return f"{model}=|n="
    return f"{model}={acc:.6f}|n={n}"

def accuracy_by_threshold(df: pd.DataFrame, thresholds: list[int]):
    if 'grade' not in df.columns or 'token_count' not in df.columns:
        raise ValueError("CSV must contain 'grade' and 'token_count' columns")
    df = df.copy()
    df['grade'] = pd.to_numeric(df['grade'], errors='coerce')
    df['token_count'] = pd.to_numeric(df['token_count'], errors='coerce')
    max_token_count = df['token_count'].max()
    # 累积阈值统计：只有数据实际达到阈值 t 时，才计算 token_count < t 的平均分
    out = {}
    for t in thresholds:
        if pd.isna(max_token_count) or max_token_count < t:
            out[t] = {
                'avg': None,
                'count': None
            }
            continue

        thresh_df = df[df['token_count'] < t]
        total = len(thresh_df)
        out[t] = {
            'avg': (float(thresh_df['grade'].sum()) / total) if total > 0 else 0.0,
            'count': int(total)
        }
    return out

def main():
    args = parse_args()
    results_dir = Path(f'results/{needle}')
    if not results_dir.exists():
        raise FileNotFoundError("'result' directory not found")

    all_files = sorted(results_dir.glob('*.csv'))
    files = [f for f in all_files if any(kw in f.name for kw in args.keywords)]
    if not files:
        raise FileNotFoundError("没找到对应的文件")

    # Use manually defined bins
    per_model = {}
    for f in files:
        model = extract_model_name(f.name)
        df = pd.read_csv(f)
        try:
            per_model[model] = accuracy_by_threshold(df, thresholds)
        except ValueError as e:
            print(f"Skip {f}: {e}")

    models = sorted(per_model)

    # Compose compact output: 累积阈值标签，例如 <5000、<8000、…、<MAX_TOKEN
    rows = []
    for t in thresholds:
        rows.append({
            'bin': f"<{t}",
            'models': ", ".join(
                format_metric(model, per_model[model].get(t, {'avg': None, 'count': None}))
                for model in models
            )
        })

    out_dir = Path(f'sorted_data/{needle}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'test_result_{time}.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # Plot line chart directly from per_model and thresholds（累积 <t）
    plt.figure(figsize=(10, 7))
    for model in models:
        ys_plot = [per_model[model].get(t, {'avg': None})['avg'] for t in thresholds]
        plt.plot(thresholds, ys_plot, marker="o", linewidth=1.5, label=model)

    plt.title(f"OpenAI MRCR {needle}")
    plt.xlabel("Max token")
    plt.ylabel("Average grade")
    plt.xticks(thresholds, [str(t) for t in thresholds], rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.yticks([i / 20 for i in range(0, 21)])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    fig_path = out_dir / f'test_result_{time}.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved line chart to {fig_path}")

if __name__ == '__main__':
    main()