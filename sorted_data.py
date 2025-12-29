import pandas as pd
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

time=datetime.now().strftime("%Y%m%d%H%M%S")
MAX_TOKEN = int(128000)

thresholds = [5000,10000,20000,40000,60000,80000,100000,MAX_TOKEN]

def extract_model_name(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"^(.+)(?=_[^_]+_[^_]+$)", base)

    if not m:
        raise ValueError(f"Cannot extract model name from filename: {filename}")
    return m.group(1)

def accuracy_by_threshold(df: pd.DataFrame, thresholds: list[int]):
    if 'grade' not in df.columns or 'token_count' not in df.columns:
        raise ValueError("CSV must contain 'grade' and 'token_count' columns")
    df = df.copy()
    df['grade'] = pd.to_numeric(df['grade'], errors='coerce')
    df['token_count'] = pd.to_numeric(df['token_count'], errors='coerce')
    # 累积阈值统计：对每个阈值 t 计算 token_count < t 的平均分
    out = {}
    for t in thresholds:
        thresh_df = df[df['token_count'] < t]
        total = len(thresh_df)
        sum_grade = float(thresh_df['grade'].sum())
        avg = (sum_grade / total) if total > 0 else 0.0
        out[t] = {
            'avg': avg,
            'count': int(total)
        }
    return out

def main():
    results_dir = Path('result')
    if not results_dir.exists():
        raise FileNotFoundError("'result' directory not found")

    all_files = sorted(results_dir.glob('*.csv'))
    keywords = ["gpt-5.2","gemini-3","deepseek-3.2-thinking","kimi-k2","minimax-m2.1","glm","sonnet-4.5"]
    files = [f for f in all_files if any(kw in f.name for kw in keywords)]
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

    # Compose compact output: 累积阈值标签，例如 <5000、<8000、…、<MAX_TOKEN
    rows = []
    for t in thresholds:
        label = f"<{t}"
        pairs = []
        for model in sorted(per_model.keys()):
            metric = per_model[model].get(t, {'avg': 0.0, 'count': 0})
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
    out_path = out_dir / f'test_result_{time}.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # Plot line chart directly from per_model and thresholds（累积 <t）
    x_positions_plot = [t for t in thresholds]
    x_labels_plot = [f"{t}" for t in thresholds]

    plt.figure(figsize=(10, 7))
    for model in sorted(per_model.keys()):
        ys_plot = [per_model[model].get(t, {'avg': 0.0})['avg'] for t in thresholds]
        plt.plot(x_positions_plot, ys_plot, marker="o", linewidth=1.5, label=model)

    plt.title("OpenAI MRCR 2 needle")
    plt.xlabel("Max token")
    plt.ylabel("Average grade")
    plt.xticks(x_positions_plot, x_labels_plot, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.yticks([i / 20 for i in range(0, 21)])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_dir = Path("sorted_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f'test_result_{time}.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved line chart to {fig_path}")

if __name__ == '__main__':
    main()