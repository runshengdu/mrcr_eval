import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


def parse_bin(label: str):
    m = re.match(r"^\[(\d+),\s*([\d\.]+)\)\s*$", label)
    if not m:
        raise ValueError(f"Bad bin label: {label}")
    start = int(m.group(1))
    end_str = m.group(2)
    end = float(end_str) if "." in end_str else int(end_str)
    return int(start), end


def parse_models_field(field: str):
    pairs = {}
    # Expect format: "model=avg|n=count, model2=avg|n=count"
    for part in str(field).split(", "):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([^=]+)=([0-9\.]+)\|n=(\d+)$", part)
        if not m:
            # Skip malformed segment gracefully
            continue
        model = m.group(1)
        avg = float(m.group(2))
        count = int(m.group(3))
        pairs[model] = {"avg": avg, "count": count}
    return pairs


def main():
    in_path = Path("sorted_data") / "accuracy.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"CSV not found: {in_path}. Please run sorted_data.py first.")

    df = pd.read_csv(in_path)

    # Extract bins and models data
    bins = []
    x_labels = []
    x_positions = []
    per_model = {}

    for _, row in df.iterrows():
        label = str(row["bin"]) if "bin" in row else str(row[0])
        start, end = parse_bin(label)
        bins.append((start, end))
        x_labels.append(f"[{start}, {end})")
        x_positions.append(start)

        models_map = parse_models_field(row["models"]) if "models" in row else {}
        for model, metric in models_map.items():
            per_model.setdefault(model, []).append(metric["avg"])

    x_positions_plot = x_positions
    x_labels_plot = x_labels

    plt.figure(figsize=(15, 10))
    for model in sorted(per_model.keys()):
        ys_full = per_model[model]
        ys_plot = ys_full
        plt.plot(x_positions_plot, ys_plot, marker="o", linewidth=1.5, label=model)

    plt.title("MRCR Average Grade by Token Bins")
    plt.xlabel("Token Bin")
    plt.ylabel("Average grade")
    plt.xticks(x_positions_plot, x_labels_plot, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.yticks([i / 10 for i in range(0, 11)])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_dir = Path("sorted_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "accuracy.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved line chart to {fig_path}")


if __name__ == "__main__":
    main()