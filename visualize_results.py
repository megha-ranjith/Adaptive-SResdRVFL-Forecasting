from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception as exc:
    raise SystemExit("seaborn is required. Install with: pip install seaborn") from exc


REQUIRED_COLUMNS: List[str] = [
    "Actual",
    "Predicted",
    "Lower_Bound",
    "Upper_Bound",
    "Block1_Attn",
    "Block2_Attn",
    "Block3_Attn",
    "Block4_Attn",
    "Block5_Attn",
]
ATTN_COLUMNS: List[str] = [
    "Block1_Attn",
    "Block2_Attn",
    "Block3_Attn",
    "Block4_Attn",
    "Block5_Attn",
]


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "results.csv is missing required columns: " + ", ".join(missing)
        )

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Actual", "Predicted", "Lower_Bound", "Upper_Bound"])
    if df.empty:
        raise ValueError("No valid rows found after numeric conversion.")

    df = df.reset_index(drop=True)
    return df


def plot_forecast_vs_actual(df: pd.DataFrame, out_dir: Path) -> Path:
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, df["Actual"], label="Actual", linewidth=2.0, color="#1f77b4")
    ax.plot(x, df["Predicted"], label="Predicted", linewidth=2.0, color="#ff7f0e")

    ax.fill_between(
        x,
        df["Lower_Bound"].values,
        df["Upper_Bound"].values,
        color="#ff7f0e",
        alpha=0.20,
        label="95% Confidence Interval",
    )

    ax.set_title("Forecast vs Actual with 95% Confidence Ribbon")
    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    out_path = out_dir / "forecast_vs_actual_ci.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return out_path


def plot_attention_evolution(df: pd.DataFrame, out_dir: Path) -> Path:
    attn = df[ATTN_COLUMNS].fillna(0.0).to_numpy().T  # shape (5, n)

    # Normalize each timestep so area chart sums to 1 even if logs are slightly noisy.
    denom = np.sum(attn, axis=0)
    denom = np.where(denom <= 0.0, 1.0, denom)
    attn = attn / denom

    x = np.arange(df.shape[0], dtype=float)

    # For single-row files, duplicate one point so stackplot has visible area.
    if len(x) == 1:
        x_plot = np.array([0.0, 1.0])
        attn_plot = np.repeat(attn, 2, axis=1)
    else:
        x_plot = x
        attn_plot = attn

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = sns.color_palette("tab10", n_colors=5)
    labels = ["Block 1", "Block 2", "Block 3", "Block 4", "Block 5"]

    ax.stackplot(x_plot, attn_plot, labels=labels, colors=colors, alpha=0.90)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Stacked Attention Evolution")
    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("Attention Weight")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.20)

    out_path = out_dir / "attention_evolution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return out_path


def plot_error_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    errors = (df["Actual"] - df["Predicted"]).astype(float)

    fig, ax = plt.subplots(figsize=(10, 5.2))

    if len(errors) >= 3:
        bins = max(5, min(30, int(np.sqrt(len(errors)) * 2)))
        sns.histplot(errors, bins=bins, kde=True, ax=ax, color="#2ca02c", alpha=0.85)
    elif len(errors) == 2:
        sns.histplot(errors, bins=2, kde=False, ax=ax, color="#2ca02c", alpha=0.85)
    else:
        ax.bar([0], [errors.iloc[0]], width=0.5, color="#2ca02c", alpha=0.85)
        ax.set_xticks([0])
        ax.set_xticklabels(["Single Sample"])

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Prediction Error Distribution (Actual - Predicted)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.20)

    out_path = out_dir / "error_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize results.csv from Adaptive-SResdRVFL")
    parser.add_argument(
        "--input",
        type=str,
        default="results.csv",
        help="Path to results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving",
    )
    return parser


def main() -> None:
    plt.style.use("seaborn-v0_8")
    sns.set_theme(style="whitegrid", context="talk")

    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(input_path)

    saved_paths = [
        plot_forecast_vs_actual(df, output_dir),
        plot_attention_evolution(df, output_dir),
        plot_error_distribution(df, output_dir),
    ]

    print("Generated plots:")
    for p in saved_paths:
        print(f"- {p}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
