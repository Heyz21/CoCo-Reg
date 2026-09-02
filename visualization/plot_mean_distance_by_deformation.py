"""
Read evaluation CSV and write:

  Registration/Evaluation/GPC_Reg_Final/
  eval_modelnet10_mean_distance_by_deformation.pdf

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = (
    "Registration/Evaluation/GPC_Reg_Final/per_sample_metrics.csv"
)
DEFAULT_OUTPUT = (
    "Registration/Evaluation/GPC_Reg_Final/"
    "eval_modelnet10_mean_distance_by_deformation.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean distance at deformation levels 0.1--0.9."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no_robust",
        action="store_true",
        help="Omit the supplementary Robust-DefReg curve.",
    )
    return parser.parse_args()


def resolved(path_text: str) -> Path:
    return Path(os.path.expanduser(path_text)).resolve()


def main() -> None:
    args = parse_args()
    csv_path = resolved(args.csv)
    output_path = resolved(args.output)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required = {
        "Deformation Level",
        "Initial Mean",
        "Robust Mean",
        "DefTransNet Mean",
        "GPC Mean",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )

    grouped = (
        data.groupby("Deformation Level", as_index=True)
        .mean(numeric_only=True)
        .sort_index()
    )
    levels = grouped.index.to_numpy()

    figure, axis = plt.subplots(figsize=(8.2, 4.9))

    # The input error is a reference curve, not a registration method.
    # A dashed line makes this distinction immediately visible.
    axis.plot(
        levels,
        grouped["Initial Mean"].to_numpy(),
        color="#2f2f2f",
        linestyle=(0, (5, 3)),
        marker="o",
        markersize=6.2,
        linewidth=2.1,
        label="Before registration",
        zorder=2,
    )

    if not args.no_robust:
        axis.plot(
            levels,
            grouped["Robust Mean"].to_numpy(),
            color="#777777",
            linestyle="-",
            marker="s",
            markersize=5.8,
            linewidth=2.0,
            label="Robust-DefReg",
            zorder=3,
        )

    axis.plot(
        levels,
        grouped["DefTransNet Mean"].to_numpy(),
        color="#2f6db0",
        linestyle="-",
        marker="D",
        markersize=5.6,
        linewidth=2.0,
        label="DefTransNet",
        zorder=4,
    )
    axis.plot(
        levels,
        grouped["GPC Mean"].to_numpy(),
        color="#c4511b",
        linestyle="-",
        marker="^",
        markersize=6.6,
        linewidth=2.2,
        label="GPC-Reg",
        zorder=5,
    )

    axis.set_title("Mean Distance vs Deformation", fontsize=15, pad=10)
    axis.set_xlabel("Deformation level", fontsize=13)
    axis.set_ylabel("Identity-specific mean distance", fontsize=13)
    axis.set_xticks(levels)
    axis.tick_params(axis="both", labelsize=11)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    axis.legend(fontsize=10.5, frameon=False, loc="upper left")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_base = (
        output_path.with_suffix("")
        if output_path.suffix
        else output_path
    )
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(figure)

    print("Saved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
