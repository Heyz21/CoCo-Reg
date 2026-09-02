"""
Default output:
  Registration/Evaluation/GPC_Reg_Final/
  eval_modelnet10_failure_cases.pdf
  eval_modelnet10_failure_cases.png
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import torch

from evaluation import evaluate_gpcreg as evaluation


BLACKBODY_CMAP = LinearSegmentedColormap.from_list(
    "plotly_blackbody",
    [
        (0.00, "#000000"),
        (0.25, "#e60000"),
        (0.55, "#e6d200"),
        (1.00, "#80b7ff"),
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DefTransNet/GPC-Reg heatmaps for two failure cases."
    )
    parser.add_argument(
        "--pair_cache",
        default=(
            "Registration/Evaluation/GPC_Reg_Final/"
            "fixed_test_pairs.pt"
        ),
        help="Fixed test-pair cache written by evaluate_gpcreg.py.",
    )
    parser.add_argument(
        "--deftransnet_ckpt",
        default="DeftransNet.pth",
    )
    parser.add_argument(
        "--gpc_ckpt",
        default="Registration/gpcreg/seed_42/Saves/epoch_20.pth",
    )
    parser.add_argument(
        "--gpc_script",
        default="training/train_gpcreg.py",
    )
    parser.add_argument(
        "--mild_name",
        default="night_stand_0264.off",
    )
    parser.add_argument(
        "--severe_name",
        default="table_0409.off",
    )
    parser.add_argument(
        "--output",
        default=(
            "Registration/Evaluation/GPC_Reg_Final/"
            "eval_modelnet10_failure_cases.pdf"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_patches", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--num_geo_blocks", type=int, default=1)
    parser.add_argument("--refine_alpha", type=float, default=0.10)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument(
        "--error_percentile",
        type=float,
        default=95.0,
        help=(
            "Joint percentile used as the heatmap maximum within each row. "
            "Values above it are clipped only for visualization."
        ),
    )
    return parser.parse_args()


def resolved(path_text: str) -> Path:
    return Path(os.path.expanduser(path_text)).resolve()


def load_pair_cache(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fixed pair cache not found:\n  {path}\n"
            "Use the fixed_test_pairs.pt produced by the formal evaluation."
        )
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(path, map_location="cpu")

    if not isinstance(cache, dict) or "pairs" not in cache:
        raise ValueError(f"Unexpected fixed-pair cache format: {path}")
    return cache["pairs"]


def find_pair(pairs: list[dict], name: str) -> dict:
    matches = [pair for pair in pairs if str(pair.get("name")) == name]
    if len(matches) != 1:
        available = [str(pair.get("name")) for pair in pairs]
        similar = [item for item in available if name.split("_")[0] in item]
        raise ValueError(
            f"Expected exactly one pair named {name!r}, found {len(matches)}.\n"
            f"Similar names: {similar[:15]}"
        )
    return matches[0]


def build_models(
    args: argparse.Namespace,
    gpc_module,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    deftransnet = evaluation.benchmark_mod.DefTransNet(show=0)
    gpc_reg = gpc_module.GeoPatchDefTransNet(
        token_dim=args.token_dim,
        num_geo_blocks=args.num_geo_blocks,
        show=0,
        refine_alpha=args.refine_alpha,
    )

    evaluation.load_state(
        deftransnet,
        resolved(args.deftransnet_ckpt),
        device,
        "DefTransNet",
        strict=True,
    )
    evaluation.load_state(
        gpc_reg,
        resolved(args.gpc_ckpt),
        device,
        "GPC-Reg",
        strict=True,
    )
    return deftransnet.to(device).eval(), gpc_reg.to(device).eval()


@torch.inference_mode()
def register_pair(
    pair: dict,
    deftransnet: torch.nn.Module,
    gpc_reg: torch.nn.Module,
    args: argparse.Namespace,
    gpc_module,
    device: torch.device,
) -> dict[str, np.ndarray | float | str]:
    source = pair["source"].to(device)
    target = pair["target"].to(device)
    target_shuffled = pair["target_shuffled"].to(device)

    def_displacement, _, _ = evaluation.predict_benchmark(
        deftransnet,
        source,
        target_shuffled,
        device,
    )
    gpc_displacement, _, _ = evaluation.predict_gpc(
        gpc_reg,
        source,
        target_shuffled,
        args,
        gpc_module,
        device,
    )

    def_registered = source + def_displacement
    gpc_registered = source + gpc_displacement

    # The unshuffled target preserves the ground-truth point identity.
    def_errors = torch.linalg.norm(
        def_registered - target,
        dim=-1,
    ).squeeze(0)
    gpc_errors = torch.linalg.norm(
        gpc_registered - target,
        dim=-1,
    ).squeeze(0)

    return {
        "name": str(pair["name"]),
        "level": float(pair["deformation_level"]),
        "source": source.squeeze(0).cpu().numpy(),
        "target": target.squeeze(0).cpu().numpy(),
        "def_registered": def_registered.squeeze(0).cpu().numpy(),
        "gpc_registered": gpc_registered.squeeze(0).cpu().numpy(),
        "def_errors": def_errors.cpu().numpy(),
        "gpc_errors": gpc_errors.cpu().numpy(),
        "def_mean": float(def_errors.mean().item()),
        "gpc_mean": float(gpc_errors.mean().item()),
    }


def row_limits(result: dict) -> tuple[np.ndarray, float]:
    points = np.concatenate(
        [
            result["source"],
            result["target"],
            result["def_registered"],
            result["gpc_registered"],
        ],
        axis=0,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    centre = 0.5 * (minimum + maximum)
    radius = 0.51 * float(np.max(maximum - minimum))
    radius = max(radius, 1e-4)
    return centre, radius


def style_axis(
    axis,
    centre: np.ndarray,
    radius: float,
    elev: float,
    azim: float,
) -> None:
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elev, azim=azim)
    axis.set_axis_off()


def scatter_cloud(axis, points: np.ndarray, color: str) -> None:
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=5.5,
        c=color,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
    )


def scatter_heatmap(
    axis,
    registered: np.ndarray,
    target: np.ndarray,
    errors: np.ndarray,
    norm: Normalize,
) -> None:
    # A faint target silhouette makes spatial misalignment easier to see.
    axis.scatter(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        s=2.5,
        c="#b8b8b8",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        registered[:, 0],
        registered[:, 1],
        registered[:, 2],
        s=6.5,
        c=errors,
        cmap=BLACKBODY_CMAP,
        norm=norm,
        alpha=0.96,
        linewidths=0,
        rasterized=True,
    )


def make_figure(
    mild: dict,
    severe: dict,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    results = [mild, severe]

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(13.2, 6.5),
        subplot_kw={"projection": "3d"},
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.94,
        bottom=0.03,
        top=0.975,
        wspace=-0.18,
        hspace=0.06,
    )

    for row, result in enumerate(results):
        centre, radius = row_limits(result)
        combined_errors = np.concatenate(
            [result["def_errors"], result["gpc_errors"]]
        )
        vmax = float(
            np.percentile(combined_errors, args.error_percentile)
        )
        vmax = max(vmax, 1e-8)
        norm = Normalize(vmin=0.0, vmax=vmax, clip=True)

        scatter_cloud(axes[row, 0], result["source"], "#2878b5")
        scatter_cloud(axes[row, 1], result["target"], "#e07a24")
        scatter_heatmap(
            axes[row, 2],
            result["def_registered"],
            result["target"],
            result["def_errors"],
            norm,
        )
        scatter_heatmap(
            axes[row, 3],
            result["gpc_registered"],
            result["target"],
            result["gpc_errors"],
            norm,
        )

        axes[row, 0].set_title(
            "Source PC\n"
            f"{result['name']}, level {result['level']:.1f}",
            fontsize=13,
            pad=-2,
        )
        axes[row, 1].set_title("Target PC", fontsize=13, pad=-2)
        axes[row, 2].set_title(
            f"DefTransNet\nerror = {result['def_mean']:.6f}",
            fontsize=13,
            pad=-2,
        )
        axes[row, 3].set_title(
            f"GPC-Reg\nerror = {result['gpc_mean']:.6f}",
            fontsize=13,
            pad=-2,
        )

        for column in range(4):
            style_axis(
                axes[row, column],
                centre,
                radius,
                args.elev,
                args.azim,
            )

        scalar_map = ScalarMappable(norm=norm, cmap=BLACKBODY_CMAP)
        scalar_map.set_array([])
        colorbar = figure.colorbar(
            scalar_map,
            ax=axes[row, :].tolist(),
            fraction=0.013,
            pad=0.002,
            shrink=0.82,
        )
        colorbar.set_label(
            "",
            fontsize=10,
        )
        colorbar.ax.tick_params(labelsize=8)

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

    print("\nSaved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


def main() -> None:
    args = parse_args()

    gpc_script = resolved(args.gpc_script)
    if not gpc_script.is_file():
        raise FileNotFoundError(f"GPC-Reg script not found: {gpc_script}")

    gpc_module = evaluation.import_module_from_path(
        "gpcreg_failure_visualization",
        gpc_script,
    )
    device = evaluation.configure_device(args.device, gpc_module)
    evaluation.set_all_seeds(42)

    pairs = load_pair_cache(resolved(args.pair_cache))
    mild_pair = find_pair(pairs, args.mild_name)
    severe_pair = find_pair(pairs, args.severe_name)
    deftransnet, gpc_reg = build_models(args, gpc_module, device)

    print(f"Device: {device}")
    print(f"Mild failure:   {args.mild_name}")
    print(f"Severe failure: {args.severe_name}")

    mild = register_pair(
        mild_pair,
        deftransnet,
        gpc_reg,
        args,
        gpc_module,
        device,
    )
    severe = register_pair(
        severe_pair,
        deftransnet,
        gpc_reg,
        args,
        gpc_module,
        device,
    )

    print("\nRecomputed identity-specific means:")
    for label, result in (("Mild", mild), ("Severe", severe)):
        delta = result["gpc_mean"] - result["def_mean"]
        print(
            f"  {label}: DefTransNet={result['def_mean']:.6f}, "
            f"GPC-Reg={result['gpc_mean']:.6f}, "
            f"delta={delta:+.6f}"
        )

    make_figure(
        mild,
        severe,
        resolved(args.output),
        args,
    )


if __name__ == "__main__":
    main()
