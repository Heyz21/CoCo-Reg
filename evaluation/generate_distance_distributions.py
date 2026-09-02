
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import directed_hausdorff

from models.DefTransNet import DefTransNet, chamfer_distance_without_batch, sLBP_GF
from models import DefTransNet as benchmark_mod
from models.deformation import SHIFT_SCALE

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPECTED_TOTAL = 908
EXPECTED_VALIDATION = 182
EXPECTED_TEST = 726
SPLIT_SEED = 42

COLOR_INITIAL = "#9ecae1"
COLOR_DEFTRANSNET = "#08519c"
COLOR_GPC = "#006d2c"


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_patchwise_module() -> object:
    path = PROJECT_ROOT / "training" / "train_gpcreg.py"
    if not path.exists():
        raise FileNotFoundError(f"Final GPC-Reg script not found: {path}")
    return import_module_from_path("gpcreg_model", path)


def configure_device(
    device_str: str | None = None,
    disable_cudnn: bool = False,
    patchwise_mod=None,
) -> torch.device:
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    if device_str:
        dev = torch.device(device_str)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    if dev.type == "cuda":
        try:
            torch.cuda.set_device(dev)
            torch.zeros(1, device=dev)
            torch.cuda.synchronize(dev)
            print(f"CUDA ready: {torch.cuda.get_device_name(dev)}")
        except RuntimeError as exc:
            print(f"CUDA init failed ({exc}); falling back to CPU.")
            dev = torch.device("cpu")
            torch.backends.cudnn.enabled = False

    benchmark_mod.device = dev
    if patchwise_mod is not None and hasattr(patchwise_mod, "device"):
        patchwise_mod.device = dev
    return dev


def set_sample_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shuffle_target(target: torch.Tensor, seed: int) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    ind = np.arange(target.shape[1])
    rng.shuffle(ind)
    return target[:, ind, :]


def load_seed42_test_manifest(path: Path) -> pd.DataFrame:
    """Load and strictly validate the formal 182/726 split."""
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    manifest = pd.read_csv(path)
    required = {
        "Dataset Index",
        "Name",
        "Split",
        "Sample Seed",
        "Target Shuffle Seed",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing: {sorted(missing)}")
    if len(manifest) != EXPECTED_TOTAL:
        raise ValueError(
            f"Manifest must contain {EXPECTED_TOTAL} objects, found {len(manifest)}."
        )

    split = manifest["Split"].astype(str).str.strip().str.lower()
    n_validation = int((split == "validation").sum())
    n_test = int((split == "test").sum())
    if n_validation != EXPECTED_VALIDATION or n_test != EXPECTED_TEST:
        raise ValueError(
            "Wrong split counts: "
            f"validation={n_validation}, test={n_test}; expected "
            f"{EXPECTED_VALIDATION} and {EXPECTED_TEST}."
        )

    dataset_index = manifest["Dataset Index"].to_numpy(dtype=np.int64)
    sample_seed = manifest["Sample Seed"].to_numpy(dtype=np.int64)
    shuffle_seed = manifest["Target Shuffle Seed"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_seed, SPLIT_SEED + dataset_index):
        raise ValueError("Manifest does not follow the fixed seed-42 sample rule.")
    if not np.array_equal(shuffle_seed, sample_seed + 1):
        raise ValueError("Target Shuffle Seed must equal Sample Seed + 1.")

    test_manifest = manifest.loc[split == "test"].copy()
    test_manifest = test_manifest.sort_values("Dataset Index").reset_index(drop=True)
    if test_manifest["Name"].astype(str).duplicated().any():
        raise ValueError("Held-out test object names must be unique.")
    print(
        f"Validated seed-42 split: {n_validation} validation objects excluded; "
        f"{n_test} held-out test objects retained."
    )
    return test_manifest


@torch.no_grad()
def load_sample_pair(dataset, idx: int, seed: int) -> dict:
    set_sample_seed(seed)
    return dataset[idx]


@torch.no_grad()
def compute_initial_metrics(source: torch.Tensor, target: torch.Tensor) -> dict:
    err_init = torch.linalg.norm(target.squeeze() - source.squeeze(), dim=1)
    src_cpu = source.squeeze().cpu()
    tgt_cpu = target.squeeze().cpu()
    dhd, _, _ = directed_hausdorff(src_cpu, tgt_cpu)
    return {
        "Initial Mean": err_init.mean().item(),
        "Initial Std": err_init.std().item(),
        "Initial Max": err_init.max().item(),
        "Initial Chamfer": chamfer_distance_without_batch(source, target).item(),
        "Initial DHD": float(dhd),
    }


@torch.no_grad()
def compute_registered_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    disp_pred: torch.Tensor,
) -> dict:
    reg = source + disp_pred
    err_reg = torch.linalg.norm(reg.squeeze() - target.squeeze(), dim=1)
    reg_cpu = reg.squeeze().cpu()
    tgt_cpu = target.squeeze().cpu()
    dhd_reg, _, _ = directed_hausdorff(reg_cpu, tgt_cpu)
    return {
        "Registered Mean": err_reg.mean().item(),
        "Registered Std": err_reg.std().item(),
        "Registered Max": err_reg.max().item(),
        "Registered Chamfer": chamfer_distance_without_batch(reg, target).item(),
        "Registered DHD": float(dhd_reg),
    }


def default_benchmark_ckpt() -> Path:
    return PROJECT_ROOT / "DeftransNet.pth"


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _count_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def load_benchmark_net(
    ckpt: str | None,
    dev: torch.device,
) -> DefTransNet:
    """Load the Transformer-based DefTransNet same-backbone baseline."""
    checkpoint = Path(os.path.expanduser(ckpt)) if ckpt else default_benchmark_ckpt()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"DefTransNet checkpoint not found: {checkpoint}\n"
            f"Default: {default_benchmark_ckpt()}"
        )
    net = DefTransNet(show=0)
    state = _unwrap_state_dict(torch.load(checkpoint, map_location=dev))
    net.load_state_dict(state, strict=True)
    net.to(dev).eval()
    has_tf = hasattr(net, "transform")
    if not has_tf:
        raise RuntimeError(
            "The loaded baseline has no Transformer and is not DefTransNet."
        )
    print(f"Loaded DefTransNet baseline: {checkpoint}")
    print(f"  class={type(net).__name__}  params={_count_params(net)}  has_Transformer={has_tf}")
    return net


def load_patchwise_net(
    epoch: int,
    save_dir: str,
    run_name: str,
    token_dim: int,
    num_geo_blocks: int,
    refine_alpha: float,
    ckpt: str | None,
    dev: torch.device,
    patchwise_mod,
    use_ar: bool = True,
    ar_layers: int = 1,
) -> torch.nn.Module:
    if ckpt:
        checkpoint = Path(os.path.expanduser(ckpt))
    else:
        checkpoint = Path(save_dir) / run_name / "Saves" / f"epoch_{epoch}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"GPC-Reg checkpoint not found: {checkpoint}")
    state = _unwrap_state_dict(torch.load(checkpoint, map_location=dev))
    ckpt_has_ar = any(k.startswith("matcher.ar_encoder.") for k in state.keys())
    if ckpt_has_ar:
        raise RuntimeError(
            "The supplied checkpoint contains AR weights. "
            "Use the final non-AR GPC-Reg checkpoint."
        )

    net = patchwise_mod.GeoPatchDefTransNet(
        token_dim=token_dim,
        num_geo_blocks=num_geo_blocks,
        show=0,
        refine_alpha=refine_alpha,
    )

    missing, unexpected = net.load_state_dict(state, strict=False)
    loaded_feat = any(k.startswith("feature_net.") for k in state.keys())
    if not loaded_feat:
        raise RuntimeError(
            f"GPC-Reg checkpoint has no feature_net.* keys.\n"
            f"File: {checkpoint}\n"
            f"First keys: {list(state.keys())[:8]}"
        )
    if missing or unexpected:
        raise RuntimeError(
            "GPC-Reg checkpoint does not exactly match the model.\n"
            f"Missing ({len(missing)}): {missing[:8]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:8]}"
        )
    net.to(dev).eval()
    has_tf = hasattr(net.feature_net, "transform")
    print(f"Loaded GPC-Reg: {checkpoint}")
    print(
        f"  class={type(net).__name__}  params={_count_params(net)}  "
        f"has_Transformer={has_tf}"
    )
    return net


@torch.no_grad()
def predict_benchmark(
    net: DefTransNet,
    source: torch.Tensor,
    target_shuf: torch.Tensor,
) -> torch.Tensor:
    return sLBP_GF(source, target_shuf, net)


@torch.no_grad()
def predict_patchwise(
    net,
    source: torch.Tensor,
    target_shuf: torch.Tensor,
    num_patches: int,
    patch_size: int,
    patchwise_mod,
) -> torch.Tensor:
    disp_gt = torch.zeros_like(source)
    out = net.patch_forward(source, target_shuf, disp_gt, num_patches, patch_size)
    src_ref, tgt_ref = net.refine_point_features(out)
    return patchwise_mod.inference(source, target_shuf, src_ref, tgt_ref, f=1)


def build_dataset(data_root: Path, def_level: float, sample_filter: str | None, patchwise_mod):
    ds = patchwise_mod.PointCloudData_ModelNet(
        data_root,
        folder="test",
        transform=patchwise_mod.train_transforms,
        typ=patchwise_mod.BENCHMARK_TRAIN_TYPES,
        rotation=patchwise_mod.rotation,
        def_levels=[def_level],
    )
    if sample_filter:
        ds.files = [f for f in ds.files if sample_filter in f["name"]]
        if not ds.files:
            raise FileNotFoundError(f"No test sample matches: {sample_filter}")
    return ds


def print_config(def_levels, patchwise_script: str):
    print("=== GPC-Reg vs DefTransNet distribution data ===")
    print("  GPC-Reg:      final non-AR model")
    print("  DefTransNet:  identical point-level backbone without patch branch")
    print(f"  GPC-Reg script:   {patchwise_script}")
    print("  baseline module:  DefTransNet.py (class DefTransNet)")
    print(f"  Deformation:      Gaussian TPS  (SHIFT_SCALE={SHIFT_SCALE})")
    print(f"  Levels:           {def_levels}")
    print("  Evaluation set:   726 held-out test objects only")
    print("  Validation set:   182 objects excluded")
    print(f"  Rotation:         0 (disabled)")
    print(f"  Pairing:          same source/target + Initial per (level, sample)")
    print(f"  Shuffle:          shared shuffle_seed = sample_seed + 1")
    print()


def evaluate_paired(
    benchmark_net: DefTransNet,
    patchwise_net,
    data_root: Path,
    dev: torch.device,
    def_levels: list[float],
    num_patches: int,
    patch_size: int,
    patchwise_mod,
    test_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_rows = []
    patchwise_rows = []

    for def_lvl in def_levels:
        dataset = build_dataset(data_root, def_lvl, None, patchwise_mod)
        if len(dataset) != EXPECTED_TOTAL:
            raise ValueError(
                f"ModelNet10 test dataset must contain {EXPECTED_TOTAL} objects, "
                f"found {len(dataset)}."
            )

        # Verify that manifest indices still identify the same dataset objects.
        test_records = test_manifest.to_dict(orient="records")
        for row in test_records:
            idx = int(row["Dataset Index"])
            expected_name = str(row["Name"])
            actual_name = str(dataset.files[idx]["name"])
            if actual_name != expected_name:
                raise ValueError(
                    "Dataset ordering differs from split_manifest.csv: "
                    f"index {idx} is {actual_name!r}, expected {expected_name!r}."
                )

        print(
            f"\n--- Deformation level {def_lvl:.1f}: "
            f"{EXPECTED_TEST} held-out test objects ---"
        )

        for test_position, row in enumerate(test_records, 1):
            idx = int(row["Dataset Index"])
            sample_seed = int(row["Sample Seed"])
            target_shuffle_seed = int(row["Target Shuffle Seed"])
            data = load_sample_pair(dataset, idx, sample_seed)

            source = data["source_pointcloud"].to(dev).float().unsqueeze(0)
            target = data["target_pointcloud"].to(dev).float().unsqueeze(0)
            setting = data["setting"]
            name = data["name"]
            typ = data["type"]

            init = compute_initial_metrics(source, target)
            if str(name) != str(row["Name"]):
                raise ValueError(
                    f"Loaded object {name!r}, expected manifest object {row['Name']!r}."
                )
            target_shuf = shuffle_target(target, target_shuffle_seed)

            disp_bench = predict_benchmark(benchmark_net, source, target_shuf)
            disp_patch = predict_patchwise(
                patchwise_net, source, target_shuf, num_patches, patch_size, patchwise_mod
            )

            disp_max_diff = (disp_bench - disp_patch).abs().max().item()
            disp_mean_diff = (disp_bench - disp_patch).abs().mean().item()
            same_disp = bool(torch.allclose(disp_bench, disp_patch, rtol=0.0, atol=1e-8))

            reg_bench = compute_registered_metrics(source, target, disp_bench)
            reg_patch = compute_registered_metrics(source, target, disp_patch)

            meta = {
                "Type": typ,
                "Deformation Level": def_lvl,
                "Setting": setting,
                "Name": name,
                "Seed": sample_seed,
                "Target Shuffle Seed": target_shuffle_seed,
                "Disp Max Diff": disp_max_diff,
                "Disp Mean Diff": disp_mean_diff,
            }
            benchmark_rows.append({**meta, **init, **reg_bench})
            patchwise_rows.append({**meta, **init, **reg_patch})

            print(
                f"  [{test_position}/{EXPECTED_TEST}] {name}  d={def_lvl:.1f}  "
                f"initial={init['Initial Mean']:.6f}  "
                f"DefTransNet={reg_bench['Registered Mean']:.6f}  "
                f"GPC-Reg={reg_patch['Registered Mean']:.6f}  "
                f"|Δdisp|_max={disp_max_diff:.6e}"
            )
            if same_disp:
                print(
                    "  WARNING: disp_bench and disp_patch are tensor-identical "
                    "(possible wrong ckpt / shared prediction path)."
                )

    expected_rows = EXPECTED_TEST * len(def_levels)
    if len(benchmark_rows) != expected_rows or len(patchwise_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows per model, obtained "
            f"{len(benchmark_rows)} and {len(patchwise_rows)}."
        )
    return pd.DataFrame(benchmark_rows), pd.DataFrame(patchwise_rows)


def summarize_by_level(df_benchmark: pd.DataFrame, df_patchwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for def_lvl in sorted(df_benchmark["Deformation Level"].unique()):
        b = df_benchmark[df_benchmark["Deformation Level"] == def_lvl]
        p = df_patchwise[df_patchwise["Deformation Level"] == def_lvl]
        rows.append(
            {
                "Deformation Level": def_lvl,
                "Count": len(b),
                "Initial Mean": b["Initial Mean"].mean(),
                "DefTransNet Registered Mean": b["Registered Mean"].mean(),
                "DefTransNet Registered Chamfer": b["Registered Chamfer"].mean(),
                "DefTransNet Registered DHD": b["Registered DHD"].mean(),
                "GPC-Reg Registered Mean": p["Registered Mean"].mean(),
                "GPC-Reg Registered Chamfer": p["Registered Chamfer"].mean(),
                "GPC-Reg Registered DHD": p["Registered DHD"].mean(),
                "GPC-Reg - DefTransNet (Mean)": (
                    p["Registered Mean"].mean()
                    - b["Registered Mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, df_benchmark: pd.DataFrame, df_patchwise: pd.DataFrame):
    print("\n=== Final results by deformation level ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nOverall Initial Mean (shared):      {df_benchmark['Initial Mean'].mean():.5f}")
    print(f"Overall DefTransNet Registered Mean: {df_benchmark['Registered Mean'].mean():.5f}")
    print(f"Overall GPC-Reg Registered Mean:     {df_patchwise['Registered Mean'].mean():.5f}")
    delta = df_patchwise["Registered Mean"].mean() - df_benchmark["Registered Mean"].mean()
    print(f"Overall GPC-Reg - DefTransNet:       {delta:.5f}  (negative = GPC-Reg better)")


def _shared_bins(arrays: list[np.ndarray], number_of_bins: int) -> np.ndarray:
    values = np.concatenate(arrays)
    lower = max(0.0, float(values.min()))
    upper = float(values.max())
    if upper <= lower:
        upper = lower + 1e-6
    padding = 0.02 * (upper - lower)
    return np.linspace(max(0.0, lower - padding), upper + padding, number_of_bins + 1)


def plot_identity_distribution(
    df_benchmark: pd.DataFrame,
    df_patchwise: pd.DataFrame,
    def_level: float,
    output_dir: Path,
    number_of_bins: int,
) -> Path:
    """Plot one fixed-level identity-specific distribution figure."""
    b = df_benchmark[
        np.isclose(df_benchmark["Deformation Level"].astype(float), def_level)
    ].copy()
    p = df_patchwise[
        np.isclose(df_patchwise["Deformation Level"].astype(float), def_level)
    ].copy()

    keys = ["Name", "Seed", "Target Shuffle Seed"]
    b = b.sort_values(keys).reset_index(drop=True)
    p = p.sort_values(keys).reset_index(drop=True)
    if len(b) != EXPECTED_TEST or len(p) != EXPECTED_TEST:
        raise ValueError(
            f"Level {def_level:.1f} must contain {EXPECTED_TEST} objects per model; "
            f"found {len(b)} and {len(p)}."
        )
    if not b[keys].equals(p[keys]):
        raise ValueError("DefTransNet and GPC-Reg do not use identical test pairs/seeds.")
    if not np.allclose(
        b["Initial Mean"].to_numpy(float),
        p["Initial Mean"].to_numpy(float),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Initial identity-specific distances differ between models.")

    initial = b["Initial Mean"].to_numpy(float)
    registered_def = b["Registered Mean"].to_numpy(float)
    registered_gpc = p["Registered Mean"].to_numpy(float)
    bins = _shared_bins([initial, registered_def, registered_gpc], number_of_bins)
    maximum_count = max(
        int(np.histogram(values, bins=bins)[0].max())
        for values in (initial, registered_def, registered_gpc)
    )

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True
    )
    panels = (
        (axes[0], "DefTransNet", registered_def, COLOR_DEFTRANSNET),
        (axes[1], "GPC-Reg", registered_gpc, COLOR_GPC),
    )
    for axis, title, registered, registered_color in panels:
        axis.hist(
            initial,
            bins=bins,
            color=COLOR_INITIAL,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            label="Before registration",
        )
        axis.hist(
            registered,
            bins=bins,
            color=registered_color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.45,
            label="After registration",
        )
        axis.axvline(initial.mean(), color=COLOR_INITIAL, linestyle="--", linewidth=1.6)
        axis.axvline(registered.mean(), color=registered_color, linestyle="--", linewidth=1.6)
        axis.set_title(title, pad=9)
        axis.set_xlabel("Identity-specific mean distance", labelpad=8)
        axis.set_xlim(0.0, bins[-1])
        axis.set_ylim(0.0, maximum_count * 1.10)
        axis.legend(frameon=False, loc="upper right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)

    axes[0].set_ylabel("Number of test objects", labelpad=8)
    figure.suptitle(
        "Identity-Specific Mean Distance Distribution\n"
        f"(deformation = {def_level:.1f}, N = {EXPECTED_TEST})",
        fontsize=18,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.17, top=0.78, wspace=0.15
    )

    level_tag = int(round(def_level * 10))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"eval_modelnet10_mean_distance_distribution_level_{level_tag:02d}.pdf"
    )
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
    print(f"Saved figure: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the 726 held-out objects and generate the paired CSVs "
            "and thesis distribution figures."
        )
    )
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument(
        "--data_root", type=str, default=str(PROJECT_ROOT / "ModelNet10")
    )
    parser.add_argument("--save_dir", type=str, default="Registration")
    parser.add_argument("--run_name", type=str, default="gpcreg/seed_42")
    parser.add_argument(
        "--deftransnet_ckpt",
        type=str,
        default=None,
        help=(
            "DefTransNet checkpoint; default is DeftransNet.pth "
            "in the project root"
        ),
    )
    parser.add_argument(
        "--gpc_ckpt",
        type=str,
        default="Registration/gpcreg/seed_42/Saves/epoch_20.pth",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.3, 0.6],
        help="Fixed deformation levels. Formal figures require: 0.3 0.6",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="DefTransNet_GPC_Distribution",
        help="Output folder under Registration/Evaluation/",
    )
    parser.add_argument(
        "--split_manifest",
        type=str,
        default="Registration/Evaluation/GPC_Reg_Final/split_manifest.csv",
        help="Seed-42 manifest containing 182 validation and 726 test objects",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures",
        help="Directory for the two output PDF files",
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable_cudnn", action="store_true")
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--num_geo_blocks", type=int, default=1)
    parser.add_argument("--num_patches", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--refine_alpha", type=float, default=0.10)
    args = parser.parse_args()

    patchwise_mod = load_patchwise_module()
    dev = configure_device(args.device, disable_cudnn=args.disable_cudnn, patchwise_mod=patchwise_mod)

    data_root = Path(os.path.expanduser(args.data_root))
    save_dir = Path(os.path.expanduser(args.save_dir))
    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")

    def_levels = [float(level) for level in args.levels]
    if len(def_levels) != 2 or not all(
        any(np.isclose(level, required) for level in def_levels)
        for required in (0.3, 0.6)
    ):
        raise ValueError(
            "The formal run must contain exactly --levels 0.3 0.6."
        )
    allowed_levels = set(float(level) for level in patchwise_mod.BENCHMARK_DEF_LEVELS)
    invalid_levels = [level for level in def_levels if level not in allowed_levels]
    if invalid_levels:
        raise ValueError(
            f"Unsupported deformation levels: {invalid_levels}; "
            f"allowed: {sorted(allowed_levels)}"
        )

    patchwise_script = getattr(
        patchwise_mod, "__file__", "training/train_gpcreg.py"
    )
    print_config(def_levels, patchwise_script)
    print(f"Device: {dev}")
    print(
        f"DefTransNet ckpt: "
        f"{Path(os.path.expanduser(args.deftransnet_ckpt)) if args.deftransnet_ckpt else default_benchmark_ckpt()}"
    )
    print(f"GPC-Reg ckpt: {Path(os.path.expanduser(args.gpc_ckpt))}")

    test_manifest = load_seed42_test_manifest(
        Path(os.path.expanduser(args.split_manifest))
    )

    benchmark_net = load_benchmark_net(args.deftransnet_ckpt, dev)
    patchwise_net = load_patchwise_net(
        args.epoch,
        str(save_dir),
        args.run_name,
        args.token_dim,
        args.num_geo_blocks,
        args.refine_alpha,
        args.gpc_ckpt,
        dev,
        patchwise_mod,
        use_ar=False,
        ar_layers=1,
    )

    df_benchmark, df_patchwise = evaluate_paired(
        benchmark_net,
        patchwise_net,
        data_root,
        dev,
        def_levels=def_levels,
        num_patches=args.num_patches,
        patch_size=args.patch_size,
        patchwise_mod=patchwise_mod,
        test_manifest=test_manifest,
    )

    out_dir = save_dir / "Evaluation" / args.save
    os.makedirs(out_dir, exist_ok=True)

    bench_csv = out_dir / "dataframe_deftransnet.csv"
    patch_csv = out_dir / "dataframe_gpcreg.csv"
    summary_csv = out_dir / "summary_by_level.csv"

    df_benchmark.to_csv(bench_csv, index=False)
    df_patchwise.to_csv(patch_csv, index=False)

    summary = summarize_by_level(df_benchmark, df_patchwise)
    summary.to_csv(summary_csv, index=False)

    print(f"\nSaved DefTransNet CSV: {bench_csv}")
    print(f"Saved GPC-Reg CSV:     {patch_csv}")
    print(f"Saved level summary:   {summary_csv}")

    print_summary(summary, df_benchmark, df_patchwise)

    figures_dir = Path(os.path.expanduser(args.figures_dir))
    for def_level in (0.3, 0.6):
        plot_identity_distribution(
            df_benchmark,
            df_patchwise,
            def_level=def_level,
            output_dir=figures_dir,
            number_of_bins=args.bins,
        )

    print("\nCompleted: 726 held-out test objects at levels 0.3 and 0.6.")
    print("No validation object was evaluated or plotted.")


if __name__ == "__main__":
    main()
