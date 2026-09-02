"""
Outputs:

    figures/eval_modelnet10_mean_distance_distribution_level_03.pdf
    figures/eval_modelnet10_mean_distance_distribution_level_06.pdf
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
import numpy as np
import pandas as pd


LEVELS = (0.3, 0.6)
EXPECTED_TOTAL = 908
EXPECTED_VALIDATION = 182
EXPECTED_TEST = 726
SPLIT_SEED = 42

COLOR_BEFORE = "#9ecae1"
COLOR_DEFTRANSNET = "#08519c"
COLOR_GPC = "#006d2c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot identity-specific mean-distance distributions using only "
            "the 726 seed-42 held-out test objects."
        )
    )
    parser.add_argument(
        "--csv_dir",
        default="Registration/Evaluation/DefTransNet_GPC_Distribution",
        help="Directory containing dataframe_deftransnet.csv and dataframe_gpcreg.csv.",
    )
    parser.add_argument(
        "--split_manifest",
        default="Registration/Evaluation/GPC_Reg_Final/split_manifest.csv",
        help="Seed-42 split_manifest.csv containing the 908-object split.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures",
        help="Output directory for the two thesis PDF figures.",
    )
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_seed42_test_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    manifest = pd.read_csv(path)
    require_columns(
        manifest,
        {
            "Dataset Index",
            "Name",
            "Split",
            "Sample Seed",
            "Target Shuffle Seed",
        },
        "split manifest",
    )

    if len(manifest) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} objects in the split manifest, "
            f"found {len(manifest)}."
        )

    split = manifest["Split"].astype(str).str.strip().str.lower()
    number_validation = int((split == "validation").sum())
    number_test = int((split == "test").sum())
    if number_validation != EXPECTED_VALIDATION or number_test != EXPECTED_TEST:
        raise ValueError(
            "Unexpected split counts: "
            f"validation={number_validation}, test={number_test}; "
            f"expected {EXPECTED_VALIDATION} and {EXPECTED_TEST}."
        )

    # The formal manifest was generated with base seed 42:
    # sample_seed = 42 + dataset_index and shuffle_seed = sample_seed + 1.
    expected_sample_seed = (
        SPLIT_SEED + manifest["Dataset Index"].to_numpy(dtype=np.int64)
    )
    actual_sample_seed = manifest["Sample Seed"].to_numpy(dtype=np.int64)
    actual_shuffle_seed = manifest["Target Shuffle Seed"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_sample_seed, expected_sample_seed):
        raise ValueError(
            "The manifest does not match the expected seed-42 sample-seed rule."
        )
    if not np.array_equal(actual_shuffle_seed, actual_sample_seed + 1):
        raise ValueError(
            "The manifest target-shuffle seeds are inconsistent with sample_seed + 1."
        )

    if manifest["Name"].duplicated().any():
        duplicates = manifest.loc[
            manifest["Name"].duplicated(keep=False), "Name"
        ].tolist()
        raise ValueError(
            "Object names are not unique in the split manifest. "
            f"Examples: {duplicates[:5]}"
        )

    test_manifest = manifest.loc[
        split == "test",
        [
            "Dataset Index",
            "Name",
            "Sample Seed",
            "Target Shuffle Seed",
        ],
    ].copy()
    test_manifest = test_manifest.sort_values("Dataset Index").reset_index(drop=True)
    assert len(test_manifest) == EXPECTED_TEST
    return test_manifest


def load_paired_csvs(csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def_path = csv_dir / "dataframe_deftransnet.csv"
    gpc_path = csv_dir / "dataframe_gpcreg.csv"
    if not def_path.is_file() or not gpc_path.is_file():
        raise FileNotFoundError(
            "Required paired CSV files were not found.\n"
            f"Expected:\n  {def_path}\n  {gpc_path}"
        )

    df_def = pd.read_csv(def_path)
    df_gpc = pd.read_csv(gpc_path)
    required = {
        "Deformation Level",
        "Name",
        "Initial Mean",
        "Registered Mean",
    }
    require_columns(df_def, required, "DefTransNet CSV")
    require_columns(df_gpc, required, "GPC-Reg CSV")
    return df_def, df_gpc


def select_heldout_level(
    frame: pd.DataFrame,
    test_manifest: pd.DataFrame,
    deformation_level: float,
    label: str,
) -> pd.DataFrame:
    level_rows = frame.loc[
        np.isclose(
            frame["Deformation Level"].to_numpy(dtype=float),
            deformation_level,
        )
    ].copy()
    if level_rows.empty:
        raise ValueError(
            f"{label}: no rows found for deformation level {deformation_level:.1f}."
        )
    if level_rows["Name"].duplicated().any():
        raise ValueError(
            f"{label}: duplicate object names at deformation "
            f"level {deformation_level:.1f}."
        )

    selected = test_manifest.merge(
        level_rows,
        on="Name",
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", ""),
    )
    if selected["Initial Mean"].isna().any() or selected["Registered Mean"].isna().any():
        missing = selected.loc[
            selected["Initial Mean"].isna() | selected["Registered Mean"].isna(),
            "Name",
        ].tolist()
        raise ValueError(
            f"{label}: {len(missing)} held-out test objects are missing at "
            f"deformation {deformation_level:.1f}. Examples: {missing[:5]}"
        )
    if len(selected) != EXPECTED_TEST:
        raise ValueError(
            f"{label}: expected {EXPECTED_TEST} held-out objects, "
            f"found {len(selected)}."
        )

    # Confirm that no validation name can enter through the merge.
    if set(selected["Name"]) != set(test_manifest["Name"]):
        raise AssertionError(f"{label}: selected object set differs from test manifest.")

    return selected.sort_values("Dataset Index").reset_index(drop=True)


def validate_pairing(
    df_def: pd.DataFrame,
    df_gpc: pd.DataFrame,
    deformation_level: float,
) -> None:
    if len(df_def) != EXPECTED_TEST or len(df_gpc) != EXPECTED_TEST:
        raise ValueError("Both model panels must contain exactly 726 objects.")

    if not df_def[["Dataset Index", "Name"]].equals(
        df_gpc[["Dataset Index", "Name"]]
    ):
        raise ValueError(
            "DefTransNet and GPC-Reg do not contain the same ordered test objects."
        )

    initial_difference = np.max(
        np.abs(
            df_def["Initial Mean"].to_numpy(dtype=float)
            - df_gpc["Initial Mean"].to_numpy(dtype=float)
        )
    )
    if initial_difference > 1e-8:
        raise ValueError(
            "The two models do not use identical source-target pairs at "
            f"deformation {deformation_level:.1f}; maximum Initial Mean "
            f"difference is {initial_difference:.3e}."
        )

    # If the evaluation CSVs contain seeds, require identical deformation and
    # target-permutation seeds for both model outputs.
    for optional_column in ("Seed", "Target Shuffle Seed"):
        if optional_column in df_def.columns and optional_column in df_gpc.columns:
            if not np.array_equal(
                df_def[optional_column].to_numpy(),
                df_gpc[optional_column].to_numpy(),
            ):
                raise ValueError(
                    f"DefTransNet and GPC-Reg differ in {optional_column}."
                )


def make_shared_bins(
    arrays: list[np.ndarray],
    number_of_bins: int,
) -> np.ndarray:
    values = np.concatenate(arrays)
    lower = max(0.0, float(values.min()))
    upper = float(values.max())
    if upper <= lower:
        upper = lower + 1e-6
    padding = max(1e-6, 0.02 * (upper - lower))
    return np.linspace(
        max(0.0, lower - padding),
        upper + padding,
        number_of_bins + 1,
    )


def plot_level(
    df_def: pd.DataFrame,
    df_gpc: pd.DataFrame,
    deformation_level: float,
    bins_count: int,
    output_path: Path,
) -> None:
    initial = df_def["Initial Mean"].to_numpy(dtype=float)
    registered_def = df_def["Registered Mean"].to_numpy(dtype=float)
    registered_gpc = df_gpc["Registered Mean"].to_numpy(dtype=float)

    bins = make_shared_bins(
        [initial, registered_def, registered_gpc],
        bins_count,
    )
    maximum_count = max(
        int(np.histogram(values, bins=bins)[0].max())
        for values in (initial, registered_def, registered_gpc)
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.5),
        sharex=True,
        sharey=True,
    )

    panels = (
        (axes[0], "DefTransNet", registered_def, COLOR_DEFTRANSNET),
        (axes[1], "GPC-Reg", registered_gpc, COLOR_GPC),
    )
    for axis, panel_title, registered, registered_color in panels:
        axis.hist(
            initial,
            bins=bins,
            color=COLOR_BEFORE,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.4,
            label="Before registration",
        )
        axis.hist(
            registered,
            bins=bins,
            color=registered_color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.4,
            label="After registration",
        )
        axis.axvline(
            initial.mean(),
            color="#5b9bc5",
            linestyle="--",
            linewidth=1.2,
        )
        axis.axvline(
            registered.mean(),
            color=registered_color,
            linestyle="--",
            linewidth=1.2,
        )
        axis.set_title(panel_title)
        axis.set_xlabel("Identity-specific mean distance [normalised units]")
        axis.set_xlim(float(bins[0]), float(bins[-1]))
        axis.set_ylim(0.0, max(1.0, maximum_count * 1.10))
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="upper right")

    axes[0].set_ylabel("Number of test objects")
    figure.suptitle(
        "Identity-Specific Mean Distance Distribution\n"
        f"(deformation = {deformation_level:.1f}, N = {EXPECTED_TEST})",
        fontsize=14,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.16,
        top=0.80,
        wspace=0.16,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)
    print(f"Saved: {output_path}")
    print(
        f"  deformation={deformation_level:.1f}, N={len(df_def)}, "
        f"initial mean={initial.mean():.6f}, "
        f"DefTransNet mean={registered_def.mean():.6f}, "
        f"GPC-Reg mean={registered_gpc.mean():.6f}"
    )


def main() -> None:
    args = parse_args()
    csv_dir = Path(os.path.expanduser(args.csv_dir))
    split_manifest_path = Path(os.path.expanduser(args.split_manifest))
    output_dir = Path(os.path.expanduser(args.output_dir))

    test_manifest = load_seed42_test_manifest(split_manifest_path)
    df_def_all, df_gpc_all = load_paired_csvs(csv_dir)

    for deformation_level in LEVELS:
        df_def = select_heldout_level(
            df_def_all,
            test_manifest,
            deformation_level,
            "DefTransNet",
        )
        df_gpc = select_heldout_level(
            df_gpc_all,
            test_manifest,
            deformation_level,
            "GPC-Reg",
        )
        validate_pairing(df_def, df_gpc, deformation_level)

        level_tag = int(round(deformation_level * 10))
        output_path = output_dir / (
            "eval_modelnet10_mean_distance_distribution_"
            f"level_{level_tag:02d}.pdf"
        )
        plot_level(
            df_def,
            df_gpc,
            deformation_level,
            args.bins,
            output_path,
        )

    print("\nCompleted: both figures use only the 726 held-out test objects.")


if __name__ == "__main__":
    main()
