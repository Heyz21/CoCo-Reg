

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

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
from scipy.stats import spearmanr

from models import DefTransNet as benchmark_mod


MODEL_NAMES = ("Robust-DefReg", "DefTransNet", "GPC-Reg")
MODEL_PREFIXES = {
    "Robust-DefReg": "Robust",
    "DefTransNet": "DefTransNet",
    "GPC-Reg": "GPC",
}
DEFORMATION_LEVELS = [round(i / 10, 1) for i in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final paired Robust-DefReg / DefTransNet / GPC-Reg evaluation"
    )
    parser.add_argument("--data_root", type=str, default="ModelNet10")
    parser.add_argument(
        "--robust_ckpt",
        type=str,
        default="Robust_Trained.pth",
    )
    parser.add_argument(
        "--deftransnet_ckpt",
        type=str,
        default="DeftransNet.pth",
    )
    parser.add_argument(
        "--gpc_ckpt",
        type=str,
        default="Registration/gpcreg/seed_42/Saves/epoch_20.pth",
    )
    parser.add_argument(
        "--gpc_script",
        type=str,
        default="training/train_gpcreg.py",
        help="The final non-AR GPC-Reg implementation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Registration/Evaluation/GPC_Reg_Final",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--validation_count", type=int, default=182)
    parser.add_argument("--num_patches", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--num_geo_blocks", type=int, default=1)
    parser.add_argument("--refine_alpha", type=float, default=0.10)
    parser.add_argument(
        "--positive_overlap",
        type=float,
        default=0.10,
        help="Ground-truth overlap threshold defining a positive patch pair.",
    )
    parser.add_argument(
        "--failure_threshold",
        type=float,
        default=0.10,
        help="Threshold in normalized point-cloud units.",
    )
    parser.add_argument(
        "--boundary_fraction",
        type=float,
        default=0.20,
        help="Fraction of source points with the smallest node-distance margin.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Debug only. Limit the final test samples.",
    )
    parser.add_argument(
        "--no_pair_cache",
        action="store_true",
        help="Do not save the fixed source/target pairs.",
    )
    return parser.parse_args()


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(path_text: str) -> Path:
    return Path(os.path.expanduser(path_text)).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def configure_device(device_text: str | None, gpc_mod) -> torch.device:
    if device_text:
        dev = torch.device(device_text)
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.zeros(1, device=dev)
    if hasattr(benchmark_mod, "device"):
        benchmark_mod.device = dev
    if hasattr(gpc_mod, "device"):
        gpc_mod.device = dev
    return dev


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    state = {}
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    return state


def load_state(
    net: torch.nn.Module,
    checkpoint_path: Path,
    dev: torch.device,
    label: str,
    strict: bool = True,
) -> None:
    state = unwrap_state_dict(torch.load(checkpoint_path, map_location=dev))
    if strict:
        net.load_state_dict(state, strict=True)
    else:
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{label} checkpoint does not exactly match the requested model.\n"
                f"Missing keys ({len(missing)}): {missing[:8]}\n"
                f"Unexpected keys ({len(unexpected)}): {unexpected[:8]}"
            )


def build_models(
    args: argparse.Namespace,
    dev: torch.device,
    gpc_mod,
) -> dict[str, torch.nn.Module]:
    robust_class = getattr(benchmark_mod, "GFN", None)
    deftransnet_class = getattr(benchmark_mod, "DefTransNet", None)
    if robust_class is None:
        raise AttributeError(
            "The local DefTransNet.py does not define class GFN. "
            "Use the same DefTransNet.py that was used by the existing "
            "test_eval_compare.py and Robust_Trained.pth."
        )
    if deftransnet_class is None:
        raise AttributeError("The local DefTransNet.py does not define class DefTransNet.")
    if not hasattr(benchmark_mod, "sLBP_GF"):
        raise AttributeError("The local DefTransNet.py does not define sLBP_GF.")

    robust = robust_class(show=0)
    deftransnet = deftransnet_class(show=0)
    gpc = gpc_mod.GeoPatchDefTransNet(
        token_dim=args.token_dim,
        num_geo_blocks=args.num_geo_blocks,
        show=0,
        refine_alpha=args.refine_alpha,
    )

    load_state(
        robust,
        resolve_path(args.robust_ckpt),
        dev,
        "Robust-DefReg",
        strict=True,
    )
    load_state(
        deftransnet,
        resolve_path(args.deftransnet_ckpt),
        dev,
        "DefTransNet",
        strict=True,
    )
    load_state(
        gpc,
        resolve_path(args.gpc_ckpt),
        dev,
        "GPC-Reg",
        strict=True,
    )

    models = {
        "Robust-DefReg": robust.to(dev).eval(),
        "DefTransNet": deftransnet.to(dev).eval(),
        "GPC-Reg": gpc.to(dev).eval(),
    }
    return models


def build_dataset(args: argparse.Namespace, data_root: Path, gpc_mod):
    if list(gpc_mod.BENCHMARK_DEF_LEVELS) != DEFORMATION_LEVELS:
        raise ValueError(
            f"Unexpected deformation levels in {args.gpc_script}: "
            f"{gpc_mod.BENCHMARK_DEF_LEVELS}"
        )
    if list(gpc_mod.BENCHMARK_TRAIN_TYPES) != ["Deformation_Level"]:
        raise ValueError("The formal evaluation must use Deformation_Level only.")
    if float(gpc_mod.rotation) != 0.0:
        raise ValueError("The formal evaluation expects rotation=0.")

    dataset = gpc_mod.PointCloudData_ModelNet(
        data_root,
        folder="test",
        transform=gpc_mod.train_transforms,
        typ=gpc_mod.BENCHMARK_TRAIN_TYPES,
        rotation=gpc_mod.rotation,
    )
    return dataset


def stratified_split(
    dataset,
    validation_count: int,
    split_seed: int,
    base_seed: int,
) -> tuple[list[int], list[int], pd.DataFrame]:
    if not 0 < validation_count < len(dataset):
        raise ValueError(
            f"validation_count must be between 1 and {len(dataset) - 1}, "
            f"received {validation_count}"
        )

    groups: dict[float, list[int]] = {}
    for index, item in enumerate(dataset.files):
        level = float(item["def_lvl"])
        groups.setdefault(level, []).append(index)

    total = len(dataset)
    exact = {
        level: validation_count * len(indices) / total
        for level, indices in groups.items()
    }
    allocation = {level: int(np.floor(value)) for level, value in exact.items()}
    remaining = validation_count - sum(allocation.values())
    order = sorted(groups, key=lambda level: (-(exact[level] - allocation[level]), level))
    for level in order[:remaining]:
        allocation[level] += 1

    rng = np.random.RandomState(split_seed)
    validation_indices: list[int] = []
    test_indices: list[int] = []
    split_by_index: dict[int, str] = {}

    for level in sorted(groups):
        level_indices = np.asarray(groups[level], dtype=np.int64)
        rng.shuffle(level_indices)
        count = allocation[level]
        val = level_indices[:count].tolist()
        test = level_indices[count:].tolist()
        validation_indices.extend(val)
        test_indices.extend(test)
        split_by_index.update({idx: "validation" for idx in val})
        split_by_index.update({idx: "test" for idx in test})

    manifest_rows = []
    for index, item in enumerate(dataset.files):
        manifest_rows.append(
            {
                "Dataset Index": index,
                "File": str(item["pcd_path"]),
                "Name": item["name"],
                "Category": item["category"],
                "Deformation Level": float(item["def_lvl"]),
                "Split": split_by_index[index],
                "Sample Seed": base_seed + index,
                "Target Shuffle Seed": base_seed + index + 1,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    return sorted(validation_indices), sorted(test_indices), manifest


def shuffle_target(
    target: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    permutation_np = np.arange(target.shape[1])
    rng.shuffle(permutation_np)
    permutation = torch.as_tensor(
        permutation_np,
        dtype=torch.long,
        device=target.device,
    ).unsqueeze(0)
    shuffled = torch.gather(
        target,
        1,
        permutation[..., None].expand(-1, -1, target.shape[-1]),
    )
    return shuffled, permutation


def create_fixed_pairs(
    dataset,
    test_indices: list[int],
    base_seed: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    total = len(test_indices)
    for position, index in enumerate(test_indices):
        sample_seed = base_seed + index
        set_all_seeds(sample_seed)
        data = dataset[index]
        source = data["source_pointcloud"].float().unsqueeze(0).cpu()
        target = data["target_pointcloud"].float().unsqueeze(0).cpu()
        target_shuffled, permutation = shuffle_target(target, sample_seed + 1)
        pairs.append(
            {
                "dataset_index": index,
                "name": data["name"],
                "category": int(data["category"]),
                "deformation_level": float(data["deformation"]),
                "sample_seed": sample_seed,
                "target_shuffle_seed": sample_seed + 1,
                "source": source,
                "target": target,
                "target_shuffled": target_shuffled.cpu(),
                "target_permutation": permutation.cpu(),
            }
        )
        if (position + 1) % 50 == 0 or position + 1 == total:
            print(f"Prepared fixed pair {position + 1}/{total}")
    return pairs


def synchronize(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def begin_memory_measurement(dev: torch.device) -> int:
    if dev.type != "cuda":
        return 0
    synchronize(dev)
    baseline = int(torch.cuda.memory_allocated(dev))
    torch.cuda.reset_peak_memory_stats(dev)
    return baseline


def end_memory_measurement(dev: torch.device, baseline: int) -> float:
    if dev.type != "cuda":
        return float("nan")
    synchronize(dev)
    peak = int(torch.cuda.max_memory_allocated(dev))
    return max(0.0, (peak - baseline) / (1024.0**2))


@torch.no_grad()
def predict_benchmark(
    net: torch.nn.Module,
    source: torch.Tensor,
    target_shuffled: torch.Tensor,
    dev: torch.device,
) -> tuple[torch.Tensor, float, float]:
    baseline_memory = begin_memory_measurement(dev)
    synchronize(dev)
    start = time.perf_counter()
    displacement = benchmark_mod.sLBP_GF(source, target_shuffled, net)
    synchronize(dev)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    memory_mb = end_memory_measurement(dev, baseline_memory)
    return displacement, runtime_ms, memory_mb


@torch.no_grad()
def predict_gpc(
    net: torch.nn.Module,
    source: torch.Tensor,
    target_shuffled: torch.Tensor,
    args: argparse.Namespace,
    gpc_mod,
    dev: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, float]]:
    baseline_memory = begin_memory_measurement(dev)
    synchronize(dev)
    total_start = time.perf_counter()

    feature_start = time.perf_counter()
    source_features, target_features = net.extract_features(source, target_shuffled)
    synchronize(dev)
    feature_ms = (time.perf_counter() - feature_start) * 1000.0

    patch_start = time.perf_counter()
    output = net.matcher(
        source,
        target_shuffled,
        source_features,
        target_features,
        args.num_patches,
        args.patch_size,
    )
    output["src_feat"] = source_features
    output["tgt_feat"] = target_features
    refined_source, refined_target = net.refine_point_features(output)
    synchronize(dev)
    patch_ms = (time.perf_counter() - patch_start) * 1000.0

    slbp_start = time.perf_counter()
    displacement = gpc_mod.inference(
        source,
        target_shuffled,
        refined_source,
        refined_target,
        f=1,
    )
    synchronize(dev)
    slbp_ms = (time.perf_counter() - slbp_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    memory_mb = end_memory_measurement(dev, baseline_memory)
    timings = {
        "Total Runtime ms": total_ms,
        "Backbone Runtime ms": feature_ms,
        "Patch Branch Runtime ms": patch_ms,
        "sLBP Runtime ms": slbp_ms,
        "Temporary Peak GPU Memory MB": memory_mb,
    }
    return displacement, output, timings


def geometric_set_metrics(
    registered: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    distances = torch.cdist(registered, target, p=2)
    registered_to_target = distances.min(dim=-1).values
    target_to_registered = distances.min(dim=-2).values
    chamfer = 0.5 * (
        registered_to_target.mean() + target_to_registered.mean()
    )
    hd95 = torch.maximum(
        torch.quantile(registered_to_target, 0.95),
        torch.quantile(target_to_registered, 0.95),
    )
    return float(chamfer.item()), float(hd95.item())


def point_error_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    displacement: torch.Tensor,
    failure_threshold: float,
) -> tuple[dict[str, float], torch.Tensor]:
    registered = source + displacement
    errors = torch.linalg.norm(registered - target, dim=-1).squeeze(0)
    chamfer, hd95 = geometric_set_metrics(registered, target)
    values = {
        "Mean": float(errors.mean().item()),
        "Median": float(errors.median().item()),
        "Point Std": float(errors.std(unbiased=True).item()),
        "Point P90": float(torch.quantile(errors, 0.90).item()),
        "Point P95": float(torch.quantile(errors, 0.95).item()),
        "Max": float(errors.max().item()),
        "Point Failure Rate": float((errors > failure_threshold).float().mean().item()),
        "Chamfer": chamfer,
        "Hausdorff95": hd95,
    }
    return values, errors


def initial_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    failure_threshold: float,
) -> dict[str, float]:
    zero_displacement = torch.zeros_like(source)
    values, _ = point_error_metrics(
        source,
        target,
        zero_displacement,
        failure_threshold,
    )
    return {f"Initial {key}": value for key, value in values.items()}


def gather_target_patch_identities(
    target_patch_indices: torch.Tensor,
    target_permutation: torch.Tensor,
) -> torch.Tensor:
    batch, patches, _ = target_patch_indices.shape
    expanded_permutation = target_permutation[:, None, :].expand(
        batch,
        patches,
        -1,
    )
    return torch.gather(expanded_permutation, 2, target_patch_indices)


def correct_patch_overlap(
    output: dict[str, Any],
    target_permutation: torch.Tensor,
) -> torch.Tensor:
    source_indices = output["src_part"]["patch_idx"]
    target_identities = gather_target_patch_identities(
        output["tgt_part"]["patch_idx"],
        target_permutation,
    )
    source_masks = output["src_part"]["patch_masks"]
    target_masks = output["tgt_part"]["patch_masks"]
    source_node_masks = output["src_part"]["node_masks"]
    target_node_masks = output["tgt_part"]["node_masks"]

    batch, source_patch_count, _ = source_indices.shape
    target_patch_count = target_identities.shape[1]
    point_count = target_permutation.shape[1]

    source_incidence = torch.zeros(
        batch,
        source_patch_count,
        point_count,
        dtype=torch.float32,
        device=source_indices.device,
    )
    target_incidence = torch.zeros(
        batch,
        target_patch_count,
        point_count,
        dtype=torch.float32,
        device=source_indices.device,
    )
    source_incidence.scatter_add_(
        2,
        source_indices,
        source_masks.float(),
    )
    target_incidence.scatter_add_(
        2,
        target_identities,
        target_masks.float(),
    )
    source_incidence.clamp_(0.0, 1.0)
    target_incidence.clamp_(0.0, 1.0)

    intersection = torch.bmm(
        source_incidence,
        target_incidence.transpose(1, 2),
    )
    source_size = source_incidence.sum(dim=-1).clamp(min=1.0)
    target_size = target_incidence.sum(dim=-1).clamp(min=1.0)
    overlap = 0.5 * (
        intersection / source_size[:, :, None]
        + intersection / target_size[:, None, :]
    )
    valid_pairs = source_node_masks[:, :, None] & target_node_masks[:, None, :]
    return overlap * valid_pairs.float()


def patch_metrics(
    output: dict[str, Any],
    target_permutation: torch.Tensor,
    positive_overlap: float,
) -> tuple[dict[str, float], torch.Tensor]:
    scores = output["scores"]
    overlap = correct_patch_overlap(output, target_permutation)
    source_mask = output["src_part"]["node_masks"]
    target_mask = output["tgt_part"]["node_masks"]
    valid_pairs = source_mask[:, :, None] & target_mask[:, None, :]
    positives = (overlap >= positive_overlap) & valid_pairs

    valid_source_rows = source_mask & positives.any(dim=-1)
    positive_count = positives.sum(dim=-1).float()
    masked_scores = scores.masked_fill(~valid_pairs, -1e4)

    result: dict[str, float] = {}
    for requested_k in (1, 3, 5):
        actual_k = min(requested_k, scores.shape[-1])
        top_indices = masked_scores.topk(actual_k, dim=-1).indices
        top_positive = torch.gather(positives, -1, top_indices)
        row_hit = top_positive.any(dim=-1)
        if bool(valid_source_rows.any()):
            result[f"Patch Top{requested_k} Recall"] = float(
                row_hit[valid_source_rows].float().mean().item()
            )
        else:
            result[f"Patch Top{requested_k} Recall"] = float("nan")

    top5_k = min(5, scores.shape[-1])
    top5_indices = masked_scores.topk(top5_k, dim=-1).indices
    top5_positive = torch.gather(positives, -1, top5_indices).float()
    if bool(valid_source_rows.any()):
        result["Patch Top5 Precision"] = float(
            (top5_positive.sum(dim=-1) / top5_k)[valid_source_rows].mean().item()
        )
        result["Patch Top5 Positive Recall"] = float(
            (
                top5_positive.sum(dim=-1)
                / positive_count.clamp(min=1.0)
            )[valid_source_rows].mean().item()
        )
        result["Positive Target Patches per Source"] = float(
            positive_count[valid_source_rows].mean().item()
        )
    else:
        result["Patch Top5 Precision"] = float("nan")
        result["Patch Top5 Positive Recall"] = float("nan")
        result["Positive Target Patches per Source"] = float("nan")

    top1_indices = masked_scores.argmax(dim=-1, keepdim=True)
    top1_overlap = torch.gather(overlap, -1, top1_indices).squeeze(-1)
    if bool(source_mask.any()):
        result["Top-Scoring Pair GT Overlap"] = float(
            top1_overlap[source_mask].mean().item()
        )
    else:
        result["Top-Scoring Pair GT Overlap"] = float("nan")

    probabilities = torch.softmax(masked_scores, dim=-1)
    entropy = -(
        probabilities
        * torch.log(probabilities.clamp(min=1e-12))
    ).sum(dim=-1)
    valid_target_count = target_mask.sum(dim=-1).float().clamp(min=2.0)
    normalized_entropy = entropy / torch.log(valid_target_count[:, None])
    result["Patch Attention Entropy"] = float(
        normalized_entropy[source_mask].mean().item()
    )
    return result, overlap


def candidate_coverage(
    output: dict[str, Any],
    target_permutation: torch.Tensor,
) -> dict[str, float]:
    scores = output["scores"]
    source_point_patch = output["src_part"]["point_node"]
    target_point_patch_shuffled = output["tgt_part"]["point_node"]
    batch, point_count = target_permutation.shape

    inverse_permutation = torch.empty_like(target_permutation)
    positions = torch.arange(
        point_count,
        device=target_permutation.device,
    )[None, :].expand(batch, -1)
    inverse_permutation.scatter_(1, target_permutation, positions)
    ground_truth_target_patch = torch.gather(
        target_point_patch_shuffled,
        1,
        inverse_permutation,
    )

    score_by_source_point = torch.gather(
        scores,
        1,
        source_point_patch[..., None].expand(-1, -1, scores.shape[-1]),
    )
    result = {}
    for requested_k in (1, 3, 8, 15):
        actual_k = min(requested_k, scores.shape[-1])
        selected = score_by_source_point.topk(actual_k, dim=-1).indices
        covered = (
            selected == ground_truth_target_patch[..., None]
        ).any(dim=-1)
        result[f"Candidate Coverage Top{requested_k}"] = float(
            covered.float().mean().item()
        )
    return result


def boundary_metrics(
    source: torch.Tensor,
    output: dict[str, Any],
    deftransnet_errors: torch.Tensor,
    gpc_errors: torch.Tensor,
    boundary_fraction: float,
) -> dict[str, float]:
    centers = output["src_part"]["nodes"]
    point_center_distances = torch.cdist(source, centers, p=2).squeeze(0)
    nearest_two = point_center_distances.topk(
        k=min(2, centers.shape[1]),
        largest=False,
        dim=-1,
    ).values
    if nearest_two.shape[-1] < 2:
        margin = torch.zeros(source.shape[1], device=source.device)
    else:
        margin = nearest_two[:, 1] - nearest_two[:, 0]
    boundary_count = max(1, int(round(boundary_fraction * source.shape[1])))
    boundary_indices = margin.topk(boundary_count, largest=False).indices
    boundary_mask = torch.zeros(
        source.shape[1],
        dtype=torch.bool,
        device=source.device,
    )
    boundary_mask[boundary_indices] = True
    interior_mask = ~boundary_mask
    return {
        "Boundary Fraction": boundary_fraction,
        "DefTransNet Boundary Mean": float(
            deftransnet_errors[boundary_mask].mean().item()
        ),
        "DefTransNet Interior Mean": float(
            deftransnet_errors[interior_mask].mean().item()
        ),
        "GPC Boundary Mean": float(gpc_errors[boundary_mask].mean().item()),
        "GPC Interior Mean": float(gpc_errors[interior_mask].mean().item()),
    }


def parameter_counts(models: dict[str, torch.nn.Module]) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        rows.append(
            {
                "Model": model_name,
                "Parameters": total,
                "Trainable Parameters": trainable,
            }
        )
    gpc = models["GPC-Reg"]
    patch_parameters = sum(
        parameter.numel()
        for module in (gpc.matcher, gpc.point_context_proj)
        for parameter in module.parameters()
    )
    rows.append(
        {
            "Model": "GPC-Reg patch branch only",
            "Parameters": patch_parameters,
            "Trainable Parameters": patch_parameters,
        }
    )
    return pd.DataFrame(rows)


def add_prefixed_metrics(
    row: dict[str, Any],
    model_name: str,
    values: dict[str, float],
) -> None:
    prefix = MODEL_PREFIXES[model_name]
    for key, value in values.items():
        row[f"{prefix} {key}"] = value


def evaluate(
    args: argparse.Namespace,
    pairs: list[dict[str, Any]],
    models: dict[str, torch.nn.Module],
    gpc_mod,
    dev: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]]]:
    rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = {
        "Robust-DefReg Total Runtime ms": [],
        "DefTransNet Total Runtime ms": [],
        "GPC-Reg Total Runtime ms": [],
        "GPC-Reg Backbone Runtime ms": [],
        "GPC-Reg Patch Branch Runtime ms": [],
        "GPC-Reg sLBP Runtime ms": [],
        "Robust-DefReg Temporary Peak GPU Memory MB": [],
        "DefTransNet Temporary Peak GPU Memory MB": [],
        "GPC-Reg Temporary Peak GPU Memory MB": [],
    }

    total = len(pairs)
    for position, pair in enumerate(pairs):
        source = pair["source"].to(dev)
        target = pair["target"].to(dev)
        target_shuffled = pair["target_shuffled"].to(dev)
        permutation = pair["target_permutation"].to(dev)

        robust_displacement, robust_ms, robust_memory = predict_benchmark(
            models["Robust-DefReg"],
            source,
            target_shuffled,
            dev,
        )
        deftransnet_displacement, deftransnet_ms, deftransnet_memory = (
            predict_benchmark(
                models["DefTransNet"],
                source,
                target_shuffled,
                dev,
            )
        )
        gpc_displacement, patch_output, gpc_timing = predict_gpc(
            models["GPC-Reg"],
            source,
            target_shuffled,
            args,
            gpc_mod,
            dev,
        )

        robust_values, robust_errors = point_error_metrics(
            source,
            target,
            robust_displacement,
            args.failure_threshold,
        )
        deftransnet_values, deftransnet_errors = point_error_metrics(
            source,
            target,
            deftransnet_displacement,
            args.failure_threshold,
        )
        gpc_values, gpc_errors = point_error_metrics(
            source,
            target,
            gpc_displacement,
            args.failure_threshold,
        )

        row: dict[str, Any] = {
            "Dataset Index": pair["dataset_index"],
            "Name": pair["name"],
            "Category": pair["category"],
            "Deformation Level": pair["deformation_level"],
            "Sample Seed": pair["sample_seed"],
            "Target Shuffle Seed": pair["target_shuffle_seed"],
            **initial_metrics(source, target, args.failure_threshold),
        }
        add_prefixed_metrics(row, "Robust-DefReg", robust_values)
        add_prefixed_metrics(row, "DefTransNet", deftransnet_values)
        add_prefixed_metrics(row, "GPC-Reg", gpc_values)
        row["GPC minus DefTransNet Mean"] = (
            row["GPC Mean"] - row["DefTransNet Mean"]
        )
        row["GPC Absolute Improvement"] = (
            row["DefTransNet Mean"] - row["GPC Mean"]
        )
        row["GPC Percentage Improvement"] = (
            100.0
            * row["GPC Absolute Improvement"]
            / max(row["DefTransNet Mean"], 1e-12)
        )
        row["GPC Better Than DefTransNet"] = (
            row["GPC minus DefTransNet Mean"] < 0
        )
        rows.append(row)

        diagnostics, _ = patch_metrics(
            patch_output,
            permutation,
            args.positive_overlap,
        )
        diagnostics.update(candidate_coverage(patch_output, permutation))
        diagnostics.update(
            boundary_metrics(
                source,
                patch_output,
                deftransnet_errors,
                gpc_errors,
                args.boundary_fraction,
            )
        )
        diagnostics.update(
            {
                "Dataset Index": pair["dataset_index"],
                "Name": pair["name"],
                "Deformation Level": pair["deformation_level"],
                "GPC Absolute Improvement": row["GPC Absolute Improvement"],
            }
        )
        patch_rows.append(diagnostics)

        timing["Robust-DefReg Total Runtime ms"].append(robust_ms)
        timing["DefTransNet Total Runtime ms"].append(deftransnet_ms)
        timing["GPC-Reg Total Runtime ms"].append(
            gpc_timing["Total Runtime ms"]
        )
        timing["GPC-Reg Backbone Runtime ms"].append(
            gpc_timing["Backbone Runtime ms"]
        )
        timing["GPC-Reg Patch Branch Runtime ms"].append(
            gpc_timing["Patch Branch Runtime ms"]
        )
        timing["GPC-Reg sLBP Runtime ms"].append(
            gpc_timing["sLBP Runtime ms"]
        )
        timing["Robust-DefReg Temporary Peak GPU Memory MB"].append(
            robust_memory
        )
        timing["DefTransNet Temporary Peak GPU Memory MB"].append(
            deftransnet_memory
        )
        timing["GPC-Reg Temporary Peak GPU Memory MB"].append(
            gpc_timing["Temporary Peak GPU Memory MB"]
        )

        if (
            position == 0
            or (position + 1) % 10 == 0
            or position + 1 == total
        ):
            print(
                f"[{position + 1}/{total}] {pair['name']} "
                f"level={pair['deformation_level']:.1f} "
                f"Robust={row['Robust Mean']:.6f} "
                f"DefTransNet={row['DefTransNet Mean']:.6f} "
                f"GPC-Reg={row['GPC Mean']:.6f}"
            )

    return pd.DataFrame(rows), pd.DataFrame(patch_rows), timing


def summary_by_level(
    per_sample: pd.DataFrame,
    failure_threshold: float,
) -> pd.DataFrame:
    rows = []
    for level in sorted(per_sample["Deformation Level"].unique()):
        level_data = per_sample[
            per_sample["Deformation Level"] == level
        ]
        for model_name in MODEL_NAMES:
            prefix = MODEL_PREFIXES[model_name]
            sample_means = level_data[f"{prefix} Mean"]
            rows.append(
                {
                    "Deformation Level": level,
                    "Model": model_name,
                    "Count": len(level_data),
                    "Mean Identity Error": sample_means.mean(),
                    "Std Across Samples": sample_means.std(ddof=1),
                    "Median Identity Error": sample_means.median(),
                    "P90 Sample Error": sample_means.quantile(0.90),
                    "P95 Sample Error": sample_means.quantile(0.95),
                    "Sample Failure Rate": (
                        sample_means > failure_threshold
                    ).mean(),
                    "Mean Chamfer": level_data[f"{prefix} Chamfer"].mean(),
                    "Mean Hausdorff95": level_data[
                        f"{prefix} Hausdorff95"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def paired_summary(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in sorted(per_sample["Deformation Level"].unique()):
        group = per_sample[per_sample["Deformation Level"] == level]
        def_mean = group["DefTransNet Mean"].mean()
        gpc_mean = group["GPC Mean"].mean()
        rows.append(
            {
                "Deformation Level": level,
                "Count": len(group),
                "DefTransNet Mean": def_mean,
                "GPC-Reg Mean": gpc_mean,
                "GPC minus DefTransNet": gpc_mean - def_mean,
                "Absolute Improvement": def_mean - gpc_mean,
                "Percentage Improvement": (
                    100.0 * (def_mean - gpc_mean) / max(def_mean, 1e-12)
                ),
                "GPC Win Rate": group[
                    "GPC Better Than DefTransNet"
                ].mean(),
                "Paired Difference Std": group[
                    "GPC minus DefTransNet Mean"
                ].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def patch_summary(patch_data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in patch_data.select_dtypes(include=[np.number]).columns
        if column
        not in (
            "Dataset Index",
            "Deformation Level",
            "GPC Absolute Improvement",
        )
    ]
    return (
        patch_data.groupby("Deformation Level")[numeric_columns]
        .mean()
        .reset_index()
    )


def patch_correlations(patch_data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Patch Top1 Recall",
        "Patch Top3 Recall",
        "Patch Top5 Recall",
        "Top-Scoring Pair GT Overlap",
        "Patch Attention Entropy",
        "Candidate Coverage Top1",
        "Candidate Coverage Top3",
        "Candidate Coverage Top8",
        "Candidate Coverage Top15",
    ]
    rows = []
    for metric in metrics:
        valid = patch_data[[metric, "GPC Absolute Improvement"]].dropna()
        if len(valid) < 3:
            coefficient = float("nan")
            p_value = float("nan")
        else:
            coefficient, p_value = spearmanr(
                valid[metric],
                valid["GPC Absolute Improvement"],
            )
        rows.append(
            {
                "Patch Metric": metric,
                "Spearman Correlation with GPC Improvement": coefficient,
                "p-value": p_value,
                "Count": len(valid),
            }
        )
    return pd.DataFrame(rows)


def efficiency_summary(
    timing: dict[str, list[float]],
    parameter_table: pd.DataFrame,
) -> pd.DataFrame:
    parameter_lookup = dict(
        zip(parameter_table["Model"], parameter_table["Parameters"])
    )
    rows = []
    for model_name in MODEL_NAMES:
        runtime_values = np.asarray(
            timing[f"{model_name} Total Runtime ms"],
            dtype=np.float64,
        )
        memory_values = np.asarray(
            timing[f"{model_name} Temporary Peak GPU Memory MB"],
            dtype=np.float64,
        )
        rows.append(
            {
                "Model": model_name,
                "Parameters": parameter_lookup[model_name],
                "Mean Runtime per Pair ms": np.nanmean(runtime_values),
                "Median Runtime per Pair ms": np.nanmedian(runtime_values),
                "Runtime Std ms": np.nanstd(runtime_values, ddof=1),
                "Max Temporary Peak GPU Memory MB": (
                    np.nan
                    if np.isnan(memory_values).all()
                    else np.nanmax(memory_values)
                ),
                "Mean Patch Branch Runtime ms": (
                    np.nan
                    if model_name != "GPC-Reg"
                    else np.mean(timing["GPC-Reg Patch Branch Runtime ms"])
                ),
            }
        )
    return pd.DataFrame(rows)


def qualitative_candidates(per_sample: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for level, label in ((0.1, "low"), (0.5, "moderate"), (0.9, "strong")):
        group = per_sample[
            np.isclose(per_sample["Deformation Level"], level)
        ].copy()
        if group.empty:
            continue
        median_error = group["GPC Mean"].median()
        chosen = group.iloc[
            (group["GPC Mean"] - median_error).abs().argmin()
        ].copy()
        chosen["Selection"] = f"{label} deformation representative"
        selected_rows.append(chosen)

    failures = per_sample.sort_values(
        "GPC minus DefTransNet Mean",
        ascending=False,
    ).head(2)
    for rank, (_, chosen) in enumerate(failures.iterrows(), start=1):
        chosen = chosen.copy()
        chosen["Selection"] = f"failure case {rank}"
        selected_rows.append(chosen)
    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows)


def save_plot(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_by_level(per_sample: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    colors = {
        "Robust-DefReg": "#777777",
        "DefTransNet": "#2f6db0",
        "GPC-Reg": "#c43c39",
    }
    for model_name in MODEL_NAMES:
        prefix = MODEL_PREFIXES[model_name]
        grouped = per_sample.groupby("Deformation Level")[f"{prefix} Mean"]
        means = grouped.mean()
        standard_deviations = grouped.std()
        axis.errorbar(
            means.index,
            means.values,
            yerr=standard_deviations.values,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name,
            color=colors[model_name],
        )
    axis.set_xlabel("Deformation level")
    axis.set_ylabel("Identity-specific mean distance")
    axis.set_xticks(DEFORMATION_LEVELS)
    axis.grid(alpha=0.25)
    axis.legend()
    save_plot(fig, output_dir / "eval_mean_error_by_deformation")


def plot_paired_difference(per_sample: pd.DataFrame, output_dir: Path) -> None:
    data = []
    labels = []
    for level in sorted(per_sample["Deformation Level"].unique()):
        values = per_sample.loc[
            per_sample["Deformation Level"] == level,
            "GPC minus DefTransNet Mean",
        ].to_numpy()
        data.append(values)
        labels.append(f"{level:.1f}")
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.boxplot(data, labels=labels, showfliers=False)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Deformation level")
    axis.set_ylabel("GPC-Reg minus DefTransNet error")
    axis.grid(axis="y", alpha=0.25)
    save_plot(fig, output_dir / "eval_paired_error_difference")


def plot_error_distribution(
    per_sample: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    values = [
        per_sample["Initial Mean"].to_numpy(),
        per_sample["Robust Mean"].to_numpy(),
        per_sample["DefTransNet Mean"].to_numpy(),
        per_sample["GPC Mean"].to_numpy(),
    ]
    labels = ["Before registration", "Robust-DefReg", "DefTransNet", "GPC-Reg"]
    axis.hist(values, bins=35, label=labels, histtype="step", linewidth=1.7)
    axis.set_xlabel("Per-sample identity-specific mean distance")
    axis.set_ylabel("Number of samples")
    axis.grid(alpha=0.20)
    axis.legend()
    save_plot(fig, output_dir / "eval_error_distribution")


def write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    dev: torch.device,
    dataset_size: int,
    validation_size: int,
    test_size: int,
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "device_resolved": str(dev),
            "dataset_size": dataset_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "model_names": list(MODEL_NAMES),
            "formal_gpc_version": "non-auto-regressive",
            "checkpoint_selection": {
                "Robust-DefReg": "provided Robust_Trained.pth",
                "DefTransNet": "provided DeftransNet.pth (20 epochs)",
                "GPC-Reg": "fixed epoch 20",
            },
            "important_interpretation": (
                "This run evaluates one provided checkpoint per model. "
                "It does not estimate variation across independently trained model seeds."
            ),
        }
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    gpc_script = resolve_path(args.gpc_script)
    output_dir = resolve_path(args.output_dir)
    robust_ckpt = resolve_path(args.robust_ckpt)
    deftransnet_ckpt = resolve_path(args.deftransnet_ckpt)
    gpc_ckpt = resolve_path(args.gpc_ckpt)

    require_dir(data_root, "ModelNet10 dataset")
    require_file(gpc_script, "GPC-Reg script")
    require_file(robust_ckpt, "Robust-DefReg checkpoint")
    require_file(deftransnet_ckpt, "DefTransNet checkpoint")
    require_file(gpc_ckpt, "GPC-Reg checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpc_mod = import_module_from_path("gpcreg_model", gpc_script)
    dev = configure_device(args.device, gpc_mod)
    set_all_seeds(args.seed)

    print("=== Final three-model paired evaluation ===")
    print(f"Device:          {dev}")
    print(f"Robust-DefReg:   {robust_ckpt}")
    print(f"DefTransNet:     {deftransnet_ckpt}")
    print(f"GPC-Reg:         {gpc_ckpt}")
    print(f"GPC-Reg script:  {gpc_script}")
    print(f"Output:          {output_dir}")

    dataset = build_dataset(args, data_root, gpc_mod)
    validation_indices, test_indices, manifest = stratified_split(
        dataset,
        args.validation_count,
        args.split_seed,
        args.seed,
    )
    if args.max_samples is not None:
        test_indices = test_indices[: args.max_samples]
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    split_counts = (
        manifest.groupby(["Deformation Level", "Split"])
        .size()
        .unstack(fill_value=0)
    )
    split_counts.to_csv(output_dir / "split_counts_by_level.csv")
    print(split_counts.to_string())
    print(
        f"Dataset={len(dataset)}, validation={len(validation_indices)}, "
        f"final test={len(test_indices)}"
    )

    pairs = create_fixed_pairs(dataset, test_indices, args.seed)
    if not args.no_pair_cache:
        torch.save(
            {
                "formal_model": "GPC-Reg non-auto-regressive",
                "base_seed": args.seed,
                "split_seed": args.split_seed,
                "pairs": pairs,
            },
            output_dir / "fixed_test_pairs.pt",
        )

    models = build_models(args, dev, gpc_mod)
    parameter_table = parameter_counts(models)
    parameter_table.to_csv(output_dir / "parameter_counts.csv", index=False)

    per_sample, patch_data, timing = evaluate(
        args,
        pairs,
        models,
        gpc_mod,
        dev,
    )
    per_sample.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    patch_data.to_csv(output_dir / "patch_diagnostics.csv", index=False)

    level_summary = summary_by_level(per_sample, args.failure_threshold)
    paired = paired_summary(per_sample)
    patches = patch_summary(patch_data)
    correlations = patch_correlations(patch_data)
    efficiency = efficiency_summary(timing, parameter_table)
    candidates = qualitative_candidates(per_sample)

    level_summary.to_csv(output_dir / "summary_by_level.csv", index=False)
    paired.to_csv(output_dir / "paired_summary_by_level.csv", index=False)
    patches.to_csv(output_dir / "patch_summary_by_level.csv", index=False)
    correlations.to_csv(output_dir / "patch_improvement_correlations.csv", index=False)
    efficiency.to_csv(output_dir / "efficiency_summary.csv", index=False)
    candidates.to_csv(output_dir / "qualitative_candidates.csv", index=False)

    plot_mean_by_level(per_sample, output_dir)
    plot_paired_difference(per_sample, output_dir)
    plot_error_distribution(per_sample, output_dir)
    write_run_config(
        args,
        output_dir,
        dev,
        len(dataset),
        len(validation_indices),
        len(test_indices),
    )

    print("\n=== Evaluation completed ===")
    print(f"Results folder: {output_dir}")
    print("Main files:")
    print("  per_sample_metrics.csv")
    print("  summary_by_level.csv")
    print("  paired_summary_by_level.csv")
    print("  patch_diagnostics.csv")
    print("  efficiency_summary.csv")
    print("  eval_mean_error_by_deformation.pdf")
    print("  eval_paired_error_difference.pdf")
    print("  eval_error_distribution.pdf")


if __name__ == "__main__":
    main()
