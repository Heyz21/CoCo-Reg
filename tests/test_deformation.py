
import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.DefTransNet import Normalize_ModelNet, PointSampler, read_off
from models.deformation import SHIFT_SCALE, TPSDeform

LEVELS_09 = [round(i / 10, 1) for i in range(1, 10)]


def sample_pointcloud(pcd_path, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    with open(pcd_path, "r") as f:
        verts, faces = read_off(f)
    data = (verts, faces, 0.0, 0, 0, 0.0)
    data = PointSampler(1024)(data)
    data = Normalize_ModelNet()(data)
    return torch.from_numpy(np.asarray(data[0])).float()


def plot_single(source, target, def_lvl, sample_name, out_path):
    fig = plt.figure(figsize=(8, 4.5))

    def _plot(ax, pts, title):
        p = pts.numpy()
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#1f77b4", s=5, depthshade=False)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        mid = p.mean(0)
        r = max(np.max(np.abs(p - mid)) * 1.1, 0.5)
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)
        ax.view_init(elev=20, azim=45)

    _plot(fig.add_subplot(1, 2, 1, projection="3d"), source, "Source")
    d = (target - source).norm(dim=1).mean().item()
    _plot(fig.add_subplot(1, 2, 2, projection="3d"), target, f"Gaussian\nd={def_lvl} disp={d:.4f}")

    fig.suptitle(f"{sample_name}  SHIFT_SCALE={SHIFT_SCALE}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_progression(source, targets_by_level, sample_name, out_path):
    fig = plt.figure(figsize=(14, 14))
    for i, (lvl, tgt) in enumerate(targets_by_level.items()):
        ax = fig.add_subplot(3, 3, i+1, projection="3d")
        p = tgt.numpy()
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#1f77b4", s=4, depthshade=False)
        d = (tgt - source).norm(dim=1).mean().item()
        ax.set_title(f"d={lvl:.1f}  disp={d:.4f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        mid = p.mean(0)
        r = max(np.max(np.abs(p - mid)) * 1.1, 0.5)
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)
        ax.view_init(elev=20, azim=45)

    fig.suptitle(f"{sample_name}  Gaussian deformation 0.1→0.9", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--def-lvl", type=float, default=0.1)
    parser.add_argument("--progression", action="store_true", help="Show 0.1-0.9 grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    source = sample_pointcloud(args.sample, seed=args.seed)
    sample_name = os.path.basename(args.sample)
    out_dir = "Registration/Plots/DeformationTest"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(sample_name)[0]
    deformer = TPSDeform()

    print(f"SHIFT_SCALE = {SHIFT_SCALE}  (amp = def_lvl * {SHIFT_SCALE})")
    print(f"Levels 0.1-0.9 amplitudes: {[round(l*SHIFT_SCALE,4) for l in LEVELS_09]}\n")

    if args.progression:
        by_level = {}
        for lvl in LEVELS_09:
            by_level[lvl] = deformer.deform(source, lvl, seed=args.seed)
            d = (by_level[lvl] - source).norm(dim=1)
            print(f"d={lvl:.1f}  mean disp={d.mean():.5f}")
        out_path = args.output or f"{out_dir}/{stem}_progression_0.1-0.9.png"
        plot_progression(source, by_level, sample_name, out_path)
        print(f"\nSaved: {out_path}")
        return

    target = deformer.deform(source, args.def_lvl, seed=args.seed)
    d = (target - source).norm(dim=1)
    print(f"gaussian  mean disp={d.mean():.5f}  max disp={d.max():.5f}")

    out_path = args.output or f"{out_dir}/{stem}_d{args.def_lvl}.png"
    plot_single(source, target, args.def_lvl, sample_name, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
