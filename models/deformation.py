
import torch
from torch_tps import ThinPlateSpline

SHIFT_SCALE = 0.12
DEFAULT_ALPHA = 2.0
DEFAULT_RESOLUTION = 7


def build_control_grid(resolution=DEFAULT_RESOLUTION, device=None, dtype=torch.float32):
    xs = torch.linspace(-1, 1, steps=resolution, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, steps=resolution, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, steps=resolution, device=device, dtype=dtype)
    x, y, z = torch.meshgrid(xs, ys, zs, indexing="xy")
    return torch.stack([x, y, z], dim=3).reshape(-1, 3)


def border_mask(ctrl, tol=1e-6):
    return (
        (ctrl[:, 0].abs() > 1 - tol)
        | (ctrl[:, 1].abs() > 1 - tol)
        | (ctrl[:, 2].abs() > 1 - tol)
    )


def amplitude_from_level(def_lvl):
    """Linear map: def_lvl 0.1→0.9 gives uniformly spaced control shift."""
    return def_lvl * SHIFT_SCALE


def deform_gaussian(ctrl, pc, def_lvl, tps, seed=None):
    """Zero-mean Gaussian shift on control points + border anchors."""
    if seed is not None:
        torch.manual_seed(seed)

    ctrl = ctrl.to(pc.device, pc.dtype)
    amp = amplitude_from_level(def_lvl)
    shift = torch.randn_like(ctrl) * amp
    shift[border_mask(ctrl)] = 0.0

    perturbed = ctrl + shift
    tps.fit(ctrl, perturbed)
    return tps.transform(pc)


class TPSDeform:
    """Drop-in helper for test / visualize scripts."""

    def __init__(self, resolution=DEFAULT_RESOLUTION, alpha=DEFAULT_ALPHA):
        self.tps = ThinPlateSpline(alpha)
        self.ctrl = build_control_grid(resolution)

    def deform(self, pc, def_lvl, seed=None):
        return deform_gaussian(self.ctrl, pc, def_lvl, self.tps, seed=seed)
