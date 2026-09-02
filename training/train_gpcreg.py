

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

# deterministic cuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch_tps import ThinPlateSpline

from models.deformation import DEFAULT_ALPHA, DEFAULT_RESOLUTION, deform_gaussian

# settings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

k = 10
k1 = 128

slbp_iter = 3
slbp_cost_scale = 10
slbp_alpha = -50

BENCHMARK_DEF_LEVELS = [round(i / 10, 1) for i in range(1, 10)]
BENCHMARK_TRAIN_TYPES = ["Deformation_Level"]
rotation = 0


# keypoints / graph


def pdist(x, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def pdist2(x, y, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = (y ** 2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist


def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    dev = kpts.device

    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self) :]
    A = torch.zeros(B, N, N).to(dev)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1

    return ind, dist * A, A


def lbp_graph(kpts_fixed):
    A = knn_graph(kpts_fixed, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    return edges, edges_reverse_idx


def inference(kpts_fixed, kpts_moving, kpts_fixed_feat, kpts_moving_feat, f=1):
    N_p_fixed = kpts_fixed.shape[1]
    if f:
        dist = pdist2(kpts_fixed_feat, kpts_moving_feat)
    else:
        dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = -kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = (kpts_fixed_feat.view(1, N_p_fixed, 1, -1) - kpts_moving_feat[:, ind.view(-1), :].view(1, N_p_fixed, k1, -1)).pow(2).mean(3)
    edges, edges_reverse_idx = lbp_graph(kpts_fixed)
    messages = torch.zeros((edges.shape[0], k1)).to(device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]
    candidates_edges1 = candidates[0, edges[:, 1], :, :]
    for _ in range(slbp_iter):
        temp_message = torch.zeros((N_p_fixed, k1)).to(device).scatter_add_(
            0, edges[:, 1].view(-1, 1).expand(-1, k1), messages
        )
        multi_data_cost = torch.gather(
            temp_message + candidates_cost.squeeze(), 0, edges[:, 0].view(-1, 1).expand(-1, k1)
        )
        reverse_messages = torch.gather(messages, 0, edges_reverse_idx.view(-1, 1).expand(-1, k1))
        multi_data_cost -= reverse_messages
        messages = torch.zeros_like(multi_data_cost)
        unroll_factor = 32
        split = torch.chunk(torch.arange(multi_data_cost.shape[0]), unroll_factor)
        for i in range(unroll_factor):
            messages[split[i]] = torch.min(
                multi_data_cost[split[i]].unsqueeze(1)
                + slbp_cost_scale * (candidates_edges0[split[i]].unsqueeze(1) - candidates_edges1[split[i]].unsqueeze(2)).pow(2).sum(3),
                2,
            )[0]
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1)).unsqueeze(0)
    sm = F.softmax(slbp_alpha * reg_candidates_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)
    kpts_fixed_disp_pred = (candidates * sm).sum(2)
    return kpts_fixed_disp_pred


# network architectures


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x, ind):
        B, N, D = x.shape
        kn = ind.shape[2]

        y = x.reshape(B * N, D)[ind.reshape(B * N, kn)].reshape(B, N, kn, D)
        x = x.reshape(B, N, 1, D).expand(B, N, kn, D)

        x = torch.cat([y - x, x], dim=3)

        x = self.conv(x.permute(0, 3, 1, 2))
        x = F.max_pool2d(x, (1, kn))
        x = x.squeeze(3).permute(0, 2, 1)

        return x


class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

    def forward(self, input):
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool_size = int(xb.size(-1))
        pool = F.max_pool1d(xb, pool_size).squeeze(-1)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super().__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dims = 64
        self.N = 1
        self.dropout = 0.0
        self.ff_dims = 1024
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
            Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
        )

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class DefTransNet(nn.Module):
    def __init__(self, D=3, show=0):
        super().__init__()
        self.input_transform = Tnet(k=D)

        self.conv1 = EdgeConv(D, 32)
        self.conv2 = EdgeConv(32, 32)
        self.conv3 = EdgeConv(32, 64)

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.InstanceNorm1d(64),
            nn.Conv1d(64, 64, 1),
        )

        self.transform = Transformer()
        self.show = show

    def forward(self, x, y, k):
        matrix3x3x = self.input_transform(x.transpose(1, 2))
        x = torch.bmm(x, matrix3x3x)

        matrix3x3y = self.input_transform(y.transpose(1, 2))
        y = torch.bmm(y, matrix3x3y)

        fixed_ind = knn_graph(x, k, include_self=True)[0]
        x = self.conv1(x, fixed_ind)
        x = self.conv2(x, fixed_ind)
        x = self.conv3(x, fixed_ind)

        moving_ind = knn_graph(y, k * 3, include_self=True)[0]
        y = self.conv1(y, moving_ind)
        y = self.conv2(y, moving_ind)
        y = self.conv3(y, moving_ind)

        xfp, yfp = self.transform(x.transpose(1, 2), y.transpose(1, 2))
        x = x + xfp.transpose(1, 2)
        y = y + yfp.transpose(1, 2)

        x = self.conv4(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.conv4(y.permute(0, 2, 1)).permute(0, 2, 1)

        return x, y


def netloss(disp, disp_pred):
    criterion = torch.nn.L1Loss()
    return criterion(disp, disp_pred)


# ModelNet


def read_off(file):
    if "OFF" != file.readline().strip():
        raise ValueError("Not a valid OFF header")
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(" ")])
    verts = [[float(s) for s in file.readline().strip().split(" ")] for _ in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(" ")][1:] for _ in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces, def_lvl, typ_nr, setting, rotation = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]])

        sampled_faces = random.choices(faces, weights=areas, cum_weights=None, k=self.output_size)

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(
                verts[sampled_faces[i][0]],
                verts[sampled_faces[i][1]],
                verts[sampled_faces[i][2]],
            )

        return (sampled_points, def_lvl, typ_nr, setting, rotation)


class Normalize_ModelNet(object):
    def __call__(self, inp):
        pointcloud, def_lvl, typ_nr, setting, rotation = inp
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return (norm_pointcloud, def_lvl, typ_nr, setting, rotation)


class TPS(object):
    """Thin plate spline deformation."""

    def __init__(self, enabled=True, resolution=DEFAULT_RESOLUTION, alpha=DEFAULT_ALPHA):
        self.enabled = enabled
        self.tps = ThinPlateSpline(alpha)
        xs = torch.linspace(-1, 1, steps=resolution)
        ys = torch.linspace(-1, 1, steps=resolution)
        zs = torch.linspace(-1, 1, steps=resolution)
        x, y, z = torch.meshgrid(xs, ys, zs, indexing="xy")
        self.xyz = torch.stack([x, y, z], dim=3).reshape(-1, 3)

    def fit(self, pc, def_lvl):
        return deform_gaussian(self.xyz, pc, def_lvl, self.tps, seed=None)

    def __call__(self, inp):
        source, def_lvl, typ_nr, setting, rotation = inp
        source = torch.from_numpy(source).float()
        if not self.enabled:
            return (source, source.clone(), typ_nr, setting, rotation)
        target = self.fit(source, def_lvl)
        return (source, target, typ_nr, setting, rotation)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


class RandRotation_z(object):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __call__(self, inp):
        if not self.enabled:
            return inp[0], inp[1]
        rotation = inp[2]
        return (inp[0], self.rotate(inp[1].float(), rotation))

    def rotate(self, pointcloud, rotation):
        assert len(pointcloud.shape) == 2
        rot_matrix = _axis_angle_rotation("Z", torch.tensor(rotation))
        rot_pointcloud = torch.mm(rot_matrix.double(), pointcloud.double().T).T
        return rot_pointcloud


class Modify(object):
    """Add missing points, noise or outliers."""

    def __call__(self, inp):
        source, target, typ_nr, setting, rotation = inp
        if typ_nr == 0:
            return source, target, rotation
        elif typ_nr == 1:
            return self.incomp(source, setting), target, rotation
        elif typ_nr == 2:
            return source, self.nois(target, setting), rotation
        elif typ_nr == 3:
            return source, self.out(target, setting), rotation

    def incomp(self, pointcloud, setting):
        dist = torch.norm(pointcloud - pointcloud[int(torch.rand((1)) * len(pointcloud))], dim=1, p=None)
        knn = dist.topk(int(len(pointcloud) * (1 - setting / 100)))
        return pointcloud[knn.indices], knn.indices

    def nois(self, pointcloud, noise_lvl):
        assert len(pointcloud.shape) == 2
        noisy_pointcloud = torch.normal(pointcloud, noise_lvl)
        return noisy_pointcloud

    def out(self, pointcloud, setting):
        dims = pointcloud.shape
        outliers = (torch.rand((int(dims[0] * setting / 100), dims[1])) - 0.5) * 2
        return torch.cat((pointcloud, outliers), 0)


class PointCloudData_ModelNet(Dataset):
    def __init__(
        self,
        root_dir,
        transform,
        folder="train",
        typ=None,
        rotation=1 / 8,
        def_levels=None,
        seed=42,
    ):
        if typ is None:
            typ = ["Deformation_Level"]
        # local random state for the dataset
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        self.root_dir = Path(root_dir)
        folders = [d for d in sorted(os.listdir(self.root_dir)) if (self.root_dir / d).is_dir()]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        self.types_nr_dict = {
            "Deformation_Level": 0,
            "Incompleteness_Data": 1,
            "Noisy_Data": 2,
            "Outlier_Data": 3,
        }
        self.settings = [[0], [0, 5, 10, 15, 20, 25], [0, 0.01, 0.02, 0.03, 0.04], [0, 5, 15, 25, 35, 45]]
        self.def_levels = [float(x) for x in def_levels] if def_levels else None
        for category in self.classes.keys():
            new_dir = self.root_dir / category / folder
            for file in sorted(os.listdir(new_dir)):
                if file.endswith(".off"):
                    if len(typ) > 1:
                        typ_nrs = []
                        for t in typ:
                            typ_nrs.append(self.types_nr_dict[t])
                        choice = int(np_rng.integers(0, len(typ_nrs)))
                        typ_nr = typ_nrs[choice]
                        typ_name = typ[choice]
                    else:
                        typ_nr = self.types_nr_dict[typ[0]]
                        typ_name = typ[0]
                    sample = {}
                    sample["pcd_path"] = new_dir / file
                    sample["category"] = category
                    sample["name"] = file
                    sample["def_lvl"] = (
                        py_rng.choice(self.def_levels)
                        if self.def_levels
                        else py_rng.randrange(1, 10, 1) / 10
                    )
                    sample["type"] = typ_name
                    sample["type_nr"] = typ_nr
                    sample["setting"] = self.settings[typ_nr][py_rng.randrange(0, len(self.settings[typ_nr]), 1)]
                    sample["rotation"] = np.around(py_rng.random() * np.pi * 2.0 * rotation, 1)
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, def_lvl, typ_nr, setting, rotation):
        verts, faces = read_off(file)
        if self.transforms:
            source, target = self.transforms((verts, faces, def_lvl, typ_nr, setting, rotation))
        return source, target

    def __getitem__(self, idx):
        pcd_path = self.files[idx]["pcd_path"]
        category = self.files[idx]["category"]
        name = self.files[idx]["name"]
        def_lvl = self.files[idx]["def_lvl"]
        typ = self.files[idx]["type"]
        typ_nr = self.files[idx]["type_nr"]
        setting = self.files[idx]["setting"]
        rotation = self.files[idx]["rotation"]
        with open(pcd_path, "r") as f:
            source, target = self.__preproc__(f, def_lvl, typ_nr, setting, rotation)
        if typ_nr == 1:
            valid_ind = source[1]
            source = source[0]
            disp = target[valid_ind] - source
        else:
            disp = target[: source.shape[0]] - source
            valid_ind = np.arange(0, len(source), 1)
        return {
            "source_pointcloud": source,
            "target_pointcloud": target,
            "disp": disp,
            "category": self.classes[category],
            "name": name,
            "deformation": def_lvl,
            "type": typ,
            "setting": setting,
            "valid_ind": valid_ind,
            "rotation": rotation,
        }


class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


train_transforms = Compose(
    [
        PointSampler(1024),
        Normalize_ModelNet(),
        TPS(),
        Modify(),
        RandRotation_z(enabled=False),
    ]
)


# GeoPatch


def _format_seconds(seconds: float) -> str:
    seconds = int(max(float(seconds), 0.0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1).unsqueeze(1)
    return torch.clamp(xx + yy - 2.0 * torch.matmul(x, y.transpose(1, 2)), min=0.0)


def batched_index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, N, C = points.shape
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(points, 1, idx_exp)


def fps_deterministic(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    B, N, _ = points.shape
    out = torch.zeros(B, num_samples, dtype=torch.long, device=points.device)
    for b in range(B):
        pts = points[b]
        chosen = [0]
        dist = torch.full((N,), float("inf"), device=points.device)
        for _ in range(1, num_samples):
            last = pts[chosen[-1]]
            dist = torch.minimum(dist, ((pts - last) ** 2).sum(dim=1))
            chosen.append(int(dist.argmax().item()))
        out[b] = torch.tensor(chosen, device=points.device, dtype=torch.long)
    return out


def point_to_node_partition(points: torch.Tensor, num_patches: int, patch_size: int) -> Dict[str, torch.Tensor]:
    """Build patches around FPS nodes."""
    B, N, _ = points.shape
    P = min(int(num_patches), N)
    node_idx = fps_deterministic(points, P)
    nodes = batched_index_points(points, node_idx)
    dist_np = pairwise_sqdist(points, nodes)
    point_node = dist_np.argmin(dim=-1)
    node_masks = torch.zeros(B, P, dtype=torch.bool, device=points.device)
    node_masks.scatter_(1, point_node, True)
    owned = F.one_hot(point_node, num_classes=P).permute(0, 2, 1).bool()  # B, P, N
    dist_pn = dist_np.permute(0, 2, 1)  # B, P, N
    masked_dist = dist_pn.masked_fill(~owned, 1e9)
    k_eff = min(int(patch_size), N)
    patch_idx = (-masked_dist).topk(k_eff, dim=-1)[1]
    patch_vals = masked_dist.gather(-1, patch_idx)
    patch_masks = patch_vals < 1e9
    if k_eff < patch_size:
        pad = patch_size - k_eff
        patch_idx = torch.cat([patch_idx, node_idx[:, :, None].expand(B, P, pad)], dim=-1)
        patch_masks = torch.cat(
            [patch_masks, torch.zeros(B, P, pad, dtype=torch.bool, device=points.device)], dim=-1
        )
    center_fill = node_idx[:, :, None].expand_as(patch_idx)
    patch_idx = torch.where(patch_masks, patch_idx, center_fill)
    empty = ~node_masks.any(dim=1)
    if empty.any():
        patch_masks = patch_masks.clone()
        patch_idx = patch_idx.clone()
        patch_idx[empty] = center_fill[empty]
        patch_masks[empty] = False
        patch_masks[empty, 0] = True
    return {
        "nodes": nodes,
        "node_idx": node_idx,
        "point_node": point_node,
        "node_masks": node_masks,
        "patch_idx": patch_idx,
        "patch_masks": patch_masks,
    }


def compute_gt_patch_index_overlap(
    src_patch_idx: torch.Tensor,
    tgt_patch_idx: torch.Tensor,
    src_patch_masks: torch.Tensor,
    tgt_patch_masks: torch.Tensor,
    src_node_masks: torch.Tensor,
    tgt_node_masks: torch.Tensor,
) -> torch.Tensor:
    B, Ps, Ks = src_patch_idx.shape
    Pt = tgt_patch_idx.shape[1]
    overlap = torch.zeros(B, Ps, Pt, device=src_patch_idx.device)
    for b in range(B):
        s_idx, t_idx = src_patch_idx[b], tgt_patch_idx[b]
        s_m, t_m = src_patch_masks[b], tgt_patch_masks[b]
        for i in range(Ps):
            if not src_node_masks[b, i]:
                continue
            si = s_idx[i][s_m[i]]
            for j in range(Pt):
                if not tgt_node_masks[b, j]:
                    continue
                tj = t_idx[j][t_m[j]]
                inter = len(set(si.tolist()) & set(tj.tolist()))
                if inter == 0:
                    continue
                overlap[b, i, j] = 0.5 * (inter / max(len(si), 1) + inter / max(len(tj), 1))
    return overlap


class GeoSelfCrossBlock(nn.Module):
    """Self- and cross-attention for patch features."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dk = dim // heads
        self.qkv_self = nn.Linear(dim, dim * 3)
        self.q_cross = nn.Linear(dim, dim)
        self.kv_cross = nn.Linear(dim, dim * 2)
        self.out_self = nn.Linear(dim, dim)
        self.out_cross = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.geo_mlp = nn.Sequential(nn.Linear(1, heads), nn.ReLU(inplace=True), nn.Linear(heads, heads))

    def _self_attn(self, x, centers, mask):
        B, P, C = x.shape
        qkv = self.qkv_self(self.norm1(x)).view(B, P, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        q, k_, v = qkv[0], qkv[1], qkv[2]
        logits = torch.matmul(q, k_.transpose(-1, -2)) / np.sqrt(self.dk)
        d = torch.sqrt(torch.clamp(pairwise_sqdist(centers, centers), min=1e-12))
        scale = d.detach().median(dim=-1, keepdim=True)[0].median(dim=1, keepdim=True)[0].clamp(min=1e-3)
        logits = logits + self.geo_mlp((d / scale).unsqueeze(-1)).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], -1e4)
        out = torch.matmul(torch.softmax(logits, dim=-1), v).transpose(1, 2).reshape(B, P, C)
        return x + self.out_self(out)

    def _cross_attn(self, query, key_value, kv_mask):
        B, P, C = query.shape
        Q = self.q_cross(self.norm2(query)).view(B, P, self.heads, self.dk).transpose(1, 2)
        KV = self.kv_cross(key_value).view(B, key_value.shape[1], 2, self.heads, self.dk).permute(2, 0, 3, 1, 4)
        K_, V = KV[0], KV[1]
        logits = torch.matmul(Q, K_.transpose(-1, -2)) / np.sqrt(self.dk)
        logits = logits.masked_fill(~kv_mask[:, None, None, :], -1e4)
        out = torch.matmul(torch.softmax(logits, dim=-1), V).transpose(1, 2).reshape(B, P, C)
        return query + self.out_cross(out)

    def forward(self, src, tgt, src_centers, tgt_centers, src_mask, tgt_mask):
        src = self._self_attn(src, src_centers, src_mask)
        tgt = self._self_attn(tgt, tgt_centers, tgt_mask)
        src2 = self._cross_attn(src, tgt, tgt_mask)
        tgt2 = self._cross_attn(tgt, src, src_mask)
        src = src2 + self.ffn(self.norm3(src2))
        tgt = tgt2 + self.ffn(self.norm3(tgt2))
        return src * src_mask[..., None].float(), tgt * tgt_mask[..., None].float()


class GeoPatchMatcher(nn.Module):
    """Match source and target patches."""

    def __init__(self, point_feat_dim: int = 64, token_dim: int = 128, num_blocks: int = 1):
        super().__init__()
        self.center_feat_proj = nn.Sequential(nn.Linear(point_feat_dim, token_dim), nn.LayerNorm(token_dim))
        self.center_mlp = nn.Sequential(nn.Linear(3, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.blocks = nn.ModuleList([GeoSelfCrossBlock(token_dim, heads=4) for _ in range(num_blocks)])
        self.score_scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, source, target, src_feat, tgt_feat, num_patches, patch_size):
        src_part = point_to_node_partition(source, num_patches, patch_size)
        tgt_part = point_to_node_partition(target, num_patches, patch_size)
        src_tok = self.center_feat_proj(batched_index_points(src_feat, src_part["node_idx"]))
        tgt_tok = self.center_feat_proj(batched_index_points(tgt_feat, tgt_part["node_idx"]))
        src_tok = src_tok + 0.1 * self.center_mlp(src_part["nodes"])
        tgt_tok = tgt_tok + 0.1 * self.center_mlp(tgt_part["nodes"])
        src_tok = F.normalize(src_tok, dim=-1) * src_part["node_masks"][..., None].float()
        tgt_tok = F.normalize(tgt_tok, dim=-1) * tgt_part["node_masks"][..., None].float()
        for block in self.blocks:
            src_tok, tgt_tok = block(
                src_tok,
                tgt_tok,
                src_part["nodes"],
                tgt_part["nodes"],
                src_part["node_masks"],
                tgt_part["node_masks"],
            )
        src_tok = F.normalize(src_tok, dim=-1)
        tgt_tok = F.normalize(tgt_tok, dim=-1)
        scores = torch.matmul(src_tok, tgt_tok.transpose(1, 2)) * self.score_scale.clamp(1.0, 30.0)
        scores = scores.masked_fill(~tgt_part["node_masks"][:, None, :], -1e4)
        scores = scores.masked_fill(~src_part["node_masks"][:, :, None], -1e4)
        return {
            "scores": scores,
            "src_tokens": src_tok,
            "tgt_tokens": tgt_tok,
            "src_part": src_part,
            "tgt_part": tgt_part,
        }


def multipositive_infonce_loss(scores, overlap, src_mask, tgt_mask, pos_overlap=0.10):
    valid_pair = src_mask[:, :, None] & tgt_mask[:, None, :]
    ov = overlap.masked_fill(~valid_pair, 0.0)
    pos = ov >= pos_overlap
    max_ov, max_idx = ov.max(dim=-1)
    no_pos = (~pos.any(dim=-1)) & src_mask & (max_ov > 0)
    if no_pos.any():
        fallback = torch.zeros_like(pos)
        fallback.scatter_(-1, max_idx.unsqueeze(-1), True)
        pos = pos | (fallback & no_pos[..., None])
    row_valid = pos.any(dim=-1)
    s_all = scores.masked_fill(~valid_pair, -1e4)
    s_pos = scores.masked_fill(~pos, -1e4)
    row_loss = torch.logsumexp(s_all, dim=-1) - torch.logsumexp(s_pos, dim=-1)
    loss = row_loss[row_valid].mean() if row_valid.any() else scores.sum() * 0.0
    col_valid = pos.any(dim=-2)
    col_loss = torch.logsumexp(s_all, dim=-2) - torch.logsumexp(s_pos, dim=-2)
    if col_valid.any():
        loss = 0.5 * loss + 0.5 * col_loss[col_valid].mean()
    return loss


def best_overlap_ce_loss(scores, overlap, src_mask, tgt_mask):
    valid = src_mask[:, :, None] & tgt_mask[:, None, :]
    ov = overlap.masked_fill(~valid, -1.0)
    row_tgt = ov.argmax(dim=-1)
    row_loss = F.cross_entropy(scores, row_tgt, reduction="none")
    row_loss = row_loss[src_mask].mean() if src_mask.any() else scores.sum() * 0.0
    col_tgt = ov.argmax(dim=-2)
    col_loss = F.cross_entropy(scores.transpose(1, 2), col_tgt, reduction="none")
    col_loss = col_loss[tgt_mask].mean() if tgt_mask.any() else scores.sum() * 0.0
    return 0.5 * row_loss + 0.5 * col_loss


def compute_patch_loss(out: Dict[str, torch.Tensor], lambda_nce: float, lambda_best_ce: float, pos_overlap: float):
    scores = out["scores"]
    overlap = out["overlap"]
    src_mask = out["src_part"]["node_masks"]
    tgt_mask = out["tgt_part"]["node_masks"]
    nce = multipositive_infonce_loss(scores, overlap, src_mask, tgt_mask, pos_overlap)
    best_ce = best_overlap_ce_loss(scores, overlap, src_mask, tgt_mask)
    loss = lambda_nce * nce + lambda_best_ce * best_ce
    return loss, {"nce": float(nce.item()), "best_ce": float(best_ce.item())}


class GeoPatchDefTransNet(nn.Module):
    """DefTransNet with GeoPatch."""

    def __init__(self, token_dim: int = 128, num_geo_blocks: int = 1, show: int = 0, refine_alpha: float = 0.10):
        super().__init__()
        self.refine_alpha = refine_alpha
        self.feature_net = DefTransNet(show=show)
        self.matcher = GeoPatchMatcher(point_feat_dim=64, token_dim=token_dim, num_blocks=num_geo_blocks)
        self.point_context_proj = nn.Sequential(nn.Linear(token_dim, 64), nn.LayerNorm(64))

    def extract_features(self, source: torch.Tensor, target: torch.Tensor):
        return self.feature_net(source, target, k)

    def patch_forward(self, source, target, disp_gt, num_patches, patch_size):
        src_feat, tgt_feat = self.extract_features(source, target)
        out = self.matcher(source, target, src_feat, tgt_feat, num_patches, patch_size)
        out["src_feat"] = src_feat
        out["tgt_feat"] = tgt_feat
        out["overlap"] = compute_gt_patch_index_overlap(
            out["src_part"]["patch_idx"],
            out["tgt_part"]["patch_idx"],
            out["src_part"]["patch_masks"],
            out["tgt_part"]["patch_masks"],
            out["src_part"]["node_masks"],
            out["tgt_part"]["node_masks"],
        )
        return out

    def refine_point_features(self, out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map patch features back to points."""
        src_ctx = batched_index_points(out["src_tokens"], out["src_part"]["point_node"])
        tgt_ctx = batched_index_points(out["tgt_tokens"], out["tgt_part"]["point_node"])
        a = self.refine_alpha
        return (
            out["src_feat"] + a * self.point_context_proj(src_ctx),
            out["tgt_feat"] + a * self.point_context_proj(tgt_ctx),
        )


def build_dataloaders(data_root: Path, training_seed: int, evaluation_seed: int = 42):
    """Create training and validation loaders."""
    train_ds = PointCloudData_ModelNet(
        data_root,
        transform=train_transforms,
        typ=BENCHMARK_TRAIN_TYPES,
        rotation=rotation,
        folder="train",
        seed=training_seed,
    )
    valid_ds = PointCloudData_ModelNet(
        data_root,
        transform=train_transforms,
        typ=BENCHMARK_TRAIN_TYPES,
        rotation=rotation,
        folder="test",
        # same validation data for all runs
        seed=evaluation_seed,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(training_seed)
    valid_generator = torch.Generator()
    valid_generator.manual_seed(evaluation_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        generator=valid_generator,
    )
    return train_loader, valid_loader


def shuffle_target_like_deftransnet(target: torch.Tensor) -> torch.Tensor:
    ind = np.arange(target.shape[1])
    np.random.shuffle(ind)
    return target[:, ind, :]


def train_joint(model, train_loader, val_loader, args):
    """Train point and patch losses together."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and args.amp)
    writer = SummaryWriter(os.path.join(args.save_dir, args.run_name, "runs")) if args.run_name else None
    save_root = os.path.join(args.save_dir, args.run_name, "Saves")
    os.makedirs(save_root, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = []
        bt = None
        for i, data in enumerate(train_loader):
            t0 = time.time()
            source = data["source_pointcloud"].to(device).float()
            target = data["target_pointcloud"].to(device).float()
            disp_gt = data["disp"].to(device).float()
            target = shuffle_target_like_deftransnet(target)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and args.amp):
                out = model.patch_forward(source, target, disp_gt, args.num_patches, args.patch_size)
                patch_loss, pinfo = compute_patch_loss(
                    out, args.nce_weight, args.best_ce_weight, args.pos_overlap,
                )
                src_ref, tgt_ref = model.refine_point_features(out)
                disp_pred = inference(source, target, src_ref, tgt_ref, f=1)
                point_loss = netloss(disp_gt, disp_pred)
                loss = point_loss + args.lambda_patch * patch_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            dt = time.time() - t0
            bt = dt if bt is None else 0.9 * bt + 0.1 * dt
            running.append((loss.item(), point_loss.item(), patch_loss.item()))
            if (i + 1) % args.minibatches == 0:
                chunk = running[-args.minibatches :]
                print(
                    f"[Epoch {epoch+1} | Batch {i+1}/{len(train_loader)}] "
                    f"loss={np.mean([x[0] for x in chunk]):.4f} "
                    f"point={np.mean([x[1] for x in chunk]):.4f} "
                    f"patch={np.mean([x[2] for x in chunk]):.4f} "
                    f"nce={pinfo['nce']:.3f} bestCE={pinfo['best_ce']:.3f} "
                    f"ETA={_format_seconds(max(len(train_loader)-i-1,0)*(bt or 0))}"
                )
            if args.max_train_batches and (i + 1) >= args.max_train_batches:
                break

        ckpt = os.path.join(save_root, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved: {ckpt}")

        epoch_row = {
            "seed": int(args.seed),
            "epoch": int(epoch + 1),
            "mean_total_loss": float(np.mean([x[0] for x in running])) if running else float("nan"),
            "mean_point_loss": float(np.mean([x[1] for x in running])) if running else float("nan"),
            "mean_patch_loss": float(np.mean([x[2] for x in running])) if running else float("nan"),
            "checkpoint": ckpt,
        }
        history.append(epoch_row)

        if writer:
            avg = np.mean([x[0] for x in running]) if running else 0.0
            writer.add_scalar("train/loss", avg, epoch)
            writer.flush()

    history_path = os.path.join(args.save_dir, args.run_name, "training_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(history)
    if writer:
        writer.close()
    return history


def log_cuda_memory(dev: torch.device) -> None:
    if dev.type != "cuda":
        return
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        free, total = torch.cuda.mem_get_info(idx)
    except AttributeError:
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory
        free = total - torch.cuda.memory_reserved(idx)
    print(
        f"  GPU{idx} memory: {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB total "
        f"({torch.cuda.get_device_name(idx)})"
    )
    if free < 500e6:
        print(
            "  WARNING: < 0.5 GB free. Another job may be using this GPU.\n"
            "  Run `nvidia-smi` and kill stale python processes, or pass --device cuda:1"
        )


def prepare_device(device_str: str | None = None) -> torch.device:
    global device
    if device_str:
        device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        log_cuda_memory(device)
    return device


def parse_args():
    p = argparse.ArgumentParser(description="DefTransNet + GeoPatch Lepard-style joint training")
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_dir", default="Registration")
    p.add_argument("--run_name", default="gpcreg")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="First seed when --seeds is not supplied (default: 42)",
    )
    p.add_argument(
        "--num_seed_runs",
        type=int,
        default=3,
        help="Number of consecutive seeds starting from --seed (default: 3)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit training seeds; overrides --seed and --num_seed_runs",
    )
    p.add_argument(
        "--evaluation_seed",
        type=int,
        default=42,
        help="Fixed seed for validation metadata; keep 42 for the thesis protocol",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a seed when its final epoch checkpoint already exists",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--minibatches", type=int, default=20)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--token_dim", type=int, default=128)
    p.add_argument("--num_geo_blocks", type=int, default=1)
    p.add_argument("--num_patches", type=int, default=32)
    p.add_argument("--patch_size", type=int, default=24)
    p.add_argument("--refine_alpha", type=float, default=0.10, help="Lepard-style context residual strength")
    p.add_argument("--lambda_patch", type=float, default=0.1, help="Weight of coarse patch loss")
    p.add_argument("--nce_weight", type=float, default=1.0)
    p.add_argument("--best_ce_weight", type=float, default=1.0)
    p.add_argument("--pos_overlap", type=float, default=0.10)
    return p.parse_args()


def resolve_training_seeds(args):
    if args.seeds:
        seeds = [int(seed) for seed in args.seeds]
    else:
        if args.num_seed_runs < 1:
            raise SystemExit("--num_seed_runs must be at least 1")
        seeds = list(range(int(args.seed), int(args.seed) + int(args.num_seed_runs)))
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_one_seed(args, data_root: Path, seed: int):
    set_deterministic(seed)
    run_args = copy.deepcopy(args)
    run_args.seed = int(seed)
    base_run_name = args.run_name or "gpcreg"
    run_args.run_name = os.path.join(base_run_name, f"seed_{seed}")
    run_dir = os.path.join(args.save_dir, run_args.run_name)
    final_ckpt = os.path.join(run_dir, "Saves", f"epoch_{args.epochs}.pth")

    if args.skip_existing and os.path.exists(final_ckpt):
        print(f"\n=== Seed {seed}: skipped (checkpoint exists) ===")
        print(f"  {final_ckpt}")
        return {
            "seed": int(seed),
            "status": "skipped_existing",
            "run_name": run_args.run_name,
            "final_checkpoint": final_ckpt,
            "elapsed_seconds": 0.0,
            "final_total_loss": None,
            "final_point_loss": None,
            "final_patch_loss": None,
        }

    os.makedirs(run_dir, exist_ok=True)
    config = vars(run_args).copy()
    config.update(
        {
            "resolved_data_root": str(data_root.resolve()),
            "training_seed": int(seed),
            "fixed_evaluation_seed": int(args.evaluation_seed),
            "protocol_note": (
                "Training randomness varies by seed; validation/test metadata remains fixed. "
                "Select checkpoints on the fixed validation set, then evaluate once on the held-out test set."
            ),
        }
    )
    write_json(os.path.join(run_dir, "run_config.json"), config)

    print(f"\n=== Seed {seed}: {run_args.run_name} ===")
    train_loader, val_loader = build_dataloaders(
        data_root,
        training_seed=seed,
        evaluation_seed=args.evaluation_seed,
    )
    print(f"  train samples: {len(train_loader.dataset)}")
    print(f"  validation samples: {len(val_loader.dataset)}")

    model = GeoPatchDefTransNet(
        token_dim=args.token_dim,
        num_geo_blocks=args.num_geo_blocks,
        show=1 if args.show else 0,
        refine_alpha=args.refine_alpha,
    )
    try:
        model = model.to(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            log_cuda_memory(device)
            raise SystemExit(
                "CUDA OOM while loading model onto GPU.\n"
                "Run nvidia-smi, stop stale jobs, retry with --device cuda:1, "
                "or add --amp."
            ) from exc
        raise

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")

    started = time.time()
    history = train_joint(model, train_loader, val_loader, run_args)
    elapsed = time.time() - started
    final = history[-1]
    result = {
        "seed": int(seed),
        "status": "completed",
        "run_name": run_args.run_name,
        "final_checkpoint": final["checkpoint"],
        "elapsed_seconds": float(elapsed),
        "final_total_loss": final["mean_total_loss"],
        "final_point_loss": final["mean_point_loss"],
        "final_patch_loss": final["mean_patch_loss"],
    }
    write_json(os.path.join(run_dir, "run_result.json"), result)

    del model, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_multiseed_summary(args, seeds, results):
    base_run_name = args.run_name or "gpcreg"
    summary_dir = os.path.join(args.save_dir, base_run_name)
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = os.path.join(summary_dir, "multiseed_training_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(results)

    manifest = {
        "training_seeds": [int(seed) for seed in seeds],
        "fixed_evaluation_seed": int(args.evaluation_seed),
        "num_runs": len(seeds),
        "epochs_per_run": int(args.epochs),
        "checkpoint_selection": (
            f"Evaluate epochs 1-{int(args.epochs)} on the same 182 validation pairs; select the epoch "
            "with the lowest sample-weighted registered mean, breaking ties toward the earlier epoch."
        ),
        "reporting": (
            "Report held-out test metrics as mean +/- sample standard deviation across training seeds, "
            "overall and by deformation level. Do not replace seed variation with sample-level confidence intervals."
        ),
        "runs": results,
    }
    json_path = os.path.join(summary_dir, "multiseed_manifest.json")
    write_json(json_path, manifest)
    return csv_path, json_path


def main():
    global device
    args = parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    args.save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    seeds = resolve_training_seeds(args)

    if args.device:
        prepare_device(args.device)
    elif device.type == "cuda":
        prepare_device(str(device))

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")

    print("=== GeoPatch + DefTransNet (Lepard-style joint training) ===")
    print("  Data pipeline: train_transforms (1024 pts, Gaussian TPS)")
    print(f"  Deformation levels: {BENCHMARK_DEF_LEVELS}")
    print(f"  Rotation: {rotation} | Types: {BENCHMARK_TRAIN_TYPES}")
    print(f"  Patches: {args.num_patches} x {args.patch_size}")
    print(f"  Loss: point_L1 + {args.lambda_patch} * patch_loss")
    print(f"  Device: {device}")
    print(f"  Training seeds: {seeds}")
    print(f"  Fixed evaluation seed: {args.evaluation_seed}")

    results = [run_one_seed(args, data_root, seed) for seed in seeds]
    csv_path, json_path = save_multiseed_summary(args, seeds, results)
    print("\n=== Multi-seed training complete ===")
    print(f"  CSV summary: {csv_path}")
    print(f"  Manifest: {json_path}")


if __name__ == "__main__":
    main()
