#%load_ext autoreload
#%autoreload 2

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian as J
from torch.autograd import Function
import torch.utils.benchmark as benchmark
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
from path import Path
import glob
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import bootstrap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import copy
import pandas as pd
import random
from torch_tps import ThinPlateSpline
from models.deformation import SHIFT_SCALE, DEFAULT_ALPHA, DEFAULT_RESOLUTION, deform_gaussian

# ## Functions and Net

# settings

# data
cloud = Path("Registration/")
D, H, W = 100,100,100 #depth, height, width for discretization

#ModelNet (http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)
path = Path("ModelNet10")

# misc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# keypoints / graph
k = 10 #neighbours fixed
k1 = 128 #neighbours moving

# displacement space for dLBP
l_max = 9
l_width = l_max * 2 + 1
q = 3
disp = torch.stack(torch.meshgrid(torch.arange(- q * l_max, q * l_max + 1, q),
                                  torch.arange(- q * l_max, q * l_max + 1, q),
                                  torch.arange(- q * l_max, q * l_max + 1, q))).permute(1, 2, 3, 0).contiguous().view(1, -1, 3).float()
disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(device)

#sLBP
slbp_iter = 3
slbp_cost_scale = 10 #should not be needed due to newly introduced automatic scaler
slbp_alpha = -50 #regularization

#dLBP
dlbp_iter = 5
dlbp_cost_scale = 1
dlbp_alpha = -50#-15

#Loopy Belief Propagation-----------------------------------------------------------------------------------------------------------------------------------
#based on https://github.com/multimodallearning/deep-geo-reg

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist

def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device
    
    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    return ind, dist*A, A

def lbp_graph(kpts_fixed):
    A = knn_graph(kpts_fixed, k, include_self=False)[2][0]
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A).long()
    edges_idx[A.bool()] = torch.arange(edges.shape[0]).to(device)
    edges_reverse_idx = edges_idx.t()[A.bool()]
    return edges, edges_reverse_idx

def inference(kpts_fixed, kpts_moving,kpts_fixed_feat,kpts_moving_feat, f=1):
    N_p_fixed = kpts_fixed.shape[1]
    if f:    
        dist = pdist2(kpts_fixed_feat, kpts_moving_feat)
    else:
        dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = (kpts_fixed_feat.view(1, N_p_fixed, 1, -1) - kpts_moving_feat[:, ind.view(-1), :].view(1, N_p_fixed, k1, -1)).pow(2).mean(3)
    edges, edges_reverse_idx = lbp_graph(kpts_fixed)
    messages = torch.zeros((edges.shape[0], k1)).to(device)
    candidates_edges0 = candidates[0, edges[:, 0], :, :]
    candidates_edges1 = candidates[0, edges[:, 1], :, :]
    for _ in range(slbp_iter):
        temp_message = torch.zeros((N_p_fixed, k1)).to(device).scatter_add_(0, edges[:, 1].view(-1, 1).expand(-1, k1), messages)
        multi_data_cost = torch.gather(temp_message + candidates_cost.squeeze(), 0, edges[:,0].view(-1, 1).expand(-1, k1))
        reverse_messages = torch.gather(messages, 0, edges_reverse_idx.view(-1, 1).expand(-1, k1))
        multi_data_cost -= reverse_messages
        messages = torch.zeros_like(multi_data_cost)
        unroll_factor = 32
        split = torch.chunk(torch.arange(multi_data_cost.shape[0]), unroll_factor)
        for i in range(unroll_factor):
            messages[split[i]] = torch.min(multi_data_cost[split[i]].unsqueeze(1) + slbp_cost_scale*(candidates_edges0[split[i]].unsqueeze(1) - candidates_edges1[split[i]].unsqueeze(2)).pow(2).sum(3), 2)[0]
    reg_candidates_cost = (temp_message + candidates_cost.view(-1, k1)).unsqueeze(0)
    sm = F.softmax(slbp_alpha * reg_candidates_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)
    kpts_fixed_disp_pred = (candidates * sm).sum(2)
    return kpts_fixed_disp_pred

#For discrete LBP
def minconv(input, l_width):
    disp1d = torch.linspace(-1,1,l_width).to(input.device)
    regular1d = (disp1d.reshape(1,-1) - disp1d.reshape(-1,1)) ** 2
    
    output = torch.min( input.view(-1, l_width, 1, l_width, l_width) + regular1d.view(1, l_width, l_width, 1, 1), 1)[0]
    output = torch.min(output.view(-1, l_width, l_width, 1, l_width) + regular1d.view(1, 1, l_width, l_width, 1), 2)[0]
    output = torch.min(output.view(-1, l_width, l_width, l_width, 1) + regular1d.view(1, 1, 1, l_width, l_width), 3)[0]

    output = output - (torch.min(output.view(-1, l_width ** 3), 1)[0]).view(output.shape[0], 1, 1, 1)

    return output.view_as(input)

class InverseGridSample(Function):
    
    @staticmethod
    def forward(ctx, input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=None):
        B, C, N = input.shape
        D = grid.shape[-1]
        device = input.device
        dtype = input.dtype
        
        ctx.save_for_backward(input, grid)
        
        if D == 2:
            input_view = [B, C, -1, 1]
            grid_view = [B, -1, 1, 2]
        elif D == 3:
            input_view = [B, C, -1, 1, 1]
            grid_view = [B, -1, 1, 1, 3]
            
        ctx.grid_view = grid_view
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        with torch.enable_grad():
            output = J(lambda x: InverseGridSample.sample(input.view(*input_view), grid.view(*grid_view), x, mode, padding_mode, align_corners), (torch.zeros(B, C, *shape).to(dtype).to(device)))

        return output

    @staticmethod
    def backward(ctx, grad_output):        
        input, grid = ctx.saved_tensors
        grid_view = ctx.grid_view
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        
        grad_input = F.grid_sample(grad_output, grid.view(*grid_view), mode, padding_mode, align_corners)
        
        return grad_input.view(*input.shape), None, None, None, None, None
        
    @staticmethod
    def sample(input, grid, accu, mode='bilinear', padding_mode='zeros', align_corners=None):
        sampled = F.grid_sample(accu, grid, mode, padding_mode, align_corners)
        return -0.5 * ((input - sampled) ** 2).sum()
    
def inverse_grid_sample(input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=None):
    return InverseGridSample.apply(input, grid, shape, mode, padding_mode, align_corners)

def discretize(kpts_fixed, kpts_fixed_feat, kpts_moving, kpts_moving_feat, f=1):
    N_p_fixed = kpts_fixed.shape[1]
    disp_range = disp.max(1, keepdim=True)[0]
    if f:
        dist = pdist2(kpts_fixed_feat, kpts_moving_feat)
    else:
        dist = pdist2(kpts_fixed, kpts_moving)
    ind = (-dist).topk(k1, dim=-1)[1]
    candidates = - kpts_fixed.view(1, N_p_fixed, 1, 3) + kpts_moving[:, ind.view(-1), :].view(1, N_p_fixed, k1, 3)
    candidates_cost = (kpts_fixed_feat.view(1, N_p_fixed, 1, -1) - kpts_moving_feat[:, ind.view(-1), :].view(1, N_p_fixed, k1, -1)).pow(2).mean(3, keepdim=True)
    grid = inverse_grid_sample(candidates_cost.view(N_p_fixed, 1, -1), candidates[0]/disp_range, (l_width, l_width, l_width), mode='nearest', padding_mode='zeros', align_corners=True)
    grid_norm = inverse_grid_sample(torch.ones_like(candidates_cost.view(N_p_fixed, 1, -1)), candidates[0]/disp_range, (l_width, l_width, l_width), mode='nearest', padding_mode='zeros', align_corners=True)
    cost = grid  / (grid_norm + 0.000001)
    cost[cost==0] = 1e4
    return cost

#Wrapper smooth or discrete, new candidate search in feature space or old in 3D Euclidean space

def sLBP_GF(kpts_fixed, kpts_moving, net, f=1):
    # geometric features
    kpts_fixed_feat, kpts_moving_feat = net(kpts_fixed, kpts_moving, k)
    kpts_fixed_disp_pred = inference(kpts_fixed,kpts_moving,kpts_fixed_feat,kpts_moving_feat, f)
    return kpts_fixed_disp_pred

def sLBP_GF_old(kpts_fixed, kpts_moving, net):
    return sLBP_GF(kpts_fixed, kpts_moving, net, f=0)

def dLBP_GF(kpts_fixed, kpts_moving, net, f=1):
    N_p_fixed = kpts_fixed.shape[1]

    # geometric features
    kpts_fixed_feat, kpts_moving_feat = net(kpts_fixed, kpts_moving, k)

    # match
    cost = discretize(kpts_fixed, kpts_fixed_feat, kpts_moving, kpts_moving_feat, f)
    edges, _ = lbp_graph(kpts_fixed)
    messages = torch.zeros_like(cost)
    for _ in range(dlbp_iter):
        message_data = messages + cost
        reg_message_data = minconv(dlbp_cost_scale*message_data, l_width)/dlbp_cost_scale
        messages = torch.zeros_like(cost).view(N_p_fixed, -1).scatter_add_(0, edges[:, 0].view(-1, 1).expand(-1, l_width**3), reg_message_data[edges[:, 1]].view(-1, l_width**3)).view_as(cost)        
    reg_cost = messages + cost
    kpts_fixed_disp_pred = (disp.unsqueeze(1) * F.softmax(dlbp_alpha * reg_cost.view(1, N_p_fixed, -1), 2).unsqueeze(3)).sum(2)  
    return kpts_fixed_disp_pred


#NETWORK ARCHITECTURES-----------------------------------------------------------------------------------------------------------------------------------

class EdgeConv(nn.Module):
    #based on https://github.com/multimodallearning/deep-geo-reg
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x, ind):
        B, N, D = x.shape
        k = ind.shape[2]

        y = x.reshape(B*N, D)[ind.reshape(B*N, k)].reshape(B, N, k, D)
        x = x.reshape(B, N, 1, D).expand(B, N, k, D)
        
        x = torch.cat([y - x, x], dim=3)
        
        x = self.conv(x.permute(0, 3, 1, 2))
        x = F.max_pool2d(x, (1, k))
        x = x.squeeze(3).permute(0, 2, 1)
        
        return x

class Tnet(nn.Module):
   #based on https://gist.github.com/nikitakaraevv/d5047c9374c2fe6c9e6251886df00cdb
   def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)
    
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

   def forward(self, input):
        # input.shape ==  bs,3,n
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool_size = int(xb.size(-1))
        pool = F.max_pool1d(xb, pool_size).squeeze(-1)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        #initialize as identity
        init = torch.eye(
            self.k,
            device=xb.device,
            dtype=xb.dtype,
            requires_grad=True,
        ).repeat(bs, 1, 1)
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class TnetLayerNorm(Tnet):
    """T-Net variant used by the newly trained DefTransNet checkpoint."""

    def __init__(self, k=3):
        super().__init__(k=k)
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)
        
#Transformer --------------------------------------------------------------------------------
#based on http://nlp.seas.harvard.edu/annotated-transformer/ and Deep Closest Point https://github.com/WangYueFt/dcp

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
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__() #N=6, d_model=512, d_ff=2048, h=8, dropout=0.1 used in http://nlp.seas.harvard.edu/annotated-transformer/
        self.emb_dims = 64
        self.N = 1
        self.dropout = 0.0
        self.ff_dims = 1024
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding
    
#Feature Descriptor Network------------------------------------------------------------------------------------------------------------
    
class DefTransNet(nn.Module):
    """
    DefTransNet
    """
    def __init__(self, D=3, show = 0):
        super().__init__()
        self.input_transform = TnetLayerNorm(k=D)
        
        self.conv1 = EdgeConv(D, 32)
        self.conv2 = EdgeConv(32, 32)
        self.conv3 = EdgeConv(32, 64)

        self.conv4  = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False),
                                    nn.InstanceNorm1d(64),
                                    nn.Conv1d(64, 64, 1))
        
        self.transform = Transformer()

        self.show = show

    def forward(self,x,y,k):
        if self.show: 
            xpc = x.clone().detach().squeeze().cpu().numpy()
            ypc = y.clone().detach().squeeze().cpu().numpy()
            name = np.random.randint(0,1000000)

        #Apply T-Net
        matrix3x3x = self.input_transform(x.transpose(1,2)) 
        x = torch.bmm(x, matrix3x3x)

        matrix3x3y = self.input_transform(y.transpose(1,2)) 
        y = torch.bmm(y, matrix3x3y)

        if self.show: 
            pcshowTN(xpc, ypc, x.clone().detach().squeeze().cpu().numpy(), y.clone().detach().squeeze().cpu().numpy(), name)      

        #Apply EdgeConv
        fixed_ind = knn_graph(x, k, include_self=True)[0]
        x = self.conv1(x, fixed_ind)
        x = self.conv2(x, fixed_ind)
        x = self.conv3(x, fixed_ind)
        
        moving_ind = knn_graph(y, k*3, include_self=True)[0]
        y = self.conv1(y, moving_ind)
        y = self.conv2(y, moving_ind)
        y = self.conv3(y, moving_ind)

        #Apply Transformer
        xfp,yfp = self.transform(x.transpose(1,2),y.transpose(1,2))
        x = x + xfp.transpose(1,2)
        y = y + yfp.transpose(1,2)

        #Apply MLP
        x = self.conv4(x.permute(0,2,1)).permute(0,2,1)
        y = self.conv4(y.permute(0,2,1)).permute(0,2,1)

        if self.show: 
            ind = np.random.randint(0,len(xpc))
            xfd = torch.linalg.norm((x-x[0,ind]), dim = 2).detach().squeeze().cpu().numpy()
            yfd = torch.linalg.norm((y-y[0,ind]), dim = 2).detach().squeeze().cpu().numpy()
            pcshowTF(xpc, ypc, xfd, yfd,xpc[ind],ypc[ind], name)

        return x, y


class GFN(nn.Module):
    """
    Geometric Feature Network — DefTransNet without cross-point-cloud Transformer.
    Matches the GFN baseline in DefTransNet.ipynb (net2 in evaluate / plot vs DefTransNet).
    """

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

        x = self.conv4(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.conv4(y.permute(0, 2, 1)).permute(0, 2, 1)
        return x, y


#Loss function
def netloss(disp, disp_pred):
    criterion = torch.nn.L1Loss()
    return criterion(disp, disp_pred)

#ModelNet-----------------------------------------------------------------------------------------------------------------------------------
def read_off(file):
    #https://gist.github.com/nikitakaraevv/3b95c0f39448951c431761c054dbc3fc
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces
    
class PointSampler(object):
    #https://colab.research.google.com/github/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces, def_lvl, typ_nr, setting, rotation = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return (sampled_points, def_lvl, typ_nr, setting, rotation)

class Normalize_ModelNet(object):
    def __call__(self, inp):
        pointcloud, def_lvl, typ_nr, setting, rotation = inp
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  (norm_pointcloud, def_lvl, typ_nr, setting, rotation)

class TPS(object):
    """Perform Thin Plate Spline deformation"""
    def __init__(self, enabled=True, resolution=DEFAULT_RESOLUTION, alpha=DEFAULT_ALPHA):
        self.enabled = enabled
        self.tps = ThinPlateSpline(alpha)
        xs = torch.linspace(-1, 1, steps=resolution)
        ys = torch.linspace(-1, 1, steps=resolution)
        zs = torch.linspace(-1, 1, steps=resolution)
        x, y, z = torch.meshgrid(xs, ys, zs, indexing='xy')
        self.xyz = torch.stack([x, y, z], dim=3).reshape(-1, 3)
        
    def fit(self, pc, def_lvl):
        return deform_gaussian(self.xyz, pc, def_lvl, self.tps, seed=None)

    def __call__(self, inp): 
        source, def_lvl, typ_nr, setting, rotation = inp
        source = torch.from_numpy(source).float()
        if not self.enabled: return (source, source.clone(), typ_nr, setting, rotation)
        target = self.fit(source, def_lvl) 
        return (source, target, typ_nr, setting, rotation)

class RandRotation_z(object):
    def __init__(self, enabled=True):
        self.enabled = enabled
    def __call__(self, inp):
        if not self.enabled: return inp[0], inp[1]
        rotation = inp[2]
        return (inp[0], self.rotate(inp[1].float(), rotation))

    def rotate(self, pointcloud, rotation):
        assert len(pointcloud.shape)==2
        rot_matrix = _axis_angle_rotation('Z',torch.tensor(rotation))
        rot_pointcloud = torch.mm(rot_matrix.double(),pointcloud.double().T).T
        return  rot_pointcloud

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

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
   
class Modify(object):   
    """
    Add Incompleteness, Noise or Outliers to pointcloud.
    """
    def __call__(self, inp):
        source, target, typ_nr, setting, rotation = inp
        if typ_nr == 0:
            return source, target, rotation
        elif typ_nr == 1:
            return self.incomp(source,setting), target, rotation
        elif typ_nr == 2:
            return source, self.nois(target,setting), rotation
        elif typ_nr == 3:
            return source, self.out(target,setting), rotation
        
    def incomp(self, pointcloud, setting):
        dist = torch.norm(pointcloud - pointcloud[int(torch.rand((1))*len(pointcloud))], dim=1, p=None)
        knn = dist.topk(int(len(pointcloud)*(1-setting/100)))
        return pointcloud[knn.indices], knn.indices

    def nois(self, pointcloud, noise_lvl):
        assert len(pointcloud.shape)==2
        #noise = (torch.rand(pointcloud.shape)-0.5)*2*noise_lvl #change back for uniform instead of gaussian
        noisy_pointcloud = torch.normal(pointcloud,noise_lvl) #+ noise
        return  noisy_pointcloud

    def out(self, pointcloud, setting):
        dims = pointcloud.shape
        outliers = (torch.rand((int(dims[0]*setting/100), dims[1]))-0.5)*2
        return torch.cat((pointcloud, outliers), 0)


class ToTensor(object):
    def __call__(self, inp):
        source_pointcloud, target_pointcloud = inp[0], inp[1]
        assert len(source_pointcloud.shape)==2

        return torch.from_numpy(source_pointcloud), torch.from_numpy(target_pointcloud)#, inp[2], inp[3]
    
class PointCloudData_ModelNet(Dataset):
    def __init__(self, root_dir, transform, folder="train", typ = ["Deformation_Level"], rotation = 1/8, def_levels=None):
        """
        Class for ModelNet dataset.

        IN:
        root_dir : str
            ModelNet folder.
        transform : transform
            Transforms to apply to data
        folder : "train" or "test"
            Load train or test data
        typ : str "Deformation_Level" or "Incompleteness_Data" or "Noisy_Data" or "Outlier_Data"
            Different types of challenges to process
        rotation : float in [0,1]
            Percentage of complete rotation around z-axis
            
        OUT:
        Dataset class object
        """

        random.seed(42)
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        self.types_nr_dict= {"Deformation_Level":0,"Incompleteness_Data":1,"Noisy_Data":2,"Outlier_Data":3}
        self.settings =[[0],[0,5,10,15,20,25],[0,0.01,0.02,0.03,0.04],[0,5,15,25,35,45]]
        self.def_levels = [float(x) for x in def_levels] if def_levels else None
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):   
                    if len(typ)>1:
                        typ_nrs=[]
                        for t in typ:
                            typ_nrs.append(self.types_nr_dict[t])
                        choice = np.random.randint(0,len(typ_nrs))
                        typ_nr = typ_nrs[choice]
                        typ_name = typ[choice]
                    else:
                        typ_nr = self.types_nr_dict[typ[0]]
                        typ_name = typ[0]
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    sample['name'] = file
                    sample['def_lvl'] = random.choice(self.def_levels) if self.def_levels else random.randrange(1, 10, 1) / 10
                    sample['type'] = typ_name
                    sample['type_nr'] = typ_nr
                    sample['setting'] = self.settings[typ_nr][random.randrange(0,len(self.settings[typ_nr]),1)]
                    sample['rotation'] = np.around(random.random()*np.pi*2.*rotation,1)
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, def_lvl, typ_nr, setting, rotation):
        """
        This is a helper method that preprocesses a file. 
        It reads the vertices and faces from an OFF file using the read_off() function, and applies the provided transformations (self.transforms) to the data. 
        It returns the preprocessed source and target point clouds.
        """
        verts, faces = read_off(file)
        if self.transforms:
            source, target = self.transforms((verts, faces, def_lvl, typ_nr, setting, rotation))
        return source, target

    def __getitem__(self, idx):
        """
        This method is called to retrieve an item from the dataset at the given index.
        It retrieves the file path, category, name, deformation level, type, setting, and rotation for the specified index. 
        Then, it opens the point cloud file using open() and passes it to the __preproc__ method for preprocessing.
        Depending on the typ_nr value, it determines the valid indices and calculates the displacement (disp).
        Finally, it returns a dictionary containing the relevant data for the item.
        """
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        name = self.files[idx]['name']
        def_lvl = self.files[idx]['def_lvl']
        typ = self.files[idx]['type']
        typ_nr =  self.files[idx]['type_nr']
        setting = self.files[idx]['setting']
        rotation = self.files[idx]['rotation']
        with open(pcd_path, 'r') as f:
            source, target = self.__preproc__(f, def_lvl, typ_nr, setting, rotation)
        if typ_nr == 1:
            valid_ind=source[1]
            source = source[0]
            disp = target[valid_ind]-source
        else:
            disp = target[:source.shape[0]]-source 
            valid_ind=np.arange(0,len(source),1)
        return {'source_pointcloud': source, 
                'target_pointcloud': target,
                'disp': disp,
                'category': self.classes[category],
                'name': name,
                'deformation' : def_lvl,
                'type':typ,
                'setting':setting,
                'valid_ind' : valid_ind,
                'rotation' : rotation}

#Chamfer Distance-------------------------------------------------------------------------------------------------------------
def chamfer_distance_without_batch(p1, p2, debug=False):

    '''
    Calculate Chamfer Distance between two point sets
    https://gist.github.com/WangZixuan/4c4cdf49ce9989175e94524afc946726
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])

    p1 = p1.repeat(p2.size(1), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(0, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0])

    p2 = p2.repeat(p1.size(0), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=2)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=1)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    length = len(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist/length #Right?


#3D Plotting-----------------------------------------------------------------------------------------------------------------------------------
def visualize_rotate(data):
    #https://gist.github.com/nikitakaraevv/295f123c4f3cbecd734398eb9055fae1
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig



def pcshow(source, target, reg, err_source, err_target, err_reg):
    data=[go.Scatter3d(x=source[:,0], y=source[:,1], z=source[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_source)),
            go.Scatter3d(x=target[:,0],y=target[:,1], z=target[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_target)),
            go.Scatter3d(x=reg[:,0], y=reg[:,1], z=reg[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_reg)),
            go.Scatter3d(x=target[:,0],y=target[:,1], z=target[:,2],
                                   mode='markers',marker=dict(size=2,
                                    line=dict(width=2),
                                    color="lime", opacity=0.8))]
    fig = visualize_rotate(data)
    metric_figure = make_subplots(
        rows=1, cols=3, row_titles=["Mean distance color encoded"] ,subplot_titles=["Source", "Target", "Registered on target"],
        specs=[[{"type":"scene"}, {"type":"scene"}, {"type":"scene"}]])

    metric_figure.append_trace(fig.data[0], row=1, col=1)
    metric_figure.append_trace(fig.data[1], row=1, col=2)
    metric_figure.append_trace(fig.data[2], row=1, col=3)
    metric_figure.append_trace(fig.data[3], row=1, col=3)
    metric_figure.show()

def pcshow2(source, target, reg, reg2, err_source, err_target, err_reg, err_reg2, name,typ,setting, rotation, DefLvl):
    cmax = np.max(np.concatenate([err_source, err_target, err_reg, err_reg2]))
    cmid= cmax/2
    colorscale = "blackbody"
    data=[go.Scatter3d(x=source[:,0], y=source[:,1], z=source[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_source, showscale=True, colorbar = dict(title="Euclidean Distance [pu]", titleside = "right"), colorscale = colorscale, cmin = 0,cmid = cmid, cmax = cmax), showlegend=False),
            go.Scatter3d(x=target[:,0],y=target[:,1], z=target[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_target, colorscale = colorscale, cmin = 0,cmid = cmid, cmax = cmax), showlegend=False),
            go.Scatter3d(x=reg[:,0], y=reg[:,1], z=reg[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_reg, colorscale = colorscale, cmin = 0,cmid = cmid, cmax = cmax), showlegend=False),
            go.Scatter3d(x=reg2[:,0], y=reg2[:,1], z=reg2[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_reg2, colorscale = colorscale, cmin = 0,cmid = cmid, cmax = cmax), showlegend=False),
            go.Scatter3d(x=target[:,0],y=target[:,1], z=target[:,2],
                                   mode='markers',marker=dict(size=2,
                                    line=dict(width=2),
                                    color="lime", opacity=0.8), showlegend=False)]
    fig = visualize_rotate(data)

    metric_figure = make_subplots(
        rows=2, cols=2, horizontal_spacing=0, vertical_spacing=0.05, subplot_titles=["a) Source", "b) Target", "c) DefTransNet", "d) GFN"],
        specs=[[{"type":"scene"}, {"type":"scene"}],[{"type":"scene"}, {"type":"scene"}]])

    metric_figure.append_trace(fig.data[0], row=1, col=1)
    metric_figure.append_trace(fig.data[1], row=1, col=2)
    metric_figure.append_trace(fig.data[2], row=2, col=1)
    #metric_figure.append_trace(fig.data[4], row=2, col=1) #Overlay target in registration plot
    metric_figure.append_trace(fig.data[3], row=2, col=2)
    #metric_figure.append_trace(fig.data[4], row=2, col=2) #Overlay target in registration plot
    
    metric_figure.update_layout(margin = {"b":0,"t":20,"r":0,"l":0})
    folder = f"{cloud}/Plots/Comparison/{typ}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    metric_figure.write_image(f"{folder}/s{setting}_r{rotation}_d{DefLvl}_{name}.pdf", height=500, width=600)
    metric_figure.show()

def pcshowTN(source, target, reg, reg2, name):
    data=[go.Scatter3d(x=source[:,0], y=source[:,1], z=source[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2), color="#1f77b4"),
                                    showlegend=False),
            go.Scatter3d(x=target[:,0],y=target[:,1], z=target[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2), color="#ff7f0e"),
                                    showlegend=False),
            go.Scatter3d(x=reg[:,0], y=reg[:,1], z=reg[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2), color="#1f77b4"),
                                    showlegend=False),
            go.Scatter3d(x=reg2[:,0], y=reg2[:,1], z=reg2[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2), color="#ff7f0e"), 
                                    showlegend=False)]
    fig = visualize_rotate(data)

    metric_figure = make_subplots(
        rows=2, cols=2, horizontal_spacing=0, vertical_spacing=0.05, subplot_titles=["a) Source", "b) Target", "c) Source after T-Net", "d) Target after T-Net"],
        specs=[[{"type":"scene"}, {"type":"scene"}],[{"type":"scene"}, {"type":"scene"}]])

    metric_figure.append_trace(fig.data[0], row=1, col=1)
    metric_figure.append_trace(fig.data[1], row=1, col=2)
    metric_figure.append_trace(fig.data[2], row=2, col=1)
    metric_figure.append_trace(fig.data[3], row=2, col=2)
    
    metric_figure.update_layout(margin = {"b":0,"t":20,"r":0,"l":0})
    folder = f"{cloud}/Plots/T-Net/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    metric_figure.write_image(f"{folder}/{name}.pdf", height=500, width=600)
    metric_figure.show()

def pcshowTF(source, target, err_source, err_target, ps, pt, name):
    cmax = np.max(np.concatenate([err_source, err_target]))
    cmid= cmax/2
    colorscale = "blackbody"
    data=[go.Scatter3d(x=source[:,0], y=source[:,1], z=source[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_source, showscale=True, colorbar = dict(title="L2 norm", titleside = "right"), colorscale = colorscale, cmin = 0,cmid = cmid, cmax = cmax), showlegend=False),
            go.Scatter3d(x=target[:,0],y=target[:,1], z=target[:,2],
                                   mode='markers',marker=dict(size=3,
                                    line=dict(width=2),
                                    color=err_target, colorscale = colorscale, cmin = 0,cmid = cmid, cmax = cmax), showlegend=False),
            go.Scatter3d(x=[ps[0]], y=[ps[1]], z=[ps[2]],
                                   mode='markers',marker=dict(size=10,
                                    line=dict(width=2),
                                    color="lime"), showlegend=False),
            go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]],
                                   mode='markers',marker=dict(size=10,
                                    line=dict(width=2),
                                    color="lime"), showlegend=False)]
    fig = visualize_rotate(data)

    metric_figure = make_subplots(
        rows=1, cols=2, horizontal_spacing=0, vertical_spacing=0.05, subplot_titles=["a) Source", "b) Target"],
        specs=[[{"type":"scene"}, {"type":"scene"}]])

    metric_figure.append_trace(fig.data[0], row=1, col=1)
    metric_figure.append_trace(fig.data[1], row=1, col=2)
    metric_figure.append_trace(fig.data[2], row=1, col=1)
    metric_figure.append_trace(fig.data[3], row=1, col=2)
    
    metric_figure.update_layout(margin = {"b":0,"t":20,"r":0,"l":0})
    folder = f"{cloud}/Plots/Transformer"
    if not os.path.exists(folder):
        os.makedirs(folder)
    metric_figure.write_image(f"{folder}/{name}.pdf", height=250, width=600)
    metric_figure.show()    

# ## Initialize Data and Net

#ModelNet

train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize_ModelNet(),
                    TPS(),
                    Modify(),
                    RandRotation_z(enabled=False),
                    ])
rotation = 0
train_ds = PointCloudData_ModelNet(path, transform=train_transforms, typ=["Deformation_Level"], rotation=rotation)
valid_ds = PointCloudData_ModelNet(path, folder='test', transform=train_transforms, typ=["Deformation_Level"], rotation=rotation)

train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1)

#Network
net = DefTransNet(show=0)
#net.load_state_dict(torch.load(f'{cloud}Saves/DefTransNet/save_1.pth'))
net.to(device);
print(device)

# ## Training

def train(net=net, train_loader=train_loader, val_loader=valid_loader, predictor=sLBP_GF, epochs=10, save=None, load=0, minibatches=100):
    """
    IN:
    net : torch net
        Network to train
    train_loader : Dataloader
        Dataloader for training
    val_loader : Dataloader
        Dataloader for validation
    predictor : func
        Wrapper to handle Network predictions
    epochs : int
        Number of Epochs to train Network for
    save : bool
        Save runs using tensorboard and trained network .pth every epoch
    load : int
        To continue training. Put in, how many epochs the network has already been trained
    minibatches : int
        Number of samples after which to print and store loss
    """

    # Initialize optimizer and scaler
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    # Create a SummaryWriter for TensorBoard visualization
    if save:
        writer = SummaryWriter(f"{cloud}runs/{save}")

    # Loop over epochs
    for epoch in range(load, load + epochs):
        net.train()
        running_loss = 0.0

        # Iterate over mini-batches in the training DataLoader
        for i, data in enumerate(train_loader, 0):
            source, target, disp = (
                data["source_pointcloud"].to(device).float(),
                data["target_pointcloud"].to(device).float(),
                data["disp"].to(device),
            )

            # Reset optimizer gradients
            optimizer.zero_grad()

            # Shuffle target point clouds
            ind = np.arange(target.shape[1])
            np.random.shuffle(ind)
            target = target[:, ind, :]

            # Compute displacement prediction
            disp_pred = predictor(source, target, net)

            # Compute loss
            loss = netloss(disp, disp_pred)

            # Backpropagation and optimization step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate running loss
            running_loss += loss.item()

            if i % minibatches == minibatches - 1:
                avg_loss = running_loss / minibatches
                print(
                    '[Epoch: %d, Batch: %4d / %4d], loss: %.3f'
                    % (epoch + 1, i + 1, len(train_loader), avg_loss)
                )

                # Add training loss to TensorBoard
                if save:
                    writer.add_scalars(
                        "Training Loss",
                        {"Training": avg_loss},
                        epoch * len(train_loader) + i,
                    )

                running_loss = 0.0

        net.eval()

        # Save trained network and TensorBoard writer flush
        if save:
            if not os.path.exists(f"{cloud}Saves/{save}"):
                os.makedirs(f"{cloud}Saves/{save}")
            torch.save(net.state_dict(), f"{cloud}Saves/{save}/save_{str(epoch + 1)}.pth")
            writer.flush()

        # Perform validation if validation DataLoader is provided
        if val_loader:
            val_acc = 0
            chamf_acc = 0
            dhd_acc = 0

            with torch.no_grad():
                for data in val_loader:
                    source, target, disp = (
                        data["source_pointcloud"].to(device).float(),
                        data["target_pointcloud"].to(device).float(),
                        data["disp"].to(device),
                    )

                    # Compute displacement prediction
                    disp_pred = predictor(source, target, net)

                    # Compute registration and distance
                    reg = source + disp_pred
                    valid_ind = data['valid_ind'][0].to(device).long()
                    
                    #dist = reg - target

                    # Compute mean distance, Chamfer distance, and directed Hausdorff distance
                    val_acc += torch.mean(torch.linalg.norm((reg.squeeze() - target.squeeze()[valid_ind]), dim=1)).item()#torch.mean(torch.linalg.norm(dist.squeeze(), axis=1)).item()
                    chamf_acc += chamfer_distance_without_batch(reg, target)
                    dhd, _, _ = directed_hausdorff(reg.squeeze().cpu(), target.squeeze().cpu())
                    dhd_acc += dhd

            # Compute average validation metrics
            val_acc = val_acc / len(val_loader)
            chamf_acc = chamf_acc / len(val_loader)
            dhd_acc = dhd_acc / len(val_loader)

            # Print and add validation metrics to TensorBoard
            print(f"Mean dist: {val_acc}, Chamfer: {chamf_acc}, DHD: {dhd_acc}")

            if save:
                writer.add_scalars(
                    "Mean distance",
                    {"Valid": val_acc},
                    epoch * len(train_loader),
                )
                writer.add_scalars(
                    "Chamfer distance",
                    {"Valid": chamf_acc},
                    epoch * len(train_loader),
                )
                writer.add_scalars(
                    "DHD",
                    {"Valid": dhd_acc},
                    epoch * len(train_loader),
                )
                writer.flush()

# ## Evaluation

def evaluate(val_loader=valid_loader, predictor=sLBP_GF, save=None, skip=1, show=False, timer=False, net=net, net2=False, predictor2=sLBP_GF):
    """
    IN:
    val_loader : Dataloader
        Dataloader for valid data to evaluate
    predictor : func
        Predictor function which should handle network and predict disps
    save : string
        Folder to store evaluation in
    skip : int
        Skip data if whole evaluation would take too long, not really efficient. 1 evaluates everything, 2 every second and so on
    show : bool
        Enable plot of registration
    timer : bool
        Whether to time the predictor or not
    net : Network class 
        Network to evaluate
    net2 : Network class
        Second network, usually GFN as comparison for figures
    predictor2 : func
        Second predictor function for net2.

    Out:
        Dataframe of evaluation
    """

    if val_loader:
        results = {"Rotation" : [], "Type" : [], "Deformation Level" : [], "Setting" : [], "Name" : [], "Initial Mean" : [], "Initial Std":[], "Initial Max":[], "Initial Chamfer" : [], "Initial DHD" : [], "Registered Mean" : [], "Registered Std":[], "Registered Max":[], "Registered Chamfer" : [], "Registered DHD" : []}
        with torch.no_grad():
            s=time.time()
            if net2:
                counter=0
            for i,data in enumerate(val_loader):
                if i%skip == 0:
                    source, target, disp = data['source_pointcloud'].to(device).float(),data['target_pointcloud'].to(device).float(), data['disp'].squeeze()
                    typ = data['type'][0] 
                    DefLev = data['deformation'][0]
                    setting = data['setting'][0]
                    if not type(setting)==str:
                        setting = setting.item()
                    name = data['name'][0]
                    if setting: 
                        folder= f"Evaluation/{save}/{typ}/{setting}/{DefLev}"
                    else:
                        folder= f"Evaluation/{save}/{typ}/{DefLev}"
                    file = f"{folder}/{name}.xyz"
                    if not os.path.exists(file):
                        disp_pred = predictor(source, target, net)
                    else:
                        disp_pred = torch.from_numpy(np.loadtxt(file)).to(device)
                    if timer:
                        t0 = benchmark.Timer(stmt='predictor(source, target, net)',globals={'predictor': predictor, 'source' :source, 'target' :target, 'net':net})
                        print(t0.timeit(100))
                        #input()
                    if net2:
                        disp_pred2 = predictor2(source, target, net2)
                    try: 
                        rotation = data['rotation'][0].item()
                        print("Rotation:", rotation)
                        results["Rotation"].append(rotation)
                    except: 
                        print("No Rotation")
                        results["Rotation"].append(0)
                    
                    reg = source + disp_pred

                    valid_ind = data['valid_ind'][0].to(device).long()
                    err= torch.linalg.norm(disp, dim=1)
                    err_reg = torch.linalg.norm((reg.squeeze() - target.squeeze()[valid_ind]), dim=1)
                    err_mean=torch.mean(err).item()
                    err_std=torch.std(err).item()
                    err_max=torch.max(err).item()
                    err_reg_mean = torch.mean(err_reg).item()
                    err_reg_std=torch.std(err_reg).item()
                    err_reg_max = torch.max(err_reg).item()
                    err= err.cpu()
                    err_reg = err_reg.cpu()
                    err_source = err
                    err_target = np.zeros(target.shape[1])
                    err_target[valid_ind.cpu()]=err
                    chamf = chamfer_distance_without_batch(source,target[:,valid_ind]).item()
                    chamf_reg = chamfer_distance_without_batch(reg,target[:,valid_ind]).item()
                    target_cpu = target.squeeze()[valid_ind].cpu()
                    dhd,_,_ = directed_hausdorff(source.squeeze().cpu(),target_cpu)
                    dhd_reg,_,_ = directed_hausdorff(reg.squeeze().cpu(),target_cpu)
                    if net2:
                        reg2 = source + disp_pred2
                        err_reg2 = torch.linalg.norm((reg2.squeeze() - target.squeeze()[valid_ind]), dim=1).cpu()
                        err_reg_mean2=torch.mean(err_reg2).item()
                        err_reg_std2=torch.std(err_reg2).item()
                        if err_reg_mean <= err_reg_mean2: counter+=1

                    results["Type"].append(typ)
                    results["Deformation Level"].append(float(DefLev))
                    results["Setting"].append(setting)
                    results["Name"].append(name)
                    results["Initial Mean"].append(err_mean)
                    results["Initial Std"].append(err_std)
                    results["Initial Max"].append(err_max)
                    results["Initial Chamfer"].append(chamf)
                    results["Initial DHD"].append(dhd)
                    results["Registered Mean"].append(err_reg_mean)
                    results["Registered Std"].append(err_reg_std)
                    results["Registered Max"].append(err_reg_max)
                    results["Registered Chamfer"].append(chamf_reg)
                    results["Registered DHD"].append(dhd_reg)

                    print(f"Type: {typ} at {setting} \n Deformation: {DefLev} \n Initial: Mean: {err_mean:{1}.{5}} +/- {err_std:{1}.{5}} mm (Max: {err_max:{1}.{5}} mm), Chamfer: {chamf:{1}.{5}}, DHD: {dhd:{1}.{5}} \n Registered: Mean: {err_reg_mean:{1}.{5}} +/- {err_reg_std:{1}.{5}} mm (Max: {err_reg_max:{1}.{5}} mm), Chamfer: {chamf_reg:{1}.{5}}, DHD: {dhd_reg:{1}.{5}}")
                    if net2: print(f"Registered Net2 in d): Mean: {err_reg_mean2:{1}.{5}} +/- {err_reg_std2:{1}.{5}} mm")
                    
                    if save:
                        if setting: 
                            folder= f"Evaluation/{save}/{typ}/{setting}/{DefLev}"
                        else:
                            folder= f"Evaluation/{save}/{typ}/{DefLev}"
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        np.savetxt(f"{folder}/{name}.xyz",disp_pred.squeeze().cpu().numpy())
                    p = (i+1e-10)/len(val_loader)
                    t = (time.time()-s)*(1/p-1)
                    h = int(t/3600)
                    m = int(t%3600/60)
                    print(f"{p*100:{2}.{4}} %, Remaining: {h}:{m}h")
                    if show:
                        if net2:
                            pcshow2(source.squeeze().cpu(), target.squeeze().cpu(), reg.squeeze().cpu(), reg2.squeeze().cpu(), err_source, err_target, err_reg, err_reg2, name,typ,setting, rotation, DefLev)
                        else:
                            pcshow(source.squeeze().cpu(), target.squeeze().cpu(), reg.squeeze().cpu(), err_source, err_target, err_reg)
                        input()
                        #time.sleep(2)  
            if net2: print(f"Net 1 is in {counter/len(val_loader)*100} % of cases more accurate than net 2.")
            df=pd.DataFrame.from_dict(results)
            if save: 
                if not os.path.exists(f"{cloud}Evaluation/{save}"):
                    os.makedirs(f"{cloud}Evaluation/{save}")
                df.to_csv(f"{cloud}Evaluation/{save}/dataframe.csv", index=False)
            return df
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate DefTransNet")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    args = parser.parse_args()

    train(save="test", epochs=args.epochs)

    #ModelNet Evaluate all types for 3D visual evaluation
    _=evaluate(valid_loader, sLBP_GF, save=None, skip=15,show=1,timer=0)

    #ModelNet evaluate all types multiple times for statistical evaluation
    save = "sLBP_GFDefTransNet"
    df = pd.DataFrame()
    for i in range(20):
        valid_ds = PointCloudData_ModelNet(path, folder='test', transform=train_transforms, typ=["Deformation_Level"], rotation=rotation)
        valid_loader = DataLoader(dataset=valid_ds, batch_size=1)
        dfi=evaluate(valid_loader, sLBP_GF, save=None, skip=1,show=0,timer=0, net=net)
        df = pd.concat([df,dfi], ignore_index=True)
    if not os.path.exists(f"{cloud}Evaluation/{save}"):
            os.makedirs(f"{cloud}Evaluation/{save}")
    df.to_csv(f"{cloud}Evaluation/{save}/dataframeGaussFine.csv", index=False)

    #How to combine dataframes if needed
    save= "DefTransNet"
    df1= pd.read_csv(f"{cloud}/Evaluation/{save}/dataframe.csv")
    df2= pd.read_csv(f"{cloud}/Evaluation/{save}/dataframe2.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(f"{cloud}Evaluation/{save}/dataframeCombined.csv", index=False)

# ### Line plots

def plotevaltwo(df, data, ylabel, net_names, size=14, filter=None):
    y = ["Registered Mean", "Initial Mean"]
    if filter:
        title = f"{filter[0]} {filter[1]} {data} {y}"
    else:
        title = f"{data} {y}"

    fig = go.Figure()
    if data == 'Incompleteness_Data':
        legend_title = "Incompleteness in %"
        group='Setting'
    elif data == "Outlier_Data":
        legend_title = 'Outlier in %'
        group='Setting'
    elif data == "Deformation_Level":
        legend_title = 'Deformation Level'
        group='Deformation Level'
    elif data == "Rotation":
        legend_title = 'Rotation [rad]'
        group = data
        data = "Deformation_Level"
    else:
        legend_title = 'Noise Standard Deviation'
        group='Setting'
    dfd=df[0][df[0]['Type']==data]
    if filter:
        dfd = dfd[dfd[filter[0]] == filter[1]]
    dfd2 = df[1][df[1]['Type']==data]
    if filter:
        dfd2=dfd2[dfd2[filter[0]] == filter[1]]
    c=np.unique(dfd2[group].to_numpy())
    net0=[]
    net1=[]
    init=[]
    net0_std_l=[]
    net1_std_l=[]
    init_std_l=[]
    net0_std_r=[]
    net1_std_r=[]
    init_std_r=[]
    for i in c:      
        init_mean = dfd[dfd[group]==i]["Initial Mean"].to_numpy()
        i_l,i_r = bootstrap((init_mean,), np.mean, confidence_level=0.997, vectorized=False).confidence_interval
        init_std_l.append(i_l)
        init_std_r.append(i_r)
        init.append(np.mean(init_mean))
        net0_mean = dfd[dfd[group]==i]["Registered Mean"].to_numpy()
        n0_l,n0_r = bootstrap((net0_mean,), np.mean, confidence_level=0.997, vectorized=False).confidence_interval
        net0_std_l.append(n0_l)
        net0_std_r.append(n0_r)
        net0.append(np.mean(net0_mean))
        net1_mean = dfd2[dfd2[group]==i]["Registered Mean"].to_numpy()
        n1_l,n1_r = bootstrap((net1_mean,), np.mean, confidence_level=0.997, vectorized=False).confidence_interval
        net1_std_l.append(n1_l)
        net1_std_r.append(n1_r)
        net1.append(np.mean(net1_mean))
    offset=np.max(c)*0.005
    fig.update_xaxes(range=[np.min(c)-offset,np.max(c)+offset])
    max_y_std = np.max(init_std_r)
    max_y_mean = np.max(init)
    max_y = max_y_std + max_y_mean
    fig.update_yaxes(range=[0,max_y*(1.005)])
    fig.add_trace(go.Scatter(x=c, y=init,error_y=dict(array=init_std_r, arrayminus=init_std_l), cliponaxis=True,
                    mode='lines+markers',
                    name="Source"))
    fig.add_trace(go.Scatter(x=c, y=net1, error_y=dict(array=net1_std_r, arrayminus=net1_std_l), cliponaxis=True,
                    mode='lines+markers',
                    name=f"Registered {net_names[1]}"))
    fig.add_trace(go.Scatter(x=c, y=net0, error_y=dict(array=net0_std_r, arrayminus=net0_std_l), cliponaxis=True,
                    mode='lines+markers',
                    name=f"Registered {net_names[0]}"))
    fig.update_layout(yaxis_title=ylabel,
                        xaxis_title = legend_title,
                        legend=dict(
                                title="Distance target to",
                            ),
                        margin = {"b":0,"t":0,"r":0,"l":0},
                        font_size = size
                        )
    if not os.path.exists(f"{cloud}Plots/{net_names}"):
        os.makedirs(f"{cloud}Plots/{net_names}")
    fig.write_image(f"{cloud}/Plots/{net_names}/{title}.pdf")
    fig.show()

def plotloss(kind, yaxis):
    fig = go.Figure()
    if kind == "Mean": 
        fig.update_xaxes(range=[0, 36000])
        mode = 'lines+markers'
    else:
        mode = 'lines'
    for i,file in enumerate(glob.glob(f"{cloud}/RunsCSV/{kind}/*.csv")):
        name=Path(file).stem[1:]#"_".join(Path(file).stem.split(sep = "_")[1:-2])
        data = np.loadtxt(file, skiprows = 1, delimiter=",")
        if i == 0:
            j = data.shape[0]
        fig.add_trace(go.Scatter(x=data[:j,1], y=data[:j,2], cliponaxis=True,
                    mode=mode,
                    name=name))
    fig.update_layout(#title=title,
                    xaxis_title='Step',
                    yaxis_title=yaxis,
                    legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ),
                    margin = {"b":0,"t":0,"r":0,"l":0})
    fig.write_image(f"{cloud}/Plots/{kind}.pdf")
    fig.show()

if __name__ == "__main__":
    model = "DefTransNet"
    model2 = "GFN"
    df = pd.read_csv(f"{cloud}/Evaluation/{model}/dataframeGaussFineBig.csv")
    df2 = pd.read_csv(f"{cloud}/Evaluation/{model2}/dataframeGaussFineBig.csv")

    for s in ["Deformation_Level","Incompleteness_Data","Noisy_Data","Outlier_Data","Rotation"]: #["Rotation","Deformation_Level","Incompleteness_Data","Noisy_Data","Outlier_Data"]
        plotevaltwo([df,df2], s, "Mean Euclidean Distance [pu]", ["DefTransNet", "GFN"])#, filter=["Deformation Level", 0.1]) #, filter=["Rotation", 0])

    plotloss("Mean", 'Validation Loss (L2) [pu]')

    plotloss("Loss", 'Training Loss (L2) [pu]')
