import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.linalg import multi_dot
import torch.multiprocessing

import contextlib, io

import pandas as pd

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.poutine as poutine
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import HMC, MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample

from typing import Any, Callable, Literal, Dict, List, Tuple, Optional, Union

import einops

import random

#### TRUE MODEL ####
class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx: int, d_model: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, num_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)

        attn_matrix = F.softmax(attn_scores_pre / np.sqrt(self.d_head), dim=-1)

        z = torch.einsum("biph,biqp->biqh", v, attn_matrix)
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = torch.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_fn: Callable):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_fn = act_fn

    def forward(self, x):
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        x = self.act_fn(x) if self.act_fn != None else x
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable,
        use_mlp: bool
    ):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, num_ctx)
        self.mlp = MLP(d_model, d_mlp, act_fn) if use_mlp else None

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x) if self.mlp != None else x
        return x


class True_Model(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_vocab: int,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable = F.relu,
        use_pos_embed: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(num_ctx, d_model) if use_pos_embed else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_mlp, d_head, num_heads, num_ctx, act_fn, use_mlp)
                for i in range(num_layers)
            ]
        )
        self.unembed = Unembed(d_vocab, d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x) if self.pos_embed != None else x

        for block in self.blocks:
            x = block(x)

        x = self.unembed(x)
        return x



#### BAYES MODEL ####

class Embed_Bayes(PyroModule):
    def __init__(self, d_vocab: int, d_model: int, prior_sd: float):
        super().__init__()
        self.W_E = PyroSample(dist.Normal(torch.zeros((d_model, d_vocab)),
                                          prior_sd * torch.ones((d_model, d_vocab))))

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed_Bayes(PyroModule):
    def __init__(self, d_vocab: int, d_model: int, prior_sd: float):
        super().__init__()
        self.W_U = PyroSample(dist.Normal(torch.zeros((d_model, d_vocab)),
                                          prior_sd * torch.ones((d_model, d_vocab))))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed_Bayes(PyroModule):
    def __init__(self, max_ctx: int, d_model: int, prior_sd: float):
        super().__init__()
        self.W_pos = PyroSample(dist.Normal(torch.zeros((max_ctx, d_model)),
                                            prior_sd * torch.ones((max_ctx, d_model))))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class Attention_Bayes(PyroModule):
    def __init__(self, d_model: int, num_heads: int, d_head: int, num_ctx: int, prior_sd: float):
        super().__init__()
        self.W_K = PyroSample(dist.Normal(torch.zeros((num_heads, d_head, d_model)),
                                          prior_sd * torch.ones((num_heads, d_head, d_model))))
        self.W_Q = PyroSample(dist.Normal(torch.zeros((num_heads, d_head, d_model)),
                                          prior_sd * torch.ones((num_heads, d_head, d_model))))
        self.W_V = PyroSample(dist.Normal(torch.zeros((num_heads, d_head, d_model)),
                                          prior_sd * torch.ones((num_heads, d_head, d_model))))
        self.W_O = PyroSample(dist.Normal(torch.zeros((d_model, d_head * num_heads)),
                                          prior_sd * torch.ones((d_model, d_head * num_heads))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)

        attn_matrix = F.softmax(attn_scores_pre / np.sqrt(self.d_head), dim=-1)

        z = torch.einsum("biph,biqp->biqh", v, attn_matrix)
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = torch.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


class MLP_Bayes(PyroModule):
    def __init__(self, d_model: int, d_mlp: int, act_fn: Callable, prior_sd: float):
        super().__init__()
        self.W_in = PyroSample(dist.Normal(torch.zeros((d_mlp, d_model)),
                                           prior_sd * torch.ones((d_mlp, d_model))))
        self.b_in = PyroSample(dist.Normal(torch.zeros((d_mlp)),
                                           prior_sd * torch.ones((d_mlp))))
        self.W_out = PyroSample(dist.Normal(torch.zeros((d_model, d_mlp)),
                                            prior_sd * torch.ones((d_model, d_mlp))))
        self.b_out = PyroSample(dist.Normal(torch.zeros((d_model)),
                                            prior_sd * torch.ones((d_model))))
        self.act_fn = act_fn

    def forward(self, x):
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        x = self.act_fn(x) if self.act_fn != None else x
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out

        return x


class TransformerBlock_Bayes(PyroModule):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable,
        use_mlp: bool,
        prior_sd: float,
    ):
        super().__init__()
        self.attn = Attention_Bayes(d_model, num_heads, d_head, num_ctx, prior_sd)
        self.mlp = MLP_Bayes(d_model, d_mlp, act_fn, prior_sd) if use_mlp else None
        self.use_mlp = use_mlp

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x) if self.use_mlp else x
        return x


class Bayes_Model(PyroModule):
    def __init__(
        self,
        num_layers: int,
        d_vocab: int,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable = F.relu, # can be set to None
        use_pos_embed: bool = True,
        use_mlp: bool = True,
        prior_sd: float = 1,
    ):

        super().__init__()
        self.embed = Embed_Bayes(d_vocab, d_model, prior_sd)
        self.pos_embed = PosEmbed_Bayes(num_ctx, d_model, prior_sd) if use_pos_embed else None
        self.blocks = PyroModule[torch.nn.ModuleList](
            [TransformerBlock_Bayes(d_model, d_mlp, d_head, num_heads, num_ctx, act_fn, use_mlp, prior_sd) for layer in range(num_layers)]
        )
        self.unembed = Unembed_Bayes(d_vocab, d_model, prior_sd)

        self.use_pos_embed = use_pos_embed

    def forward(self, x, beta, y=None):
        x = self.embed(x)
        x = self.pos_embed(x) if self.use_pos_embed else x

        for block in self.blocks:
            x = block(x)

        x = self.unembed(x)
        return pyro.sample("obs", dist.Normal(x, 1/np.sqrt(beta)), obs=y)





#### HELPER FUNCTIONS ####

def generate_inputs(args):
    d_vocab = args.model_hyperparams['d_vocab']
    num_ctx = args.model_hyperparams['num_ctx']
    return [random.sample(range(d_vocab), num_ctx) for i in range(args.num_data)]




def load_true_model(hyperparams, parameters):
    transformer = True_Model(**hyperparams)

    for name, param in transformer.named_parameters():
        param.data = parameters[name]

    return transformer
