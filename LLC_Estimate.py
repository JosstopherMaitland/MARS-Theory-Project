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

from devinterp.slt import sample, LLCEstimator
from devinterp.optim import SGLD

from typing import Any, Callable, Literal, Dict, List, Tuple, Optional, Union

import einops

import random




def llc_estimate(
        model,
        data_loader,
        criterion,
        sampling_method,
        optimizer_kwargs,
        num_chains,
        num_draws,
        num_burnin_steps,
        num_steps_bw_draws,
        device,
    ):

    llc_estimator = LLCEstimator(num_chains, num_draws, num_samples, device)
    sample(model = model,
           data_loader = data_loader,
           criterion = criterion,
           sampling_method = sampling_method,
           optimizer_kwargs = optimizer_kwargs,
           num_draws = num_draws,
           num_chains = num_chains,
           num_burnin_steps = num_burnin_steps,
           num_steps_bw_draws = num_steps_bw_draws,
           cores = 1,
           seed = None,
           device = torch.device(device),
           verbose = True,
           callbacks = [llc_estimator])

    llc_mean = llc_estimator.sample()["llc/mean"]

    return llc_mean






