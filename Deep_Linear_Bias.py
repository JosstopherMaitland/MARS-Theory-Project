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


class True_Model(nn.Module):
    def __init__(
        self,
        inp_dim : int,
        out_dim : int,
    ):
        super().__init__()
        self.combined_weights = nn.Parameter(torch.randn((inp_dim, out_dim)))
        self.combined_bias = nn.Parameter(torch.randn((out_dim)))

    def forward(self, X):
        return X @ self.combined_weights + self.combined_bias




class Layer(PyroModule):
    def __init__(
        self,
        shape: Tuple,
        prior_sd: float,
    ):
        super().__init__()
        self.weights = PyroSample(dist.Normal(torch.zeros(shape),
                                          prior_sd * torch.ones(shape)))
        self.bias = PyroSample(dist.Normal(torch.zeros((shape[-1])),
                                          prior_sd * torch.ones((shape[-1]))))
  
    def forward(self, X):
        return X @ self.weights + self.bias


class Bayes_Model(PyroModule):
    def __init__(
          self,
          dims : List, # dims[i] = width of layer i (includes input and output layers)
          prior_sd : float,
      ):

        super().__init__()
        self.layers = PyroModule[torch.nn.ModuleList](
            [Layer((dims[i], dims[i+1]), prior_sd) for i in range(len(dims) - 1)]
        )

        #self.combined_weights = False
        

    def forward(self, X, beta, Y=None):

        """if not self.combined_weights:
        self.combined_weights = self.weights[0]
        for i in range(1, len(self.weights)):
            self.combined_weights = self.combined_weights @ self.weights[i]""" # gotta be a speed up here

        for layer in self.layers:
            X = layer(X)

        return pyro.sample("obs", dist.Normal(X, 1/np.sqrt(beta)), obs=Y)




def combine_weights_bias(weights, biases):
    """
    Parameters:
    - weights (list[tensor]): weights[i] = matrix of weights between layer i and i+1.
    - biases (list[tensor]): biases[i] = bias for layer i+1.

    Returns:
    - tensor: the observations.
    """

    bias = biases[-1].detach().clone()

    n = len(weights)

    stacked_weights = weights[-1].detach().clone()
    for i in range(n-2, -1, -1):
        bias += biases[i] @ stacked_weights
        stacked_weights = weights[i] @ stacked_weights

    return stacked_weights, bias



def generate_inputs(args):
    inp_dim = args.model_hyperparams['dims'][0]
    return 2 * args.x_max * torch.rand(args.num_data, inp_dim) - args.x_max



def load_true_model(
        hyperparams,
        parameters,
    ):
    
    inp_dim = hyperparams['dims'][0]
    out_dim = hyperparams['dims'][-1]
    true_model = True_Model(inp_dim, out_dim)

    combined_params = combine_weights_bias(parameters['weights'], parameters['biases'])

    true_model.combined_weights.data = combined_params[0]
    true_model.combined_bias.data = combined_params[1]

    return true_model
