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


class Layer(PyroModule):
    def __init__(
        self,
        shape: Tuple,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(shape))
        self.bias = nn.Parameter(torch.randn(shape[-1]))
  
    def forward(self, X):
        return X @ self.weights + self.bias


class True_Model(PyroModule):
    def __init__(
          self,
          dims : List,
      ):

        super().__init__()
        self.layers = nn.ModuleList(
            [Layer((dims[i], dims[i+1])) for i in range(len(dims) - 1)]
        )

    def forward(self, X):

        for layer in self.layers:
            X = layer(X)

        return X




class Bayes_Layer(PyroModule):
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
            [Bayes_Layer((dims[i], dims[i+1]), prior_sd) for i in range(len(dims) - 1)]
        )
        

    def forward(self, X, beta, Y=None):

        for layer in self.layers:
            X = layer(X)

        return pyro.sample("obs", dist.Normal(X, 1/np.sqrt(beta)), obs=Y)



'''
No Longer Needed
def combine_weights_bias(weights, biases):
    """
    Parameters:
    - weights (list[tensor]): weights[i] = matrix of weights between layer i and i+1.
    - biases (list[tensor]): biases[i] = bias for layer i+1.

    Returns:
    - tuple[tensor]: combined weights and biases.
    """

    bias = biases[-1].detach().clone()

    n = len(weights)

    stacked_weights = weights[-1].detach().clone()
    for i in range(n-2, -1, -1):
        bias += biases[i] @ stacked_weights
        stacked_weights = weights[i] @ stacked_weights

    return stacked_weights, bias
'''


def generate_inputs(args):
    inp_dim = args.true_model_hyperparams['dims'][0]
    return 2 * args.x_max * torch.rand(args.num_data, inp_dim) - args.x_max



def load_true_model(
        hyperparams,
        parameters,
    ):

    dims = hyperparams['dims']
    true_model = True_Model(dims)

    #combined_params = combine_weights_bias(parameters['weights'], parameters['biases'])
    for i in range(len(dims)-1):
        true_model.layers[i].weights = nn.Parameter(parameters['weights'][i])
        true_model.layers[i].bias = nn.Parameter(parameters['biases'][i])

    return true_model
