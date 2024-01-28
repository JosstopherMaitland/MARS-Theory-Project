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


import Deep_Linear, Deep_Linear_Bias, Transformer
MODELS = {
         'transformer' : Transformer,
         'deep_linear' : Deep_Linear,
    'deep_linear_bias' : Deep_Linear_Bias,
}


"""
hyperparams = {
                      'model' : str, # the model to be used (one of ['transformer', 'deep_linear', 'deep_linear_bias'])
    'bayes_model_hyperparams' : Dict, # hyperparameters for the model used and some model architecture specific parameters for the experiment
     'true_model_hyperparams' : Dict, # hyperparameters for the true model used
          'true_model_params' : Dict,
                'num_samples' : int, # number of samples from the posterior
                   'num_data' : int, # number of data points
                   'prior_sd' : float, # standard deviation of model parameters
                       'beta' : float, # standard deviation of model output = 1/sqrt(beta)
                 'num_warmup' : int, # number of warmup iterations performed by MCMC. If 0, set to num_samples*0.05
                      'x_max' : float, # input data X is sampled from [-x_max, x_max]^2 (ignored for transformer models)
             'exp_trial_code' : str, # unique identifying code for the experiment
           'raw_samples_path' : str, # path to the folder storing raw_samples from experiment
             'meta_data_path' : str, # path to meta data folder
            'true_param_path' : str, # path to true parameter folder
}
"""

class Experiment():
    def __init__(
    self,
        hyperparams: Dict,
    ):
        # convert the dictionary hyper_params to instance attributes.
        for k, v in hyperparams.items():
            setattr(self, k, v)

        if not self.num_warmup:
            self.num_warmup = int(self.num_samples*0.05)

        self.X = None
        self.Y = None
        self.samples = None

        self.model = MODELS[self.model] # file name containing all model architecture specific functions
        self.true_model = self.model.load_true_model(self.true_model_hyperparams,
                                                     self.true_model_params)
        self.bayes_model = self.model.Bayes_Model(**self.bayes_model_hyperparams, prior_sd = self.prior_sd)

    def run_HMC_inference(self):
        """
        # Get dataset D_n, stored in self.X and self.Y
        self.get_dataset()

        # Get true param tensors (not needed anymore, should be stored in true_model_params)
        #true_params = #

        # Run HMC sampler
        self.samples = self.run_inference()

        # Convert samples into dataframe format
        #self.df = #

        # Save data
        csv_path = self.raw_samples_path + "/" + self.exp_trial_code + ".csv"
        self.df.to_csv(csv_path, index=True)

        Ln_w_average = self.df['Ln_w'].mean()

        # Write average Ln_w to CSV
        meta_data_CSV = pd.read_csv(self.meta_data_path, index_col=[0])
        meta_data_CSV.loc[meta_data_CSV['exp_code'] == self.exp_trial_code, 'Ln_w'] = Ln_w_average
        meta_data_CSV.to_csv(self.meta_data_path)

        # Save true parameters for plotting
        true_param_path = self.true_parameters + "/" + self.exp_trial_code + "_w0.csv"
        #self.true_params_df = #data_collector.samples_to_df(self, w0_true = True)
        self.true_params_df.to_csv(true_param_path)
        """
        pass

    def get_dataset(self):
        self.X = self.model.generate_inputs(self)
        out_mean = self.true_model(self.X)
        self.Y = dist.Normal(out_mean, 1).sample() # variance 1 for now (need to be adjusted to vector of same shape as mean?)

    def run_inference(self):
        kernel = NUTS(self.bayes_model.forward)
        mcmc = MCMC(kernel, num_samples=self.num_samples, warmup_steps=self.num_warmup)
        mcmc.run(X=self.X, Y=self.Y, beta=self.beta) # Don't know whether not passing prior_sd as arg would cause problem, think not at the moment
        
        # these statements can be deleted if not needed
        print("\n[beta = {}]".format(beta))
        # print(args.exp_trial_code) # do we need equivalent to Liam's version of exp_trial_code

        # return overview of diagnostics of samples, prob=width of credibility interval, can be adjusted
        mcmc.summary(prob=0.5)

        return mcmc.get_samples()

    def samples_to_df(self, w0_true=False):
		df_dict = {}
        if w0_true:
			data_dict = self.true_model_params
			unsqueeze = True
		else:
			data_dict = self.samples
			unsqueeze = False
			
		for key, tensor in data_dict.items():
			if unsqueeze:
				tensor = torch.unsqueeze(tensor, 0)
			tensor_np = tensor.numpy()
			n = len(tensor)
			for index, value in np.ndenumerate(tensor_np[0]):
				new_index = '.'.join([str(i) for i in index])
				dict_key = str(key) + '_' + new_index
				df_dict[dict_key] = [tensor[i][index] for i in range(n)]
				
		return pd.DataFrame(df_dict)
				
        

        

