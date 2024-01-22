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

from copy import deepcopy

import unittest

import Deep_Linear, Deep_Linear_Bias, Transformer, Experiment_Class
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

class Generate_Deep_Linear_Args(): # and deep linear bias args
    def __init__(self):
        self.depth = random.randint(2, 100)
        self.dims = [random.randint(1,50) for i in range(self.depth)]

        self.weights = [1 - 2*torch.rand(self.dims[i], self.dims[i+1]) for i in range(self.depth-1)]
        self.biases = [1 - 2*torch.rand(self.dims[i+1]) for i in range(self.depth-1)]
        self.true_dl_model_params = {'weights' : self.weights}
        self.true_dlb_model_params = {'weights' : self.weights, 'biases' : self.biases}

        self.num_data = random.randint(1000,2000)
        self.X = torch.rand(self.num_data, self.dims[0])
        
        self.true_dl_model = Deep_Linear.True_Model(self.dims)
        self.true_dlb_model = Deep_Linear_Bias.True_Model(self.dims)
        
        self.prior_sd = random.random()
        self.bayes_dl_model = Deep_Linear.Bayes_Model(self.dims, self.prior_sd)
        self.bayes_dlb_model = Deep_Linear_Bias.Bayes_Model(self.dims, self.prior_sd)


class Generate_Transformer_Args():
    def __init__(self):
        self.use_pos = bool(random.randint(0,1))
        self.use_mlp = bool(random.randint(0,1))
        act_fn_opt = [F.relu, None]
        self.act_fn = act_fn_opt[random.randint(0,1)]

        self.d_vocab = random.randint(2,50)
        self.num_ctx = random.randint(2,50)
        
        self.hyperparams = {
            'num_layers' : random.randint(1,5),
            'd_vocab': self.d_vocab,
            'd_model': random.randint(100,200),
            'd_mlp': random.randint(200,400),
            'd_head': random.randint(50,100),
            'num_heads': random.randint(1,8),
            'num_ctx': self.num_ctx,
            'act_fn': self.act_fn,
            'use_pos_embed': self.use_pos,
            'use_mlp': self.use_mlp,
        }

        self.num_data = random.randint(1000,2000)
        self.X = [list(np.random.choice(range(self.d_vocab), size=self.num_ctx, replace=True)) for i in range(self.num_data)]

        self.true_model = Transformer.True_Model(**self.hyperparams)
        self.true_model_params = {name : param.data for name, param in self.true_model.named_parameters()}

        self.hyperparams_prior = deepcopy(self.hyperparams)
        self.prior_sd = random.random()
        self.hyperparams_prior['prior_sd'] = self.prior_sd
        self.bayes_model = Transformer.Bayes_Model(**self.hyperparams_prior)


class Test_Deep_Linear(unittest.TestCase):
    def setUp(self):
        self.args = Generate_Deep_Linear_Args()
    
    def test_true_model_shape(self): # also tests the function doesn't runtimeError
        args = self.args

        test_model = args.true_dl_model
        self.assertEqual(list(test_model(args.X).shape)[-1], args.dims[-1])

    def test_bayes_model_parameters(self):
        args = self.args
        
        test_model = args.bayes_dl_model
        i = random.randint(0,args.depth-2)
        self.assertEqual(len(test_model.layers[i].weights), args.dims[i])

    def test_bayes_model_parameters_random(self):
        args = self.args
        
        test_model = args.bayes_dl_model
        i = random.randint(0,args.depth-2)
        first_call = test_model.layers[i].weights
        second_call = test_model.layers[i].weights
        self.assertFalse(bool(torch.eq(first_call, second_call).all()))

    def test_load_true_model(self):
        args = self.args

        true_params = args.true_dl_model_params

        loaded_model = Deep_Linear.load_true_model({'dims' : args.dims}, true_params)

        X_new = args.X.detach().clone()
        for i in range(args.depth - 1):
            X_new = X_new @ true_params['weights'][i]

        self.assertTrue(bool(torch.eq(X_new, loaded_model(args.X)).all()))

        


class Test_Deep_Linear_Bias(unittest.TestCase):
    def setUp(self):
        self.args = Generate_Deep_Linear_Args()
    
    def test_true_model_shape(self):
        args = self.args
        
        self.assertEqual(list(args.true_dlb_model(args.X).shape)[-1], args.dims[-1])


    """
    def test_combine_weights_bias(self):
        args = self.args
        
        combined_weights, combined_bias = Deep_Linear_Bias.combine_weights_bias(args.weights, args.biases)
        X_new = torch.rand(args.dims[0])

        Y = X_new @ combined_weights + combined_bias
        
        for i in range(args.depth - 1):
            X_new = X_new @ args.weights[i] + args.biases[i]
        
        self.assertTrue(bool(torch.isclose(X_new, Y, atol=1e-06).all())) # weirdness, come back to.
        # how mush does caching the combined bias and weights speed up computation
    """

    def test_bayes_model_parameters(self):
        args = self.args
        
        i = random.randint(0, args.depth-2)
        self.assertEqual(len(args.bayes_dlb_model.layers[i].weights), args.dims[i])

    def test_bayes_model_parameters_random(self):
        args = self.args
        
        i = random.randint(0,args.depth-2)
        first_call = args.bayes_dlb_model.layers[i].weights
        second_call = args.bayes_dlb_model.layers[i].weights
        self.assertFalse(bool(torch.eq(first_call, second_call).all()))

    def test_load_true_model(self):
        args = self.args

        true_params = args.true_dlb_model_params

        loaded_model = Deep_Linear_Bias.load_true_model({'dims' : args.dims}, true_params)

        X_new = args.X.detach().clone()
        for i in range(args.depth - 1):
            X_new = X_new @ true_params['weights'][i] + true_params['biases'][i]

        self.assertTrue(bool(torch.eq(X_new, loaded_model(args.X)).all()))




class Test_Transformer(unittest.TestCase):

    def setUp(self):
        self.args = Generate_Transformer_Args()
        
    
    def test_true_model_runtime(self):
        self.args.true_model(self.args.X)

    def test_true_model_act_fn_mlp_parameters(self):
        args = self.args
        
        names = ''
        for name, param in args.true_model.named_parameters():
            names += name
            
        if args.use_mlp != bool('mlp' in names) or args.use_pos != bool('pos_embed' in names):
            self.assertFalse(True)
            

    def test_bayes_model_runtime(self):
        self.args.bayes_model(self.args.X, random.random())

    def test_bayes_model_parameters_random(self):
        args = self.args
        
        i = random.randint(0, args.hyperparams['num_layers'] - 1)
        first_call = args.bayes_model.blocks[i].attn.W_K
        second_call = args.bayes_model.blocks[i].attn.W_K
        self.assertFalse(bool(torch.eq(first_call, second_call).all()))

        first_call = args.bayes_model.embed.W_E
        second_call = args.bayes_model.embed.W_E
        self.assertFalse(bool(torch.eq(first_call, second_call).all()))

    def test_generate_inputs(self):
        pass

    def test_load_true_model(self):
        args = self.args

        true_params = args.true_model_params

        true_model = args.true_model

        loaded_model = Transformer.load_true_model(args.hyperparams, true_params)

        self.assertTrue(bool(torch.eq(true_model(args.X), loaded_model(args.X)).all()))
            


    

class Test_Experiment(unittest.TestCase):
    def setUp(self):
        self.trans_args = Generate_Transformer_Args()
        self.dl_args = Generate_Deep_Linear_Args()

        
        
        self.hyperparams_transformer = {
                              'model' : 'transformer',
            'bayes_model_hyperparams' : self.trans_args.hyperparams,
             'true_model_hyperparams' : self.trans_args.hyperparams,
                  'true_model_params' : self.trans_args.true_model_params,
                        'num_samples' : random.randint(10000, 20000),
                           'num_data' : random.randint(1,100),
                           'prior_sd' : random.random(),
                               'beta' : random.random(),
                         'num_warmup' : random.randint(0,1),
                              'x_max' : random.random(),
                     'exp_trial_code' : '11',
                   'raw_samples_path' : '',
                     'meta_data_path' : '',
                    'true_param_path' : '',
        }

        self.hyperparams_deep_linear = deepcopy(self.hyperparams_transformer)
        self.hyperparams_deep_linear['model'] = 'deep_linear'
        self.hyperparams_deep_linear['bayes_model_hyperparams'] = {'dims' : self.dl_args.dims}
        self.hyperparams_deep_linear['true_model_hyperparams'] = {'dims' : self.dl_args.dims}
        self.hyperparams_deep_linear['true_model_params'] = self.dl_args.true_dl_model_params

        self.hyperparams_deep_linear_bias = deepcopy(self.hyperparams_deep_linear)
        self.hyperparams_deep_linear_bias['model'] = 'deep_linear_bias'
        self.hyperparams_deep_linear_bias['true_model_params'] = self.dl_args.true_dlb_model_params
        

    
    def test_initialisation(self):
        exp_trans = Experiment_Class.Experiment(self.hyperparams_transformer)
        exp_dl = Experiment_Class.Experiment(self.hyperparams_deep_linear)
        exp_dlb = Experiment_Class.Experiment(self.hyperparams_deep_linear_bias)
        #exp_dlb = Experiment_Class.Experiment(self.hyperparams_deep_linear)
        

    def test_get_dataset(self):
        pass
        


if __name__ == "__main__":
    unittest.main()
    
