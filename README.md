# MARS-Theory-Project

## Program Structure
A file for each model architecture will contain:
- True model class named `True_Model`;
- Bayesian model class `Bayes_Model`;
- genereate_inputs function to generate inpute specific to the architecture;
- load_true_model function to instantiate a True_Model with the desired hyper and weight parameters;
- need to add functions for applying transformations to true parameters;

For reference, the hyperparameters for the different models take the following forms:
```
transformer_model_hyperparams = {
       'num_layers' : int,
          'd_vocab' : int,
          'd_model' : int,
            'd_mlp' : int,
           'd_head' : int,
        'num_heads' : int,
          'num_ctx' : int,
           'act_fn' : Callable, # can be set to None
    'use_pos_embed' : bool,
          'use_mlp' : bool,
}
```
```
deep_linear_hyperparams = {
     'dims' : List, # dims[i] = width of ith layer (includes input and output layers).
}
```
```
deep_linear_bias_hyperparams = deep_linear_hyperparams
```

A file for the Experiment class:
- it takes in a dictionary of the form:
    ```
    hyperparams = {
                      'model' : str, # the model to be used (one of ['transformer', 'deep_linear', 'deep_linear_bias'])
    'bayes_model_hyperparams' : Dict, # hyperparameters for the model used
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
    ```
    (Note: it may be that true_model_params is generated by some smaller set of hyperparameters)
    - the above key value pairs are converted into instance attributes (for example self.model = hyperparams['model'])
    - instantiates the true model self.true_model determined by self.true_model_hyperparams and true_model_params.
    - instantiates the bayesian model self.bayes_model determined by bayes_model_hyperparams.
    - Experiment has the methods:
        - get_dataset updates self.X to the output of generate_inputs for the given model in use, and updates self.Y to the output of the true model on self.X.
        - run_HMC_inference runs the inference on the bayesian model. To be completed.

