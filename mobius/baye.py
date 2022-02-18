#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Bayesian optimization
#

from collections import defaultdict

import botorch
import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_model
from scipy.stats import norm

from .kernels import TanimotoSimilarityKernel


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel is not None:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoSimilarityKernel())

        # make sure we're on the right device/dtype
        self.to(train_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_fitted_model(train_x, train_y, state_dict=None, kernel=None):
    # initialize and fit model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)

    # Train model!
    fit_gpytorch_model(mll)

    return model


class AcqScoring:
    def __init__(self, gp_model, acq_function, seq_transformer, y_exp, greater_is_better=True):
        self._gp_model = gp_model
        self._acq_function = acq_function
        self._seq_transformer = seq_transformer
        self._y_exp = y_exp
        self.greater_is_better = greater_is_better

    def score(self, sequences):
        seq_transformed = self._seq_transformer.transform(sequences)
        return self._acq_function(self._gp_model, self._y_exp, seq_transformed, self.greater_is_better)


def predict(model, likelihood, test_x):
    # Set model in evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        return likelihood(model(test_x))


def greedy(model, Y_train, Xsamples, greater_is_better=False):
    """ greedy acquisition function

    Arguments:
    ----------
        model: Gaussian process model
        Y_train: Array that contains all the observed energy interaction seed so far
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.

    """
    observed_pred = predict(model, model.likelihood, Xsamples)
    mu = observed_pred.mean.detach().numpy()

    return mu


def expected_improvement(model, Y_train, Xsamples, greater_is_better=False, xi=0.00):
    """ expected_improvement
    Expected improvement acquisition function.
    
    Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
    
    Arguments:
    ----------
        model: Gaussian process model
        Y_train: Array that contains all the observed energy interaction seed so far
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.
        xi: Exploitation-exploration trade-off parameter

    """
    # calculate mean and stdev via surrogate function*
    observed_pred = predict(model, model.likelihood, Xsamples)
    sigma = observed_pred.variance.sqrt().detach().numpy()
    mu = observed_pred.mean.detach().numpy()

    if greater_is_better:
        loss_optimum = np.max(Y_train.numpy())
    else:
        loss_optimum = np.min(Y_train.numpy())

    scaling_factor = (-1) ** (not greater_is_better)

    # calculate the expected improvement
    Z = scaling_factor * (mu - loss_optimum - xi) / (sigma + 1E-9)
    ei = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + (sigma * norm.pdf(Z))
    ei[sigma == 0.0] == 0.0

    return -1 * ei


# probability of improvement acquisition function
def probability_of_improvement(model, Y_train, Xsamples, greater_is_better=False):
    """ probability_of_improvement
    Probability of improvement acquisition function.

    Arguments:
    ----------
        model: Gaussian process model
        Y_train: Array that contains all the observed energy interaction seed so far
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.

    """
    # calculate mean and stdev via surrogate function
    observed_pred = predict(model, model.likelihood, Xsamples)
    sigma = observed_pred.variance.sqrt().detach().numpy()
    mu = observed_pred.mean.detach().numpy()

    if greater_is_better:
        loss_optimum = np.max(Y_train.numpy())
    else:
        loss_optimum = np.min(Y_train.numpy())

    scaling_factor = (-1) ** (not greater_is_better)

    # calculate the probability of improvement
    Z = scaling_factor * (mu - loss_optimum) / (sigma + 1E-9)
    pi = norm.cdf(Z)
    pi[sigma == 0.0] == 0.0

    return pi


class DMTSimulation:

    def __init__(self, n_step=3, n_sample=10):
        self._n_step = n_step
        self._n_sample = n_sample

    def run(self, init_sequences, init_energies, **config):        
        data = []

        helmgo = config['helmgo']
        config.pop('helmgo')

        if config['acq_function'] in [greedy, expected_improvement]:
            greater_is_better = False
        else:
            greater_is_better = True

        # Since the input scores are affinity binding energies, we need to inverse the sign 
        # depending on the acquisition function
        # greater_is_better = True -> scaling_factor = -1
        # greater_is_better = False -> scaling_factor = 1
        scaling_factor = (-1) ** (greater_is_better)

        # Add initial data
        for sequence, energy in zip(init_sequences, init_energies):
            data.append((0, 0, energy, np.nan, sequence.count('.'), sequence))

        for i in range(self._n_sample):
            print('Run: %s' % (i + 1))

            # Defined GA optimization
            gao = config['GA'](helmgo, **config)

            # We want to keep a copy of the random peptides generated
            sequences = init_sequences.copy()
            energies = init_energies.copy()

            # Compute the MAP4 fingerprint for all the peptides
            X_exp = config['seq_transformer'].transform(sequences)
            y_exp = torch.from_numpy(energies).float()
            print('Exp dataset size: (%d, %d)' % (X_exp.shape[0], X_exp.shape[1]))

            print('\n')
            print('Init.')
            print('N pep: ', X_exp.shape[0])
            print('Best peptide: %.3f' % y_exp.min())
            for n in [-14, -13, -12, -11, -10, -9, -8]:
                print('N pep under %d kcal/mol: %03d' % (n, y_exp[y_exp < n].shape[0]))
            print('Non binding pep        : %03d' % (y_exp[y_exp == 0.].shape[0]))
            print('\n')

            for j in range(self._n_step):
                print('Generation: %d' % (j + 1))

                # Fit GP model
                gp_model = get_fitted_model(X_exp, y_exp * scaling_factor, kernel=config['kernel'])

                # Initialize acquisition function
                scoring_function = AcqScoring(gp_model, config['acq_function'], config['seq_transformer'], y_exp * scaling_factor, greater_is_better=greater_is_better)

                # Find new candidates using GA optimization
                gao.run(scoring_function, sequences, energies * scaling_factor)

                # Take N best candidates found
                candidate_sequences = gao.sequences[:config['n_candidates']]
                candidates_scores = gao.scores[:config['n_candidates']]

                clusters = defaultdict(list)
                for i_seq, sequence in enumerate(candidate_sequences):
                    clusters[sequence.count('.')].append(i_seq)
                print('Final selection:', ['%d: %d' % (k, len(v)) for k, v in clusters.items()])

                # Get affinitiy binding values (MAKE TEST)
                candidate_sequences_fasta = [''.join(c.split('$')[0].split('{')[1].split('}')[0].split('.')) for c in candidate_sequences]
                candidates_energies = config['oracle'].predict_energy(candidate_sequences_fasta)

                # Add candidates to the training set
                X_exp = torch.cat([X_exp, config['seq_transformer'].transform(candidate_sequences)])
                y_exp = torch.cat([y_exp, torch.from_numpy(candidates_energies)])

                sequences = np.append(sequences, candidate_sequences)
                energies = np.append(energies, candidates_energies)

                print('')
                print('N pep: ', X_exp.shape[0])
                print('Best peptide: %.3f' % y_exp.min())
                for n in [-14, -13, -12, -11, -10, -9, -8]:
                    print('N pep under %d kcal/mol: %03d' % (n, y_exp[y_exp < n].shape[0]))
                print('Non binding pep        : %03d' % (y_exp[y_exp == 0.].shape[0]))
                print('')

                # Store data
                for seq, score, energy in zip(candidate_sequences, candidates_scores, candidates_energies):
                    data.append((i + 1, j + 1, energy, score, seq.count('.'), seq))

        columns = ['sample', 'gen', 'exp_score', 'acq_score', 'length', 'sequence']
        df = pd.DataFrame(data=data, columns=columns)

        return df
