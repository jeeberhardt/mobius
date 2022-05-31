#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Bayesian optimization
#

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance

from . import utils
from .acquisition_functions import AcqScoring, greedy, expected_improvement, random_improvement
from .gaussian_process import GPModel


class DMTSimulation:

    def __init__(self, n_step=3, n_sample=10):
        self._n_step = n_step
        self._n_sample = n_sample

    def run(self, init_sequences, init_energies, **config):        
        data = []

        if any(utils.function_equal(config['acq_function'], f) for f in [greedy, expected_improvement, random_improvement]):
            greater_is_better = False
        else:
            greater_is_better = True

        helmgo = config['helmgo']
        config.pop('helmgo')

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

            sequences = init_sequences.copy()

            # Compute the MAP4 fingerprint for all the peptides (if not using random_improvement acquisition function)
            if not utils.function_equal(config['acq_function'], random_improvement):
                X_exp = config['seq_transformer'].transform(init_sequences)
                y_exp = init_energies.copy()
            else:
                X_exp = np.array([])
                y_exp = np.array([])

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

                # Fit GP model (if not using random_improvement acquisition function)
                if not utils.function_equal(config['acq_function'], random_improvement):
                    gp_model = GPModel(kernel=config['kernel'])
                    gp_model.fit(X_exp, y_exp * scaling_factor)
                else:
                    gp_model = None

                # Initialize acquisition function
                scoring_function = AcqScoring(gp_model, config['acq_function'], y_exp * scaling_factor, config['seq_transformer'], greater_is_better=greater_is_better)

                # Find new candidates using GA optimization
                start = time.time()
                gao.run(scoring_function, sequences, y_exp * scaling_factor)
                print('Time elapsed: %.3f min.' % ((time.time() - start) / 60.))

                # Take N best candidates found
                candidate_sequences = gao.sequences[:config['n_candidates']]
                candidates_scores = gao.scores[:config['n_candidates']]

                clusters = defaultdict(list)
                for i_seq, sequence in enumerate(candidate_sequences):
                    clusters[sequence.count('.')].append(i_seq)
                print('Final selection:', ['%d: %d' % (k, len(v)) for k, v in clusters.items()])

                # Get affinitiy binding values (MAKE TEST)
                candidate_sequences_fasta = [''.join(c.split('$')[0].split('{')[1].split('}')[0].split('.')) for c in candidate_sequences]
                candidates_energies = config['oracle'].score(candidate_sequences_fasta)

                # Add candidates to the training set
                sequences = np.append(sequences, candidate_sequences)
                X_exp = np.concatenate([X_exp, config['seq_transformer'].transform(candidate_sequences)])
                y_exp = np.concatenate([y_exp, candidates_energies])

                print('')
                print('N pep: ', X_exp.shape[0])
                print('Best peptide: %.3f' % y_exp.min())
                for n in [-14, -13, -12, -11, -10, -9, -8]:
                    print('N pep under %d kcal/mol: %03d' % (n, y_exp[y_exp < n].shape[0]))
                print('Non binding pep        : %03d' % (y_exp[y_exp == 0.].shape[0]))
                try:
                    r = stats.pearsonr(candidates_scores[candidates_energies != 0], candidates_energies[candidates_energies != 0])[0]
                    print('Correlation acq-function/energies: %.3f' % r)
                except:
                    pass
                print('')

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(candidates_scores[candidates_energies != 0], candidates_energies[candidates_energies != 0])
                ax.set_xlabel('Acq function scores')
                ax.set_ylabel('Exp affinity binding')
                plt.show()

                # Store data
                for seq, score, energy in zip(candidate_sequences, candidates_scores, candidates_energies):
                    data.append((i + 1, j + 1, energy, score, seq.count('.'), seq))

        columns = ['sample', 'gen', 'exp_score', 'acq_score', 'length', 'sequence']
        df = pd.DataFrame(data=data, columns=columns)

        return df
