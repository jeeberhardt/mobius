#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Bayesian optimization
#

from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .acquisition_functions import AcqScoring
from .gaussian_process import get_fitted_model


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
            X_exp = torch.from_numpy(config['seq_transformer'].transform(sequences)).float()
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
                gao.run(scoring_function, sequences, y_exp.detach().numpy() * scaling_factor)

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
                sequences = np.append(sequences, candidate_sequences)
                X_exp = torch.cat([X_exp, torch.from_numpy(config['seq_transformer'].transform(candidate_sequences)).float()])
                y_exp = torch.cat([y_exp, torch.from_numpy(candidates_energies)])

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
