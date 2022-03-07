#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Oracle
#

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .helm import build_helm_string


def read_pssm_file(pssm_file):
    data = []
    AA = []

    with open(pssm_file) as f:
        lines = f.readlines()
        
        n_columns = int(lines[0].split('\t')[1])

        for line in lines[1:-1]:
            sline = line.strip().split('\t')
            AA.append(sline[0])
            data.append([float(v) for v in sline[1:]])

    columns = list(range(1, n_columns + 1))
    pssm = pd.DataFrame(data=data, columns=columns, index=AA)

    return pssm


class Oracle:
    
    def __init__(self, pssm_files, exp_fasta_sequences=None, exp_values=None, energy_cutoff=None):
        self._pssm = {}
        self._reg = None
        self._energy_cutoff = energy_cutoff

        # Read PSS matrices
        for pssm_file in pssm_files:
            pssm = read_pssm_file(pssm_file)
            self._pssm[len(pssm.columns)] = pssm

        if exp_fasta_sequences is not None and exp_values is not None:
            assert len(exp_fasta_sequences) == len(exp_values), 'exp_fasta_sequences and exp_values must have the same size.'

            # Score peptides using those PSSM
            pssm_scores = self.score(exp_fasta_sequences, use_cutoff_energy=False)

            # Fit PSSM scores to experimental values
            reg = LinearRegression()
            reg.fit(pssm_scores[:, None], exp_values)
            print('----- Peptide global -----')
            print('N peptide: %d' % len(exp_fasta_sequences))
            print('R2: %.3f' % reg.score(pssm_scores[:, None], exp_values))
            print('RMSD : %.3f kcal/mol' % mean_squared_error(reg.predict(pssm_scores[:, None]), exp_values, squared=False))
            print('')
            self._reg = reg

    def score(self, fasta_sequences, use_cutoff_energy=True):
        # Score peptides using those PSSM
        scores = []

        for sequence in fasta_sequences:
            score = 0

            try:
                pssm = self._pssm[len(sequence)]
            except:
                # We cannot score that peptide, so default score is 999
                score = 999
                scores.append(score)
                continue

            for i, aa in enumerate(sequence):
                score += pssm.loc[aa][i + 1]

            scores.append(score)

        scores = np.array(scores)

        if self._reg is not None:
            scores = self._reg.predict(scores[:, None])

        if self._energy_cutoff is not None and use_cutoff_energy is True:
            scores[scores > self._energy_cutoff] = 0.

        return scores

    def generate_random_peptides(self, n_peptides, peptide_lengths, energy_bounds, use_cutoff_energy=True):
        random_peptides = []
        random_peptide_scores = []

        if not isinstance(peptide_lengths, (list, tuple)):
            peptide_lengths = [peptide_lengths]

        # We don't care about which pssm we are using here
        keys = list(self._pssm.keys())
        AA = self._pssm[keys[0]].index

        while True:
            peptide_length = np.random.choice(peptide_lengths)

            p = ''.join(np.random.choice(AA, peptide_length))

            s = self.score([p], use_cutoff_energy=use_cutoff_energy)[0]

            if energy_bounds[0] <= s <= energy_bounds[1]:
                helm_string = build_helm_string({'PEPTIDE1': p}, [])

                random_peptides.append(helm_string)
                random_peptide_scores.append(s)
                #print(len(random_peptides), helm_string)

            if len(random_peptides) == n_peptides:
                break

        sorted_index = np.argsort(random_peptide_scores)
        random_peptides = np.array(random_peptides)[sorted_index]
        random_peptide_scores = np.array(random_peptide_scores)[sorted_index]

        return random_peptides, random_peptide_scores
