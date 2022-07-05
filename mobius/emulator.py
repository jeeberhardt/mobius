#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Emulator
#

from abc import ABC, abstractmethod

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


class _Emulator(ABC):
    """Abstract class for defining an emulator"""

    @abstractmethod
    def predict(self):
        raise NotImplementedError()


class LinearPeptideEmulator(_Emulator):
    
    def __init__(self, pssm_files, exp_fasta_sequences=None, exp_values=None, score_cutoff=None):
        self._pssm = {}
        self._reg = None
        self._score_cutoff = score_cutoff

        # Read PSS matrices
        for pssm_file in pssm_files:
            pssm = read_pssm_file(pssm_file)
            self._pssm[len(pssm.columns)] = pssm

        if exp_fasta_sequences is not None and exp_values is not None:
            assert len(exp_fasta_sequences) == len(exp_values), 'exp_fasta_sequences and exp_values must have the same size.'

            # Score peptides using those PSSM
            pssm_scores = self.predict(exp_fasta_sequences, use_cutoff_score=False)

            # Fit PSSM scores to experimental values
            reg = LinearRegression()
            reg.fit(pssm_scores[:, None], exp_values)
            print('----- Peptide global -----')
            print('N peptide: %d' % len(exp_fasta_sequences))
            print('R2: %.3f' % reg.score(pssm_scores[:, None], exp_values))
            print('RMSD : %.3f kcal/mol' % mean_squared_error(reg.predict(pssm_scores[:, None]), exp_values, squared=False))
            print('')
            self._reg = reg

    def predict(self, fasta_sequences, use_cutoff_score=True):
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

        if self._score_cutoff is not None and use_cutoff_score is True:
            scores[scores > self._energy_score] = 0.

        return scores

    def generate_random_peptides(self, n_peptides, peptide_lengths, score_bounds=None, use_cutoff_score=True, 
                                 monomer_symbols=None, output_format='helm'):
        random_peptides = []
        random_peptide_scores = []

        assert output_format.lower() in ['fasta', 'helm'], 'Can only output peptide sequences in HELM or FASTA formats.'

        if monomer_symbols is None:
            # If we do not provide any monomer symbols, the 20 canonical amino acids will be used
            monomer_symbols = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
                               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        else:
            monomer_symbols = monomer_symbols

        if not isinstance(peptide_lengths, (list, tuple)):
            peptide_lengths = [peptide_lengths]

        while True:
            peptide_length = np.random.choice(peptide_lengths)
            p = ''.join(np.random.choice(monomer_symbols, peptide_length))
            s = self.predict([p], use_cutoff_score=use_cutoff_score)[0]

            if score_bounds is not None:
                if score_bounds[0] <= s <= score_bounds[1]:
                    random_peptides.append(p)
                    random_peptide_scores.append(s)
            else:
                random_peptides.append(p)
                random_peptide_scores.append(s)

            if len(random_peptides) == n_peptides:
                break

        if output_format.lower() == 'helm':
            random_peptides = [build_helm_string({'PEPTIDE1': p}, []) for p in random_peptides]

        sorted_index = np.argsort(random_peptide_scores)
        random_peptides = np.asarray(random_peptides)[sorted_index]
        random_peptide_scores = np.asarray(random_peptide_scores)[sorted_index]
        
        return random_peptides, random_peptide_scores
