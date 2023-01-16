#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Emulator
#

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .utils import build_helm_string
from .utils import read_pssm_file
from .utils import convert_HELM_to_FASTA


class _Emulator(ABC):
    """Abstract class for defining an emulator"""

    @abstractmethod
    def predict(self):
        raise NotImplementedError()


class LinearPeptideEmulator(_Emulator):
    
    def __init__(self, pssm_files, score_cutoff=None):
        self._pssm = {}
        self._intercept = {}
        self._score_cutoff = score_cutoff

        # Read PSS Matrices
        for pssm_file in pssm_files:
            pssm, intercept = read_pssm_file(pssm_file)
            self._pssm[len(pssm.columns)] = pssm
            self._intercept[len(pssm.columns)] = intercept

    def predict(self, sequences, use_cutoff_score=True, input_type='helm'):
        assert input_type.lower() in ['fasta', 'helm'], 'Format (%s) not handled. Please use FASTA or HELM format.'

        # Score peptides using those PSSM
        scores = []

        if input_type == 'helm':
            sequences = convert_HELM_to_FASTA(sequences)

        for sequence in sequences:
            score = 0

            try:
                pssm = self._pssm[len(sequence)]
                intercept = self._intercept[len(sequence)]
            except:
                # We cannot score that peptide, so default score is 999
                score = 999
                scores.append(score)
                continue

            for i, aa in enumerate(sequence):
                score += pssm.loc[aa][i + 1]

            score += intercept
            scores.append(score)

        scores = np.array(scores)

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
            s = self.predict([p], use_cutoff_score=use_cutoff_score, input_type='FASTA')[0]

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
