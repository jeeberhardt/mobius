#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Emulator
#

from abc import ABC, abstractmethod

import numpy as np
import torch

from .fingerprints import Map4Fingerprint
from .kernels import TanimotoSimilarityKernel
from .utils import build_helm_string
from .utils import read_pssm_file
from .utils import convert_FASTA_to_HELM, convert_HELM_to_FASTA


class _Emulator(ABC):
    """Abstract class for defining an emulator"""

    @abstractmethod
    def predict(self):
        raise NotImplementedError()


class FindMe(_Emulator):
    """
    The FindMe `Emulator` where the goal is to find the target sequence provided
    at the initialization. This is for benchmarking purpose only.

    """

    def __init__(self, target_sequence, input_type='helm', kernel=None, input_transformer=None):
        """
        Initialize the FindMe `Emulator`.
        
        Parameters
        ----------
        target_sequence : str
            Sequence of the polymer to find.
        input_type : str, default : 'helm'
            Format of the target polymer, either FASTA or HELM.
        kernel : gpytorch.kernels.Kernel, default : None
            The kernel function used to calculate distance between polymers/peptides.
            If not defined, the Tanimoto kernel will be used.
        input_transformer : input transformer, default : None
            Function that transforms the input into data for the `kernel`.
            If not defined, the MAP4 fingerprint method will be used.

        
        Raises
        ------
        AssertionError: If output format is not 'fasta' or 'helm'.

        """
        if input_transformer != 'precomputed':
            msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
            assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

            if input_type == 'fasta':
                target_sequence = convert_FASTA_to_HELM(target_sequence)

        self._target_sequence = target_sequence

        if kernel is None:
            self._kernel = TanimotoSimilarityKernel()
        else:
            self._kernel = kernel

        if input_transformer is None:
            self._input_transformer = Map4Fingerprint()
        elif input_transformer == 'precomputed':
            self._input_transformer = 'precomputed'
        else:
            self._input_transformer = input_transformer

        if self._input_transformer != 'precomputed':
            target_sequence_transformed = self._input_transformer.transform(target_sequence)
        else:
            target_sequence_transformed = self._target_sequence

        self._target_sequence_transformed = torch.from_numpy(np.asarray(target_sequence_transformed)).float()

    def predict(self, sequences, input_type='helm'):
        """
        Score the input sequences according to the unknown target sequences 
        using the `kernel` and the `input_transformer`.
        
        Parameters
        ----------
        sequences : list of str
            Polymers/peptides to score.
        input_type : str, default : 'helm'
            Format of the input polymer sequences, either FASTA or HELM.

        Returns
        -------
        ndarray of shape (n_sequences,)
            Scores of the input polymers/peptides.

        Raises
        ------
        AssertionError: If output format is not 'fasta' or 'helm'.

        """
        # If the input sequences are precomputed, no need to check for the format
        if self._input_transformer != 'precomputed':
            msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
            assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

            if input_type == 'fasta':
                sequence = convert_FASTA_to_HELM(sequences)

        if not isinstance(sequences, (list, tuple, np.ndarray)):
            sequences = [sequences]

        if self._input_transformer != 'precomputed':
            sequences_transformed = self._input_transformer.transform(sequences)

        sequences_transformed = torch.from_numpy(np.asarray(sequences_transformed)).float()

        d = self._kernel.forward(self._target_sequence_transformed, sequences_transformed)[0]
        d = d.detach().numpy()

        return d


class LinearPeptideEmulator(_Emulator):
    """
    The LinearPeptideEmulator `Emulator` where the goal is to minimize or maximize 
    the score based on a defined Position Specific Scoring Matrix (PSSM).

    """
    
    def __init__(self, pssm_files, score_cutoff=None):
        """
        Initialize the LinearPeptideEmulator `Emulator`.
        
        Parameters
        ----------
        pssm_files : str
            Path of the PSSM file to read.
        score_cutoff : int or float, default : None
            If specified, scores superior than `score_cutoff` will be set to zero.
            It is one way to simulate a non-binding event.

        """
        self._pssm = {}
        self._intercept = {}
        self._score_cutoff = score_cutoff

        # Read PSS Matrices
        for pssm_file in pssm_files:
            pssm, intercept = read_pssm_file(pssm_file)
            self._pssm[len(pssm.columns)] = pssm
            self._intercept[len(pssm.columns)] = intercept

    def predict(self, sequences, input_type='helm'):
        """
        Score the input polymers/peptides using the previously defined PSSM.
        
        Parameters
        ----------
        sequences : list of str
            Polymers/peptides to score.
        input_type : str, default : 'helm'
            Format of the input polymers/peptides, either FASTA or HELM.

        Returns
        -------
        ndarray of shape (n_sequences,)
            Scores of the input polymers/peptides.

        Raises
        ------
        AssertionError: If output format is not 'fasta' or 'helm'.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
        assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

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

        if self._score_cutoff is not None:
            scores[scores > self._energy_score] = 0.

        return scores

    def generate_random_peptides(self, n_peptides, peptide_lengths, low_score=None, high_score=None, monomer_symbols=None):
        """
        Generate random linear polymers of a certain lengths with score comprised
        between `low_score` and `high_score` (if defined).
        
        Parameters
        ----------
        n_peptides : int
            Number of polymers/peptides to generate.
        peptide_lengths : list of int
            Sizes of the polymer/peptide sequences to generate.
        low_score : int or float, default: None
            Lowest score allowed for the randomly generated polymers/peptides.
        high_score : int or float, default: None
            Highest score allowed for the randomly generated polymers/peptides.
        monomer_symbols : list of str, default : None
            Symbol (1 letter) of the monomers that are going to be used to 
            generate random peptides. Per default, only the 20 natural amino 
            acids will be used.

        Returns
        -------
        polymers : ndarray of shape (n_peptides, )
            Randomly generated polymers/peptides.
        scores : ndarray of shape (n_peptides, )
            Scores of the randomly generated polymers/peptides.

        """
        random_peptides = []
        random_peptide_scores = []

        if monomer_symbols is None:
            # If we do not provide any monomer symbols, the 20 canonical amino acids will be used
            monomer_symbols = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
                               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        else:
            monomer_symbols = monomer_symbols

        if low_score is None:
            low_score = -np.inf

        if high_score is None:
            high_score = np.inf

        if not isinstance(peptide_lengths, (list, tuple)):
            peptide_lengths = [peptide_lengths]

        while True:
            peptide_length = np.random.choice(peptide_lengths)
            p = ''.join(np.random.choice(monomer_symbols, peptide_length))
            s = self.predict([p], input_type='FASTA')[0]

            if low_score <= s <= high_score:
                random_peptides.append(p)
                random_peptide_scores.append(s)

            if len(random_peptides) == n_peptides:
                break

        random_peptides = [build_helm_string({'PEPTIDE1': p}, []) for p in random_peptides]

        sorted_index = np.argsort(random_peptide_scores)
        random_peptides = np.asarray(random_peptides)[sorted_index]
        random_peptide_scores = np.asarray(random_peptide_scores)[sorted_index]

        return random_peptides, random_peptide_scores
