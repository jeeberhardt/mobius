#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Emulator
#

from abc import ABC, abstractmethod

import numpy as np
import torch

from .transforms import Map4Fingerprint
from .kernels import TanimotoSimilarityKernel
from .utils import build_helm_string
from .utils import read_pssm_file
from .utils import convert_FASTA_to_HELM, convert_HELM_to_FASTA


class _Emulator(ABC):
    """Abstract class for defining an emulator"""

    @abstractmethod
    def score(self):
        raise NotImplementedError()


class FindMe(_Emulator):
    """
    The FindMe `Emulator` where the goal is to find the target polymer provided
    at the initialization. This is for benchmarking purpose only.

    """

    def __init__(self, target_polymer, input_type='helm', kernel=None, transform=None):
        """
        Initialize the FindMe `Emulator`.

        Parameters
        ----------
        target_polymer : str
            Polymer to find.
        input_type : str, default : 'helm'
            Format of the target polymer, either FASTA or HELM.
        kernel : gpytorch.kernels.Kernel, default : None
            The kernel function used to calculate distance between polymers.
            If not defined, the Tanimoto kernel will be used.
        transform : callable, default : None
            Function that transforms the input into data for the `kernel`.
            If not defined, the MAP4 fingerprint method will be used.

        Raises
        ------
        AssertionError: If output format is not 'fasta' or 'helm'.

        """
        if transform != 'precomputed':
            msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
            assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

            if input_type == 'fasta':
                target_polymer = convert_FASTA_to_HELM(target_polymer)

        self._target_polymer = target_polymer

        if kernel is None:
            self._kernel = TanimotoSimilarityKernel()
        else:
            self._kernel = kernel

        if transform is None:
            self._transform = Map4Fingerprint()
        elif transform == 'precomputed':
            self._transform = 'precomputed'
        else:
            self._transform = transform

        if self._transform != 'precomputed':
            target_polymer_transformed = self._transform.transform(target_polymer)
        else:
            target_polymer_transformed = self._target_polymer

        self._target_polymer_transformed = torch.from_numpy(np.asarray(target_polymer_transformed)).float()

    def score(self, polymers, input_type='helm'):
        """
        Score the input polymers according to the unknown target polymers 
        using the `kernel` and the `transform`.
        
        Parameters
        ----------
        polymers : list of str
            Polymers to score.
        input_type : str, default : 'helm'
            Format of the input polymers, either FASTA or HELM.

        Returns
        -------
        ndarray of shape (n_polymers,)
            Scores of the input polymers.

        Raises
        ------
        AssertionError: If output format is not 'fasta' or 'helm'.

        """
        # If the input polymers are precomputed, no need to check for the format
        if self._transform != 'precomputed':
            msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
            assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

            if input_type == 'fasta':
                polymers = convert_FASTA_to_HELM(polymers)

        if not isinstance(polymers, (list, tuple, np.ndarray)):
            polymers = [polymers]

        if self._transform != 'precomputed':
            polymers_transformed = self._transform.transform(polymers)

        polymers_transformed = torch.from_numpy(np.asarray(polymers_transformed)).float()

        d = self._kernel.forward(self._target_polymer_transformed, polymers_transformed)[0]
        d = d.detach().numpy()

        return d


class LinearPeptideEmulator(_Emulator):
    """
    The LinearPeptideEmulator `Emulator` where the goal is to minimize or maximize 
    the score based on a defined Position Specific Scoring Matrix (PSSM).

    """
    
    def __init__(self, pssm_files, input_type='helm'):
        """
        Initialize the LinearPeptideEmulator `Emulator`.
        
        Parameters
        ----------
        pssm_files : str
            Path of the PSSM file to read.
        input_type : str, default : 'helm'
            Format of the input peptides, either FASTA or HELM.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
        assert input_type.lower() in ['fasta', 'helm'], msg_error % input_type

        self._pssm = {}
        self._intercept = {}
        self._input_type = input_type.lower()

        if not isinstance(pssm_files, (list, tuple)):
            pssm_files = [pssm_files]

        # Read PSS Matrices
        for pssm_file in pssm_files:
            pssm, intercept = read_pssm_file(pssm_file)
            self._pssm[len(pssm.columns)] = pssm
            self._intercept[len(pssm.columns)] = intercept

    def score(self, peptides):
        """
        Score the input peptides using the previously defined PSSM.
        
        Parameters
        ----------
        peptides : list of str
            Peptides to score.

        Returns
        -------
        ndarray of shape (n_peptides,)
            Scores of the input peptides.

        Raises
        ------
        AssertionError: If output format is not 'fasta' or 'helm'.

        """
        # Score peptides using those PSSM
        scores = []

        if self._input_type == 'helm':
            peptides = convert_HELM_to_FASTA(peptides)

        for peptide in peptides:
            score = 0

            try:
                pssm = self._pssm[len(peptide)]
                intercept = self._intercept[len(peptide)]
            except:
                # We cannot score that peptide, so default score is 999
                score = 999
                scores.append(score)
                continue

            for i, aa in enumerate(peptide):
                score += pssm.loc[aa][i + 1]

            score += intercept
            scores.append(score)

        scores = np.asarray(scores)

        return scores

    def generate_random_peptides(self, n_peptides, peptide_lengths, low_score=None, high_score=None, monomer_symbols=None, output_type='helm'):
        """
        Generate random linear peptides of a certain lengths with score comprised
        between `low_score` and `high_score` (if defined).
        
        Parameters
        ----------
        n_peptides : int
            Number of peptides to generate.
        peptide_lengths : list of int
            Sizes of the peptides to generate.
        low_score : int or float, default: None
            Lowest score allowed for the randomly generated peptides.
        high_score : int or float, default: None
            Highest score allowed for the randomly generated peptides.
        monomer_symbols : list of str, default : None
            Symbol (1 letter) of the monomers that are going to be used to 
            generate random peptides. Per default, only the 20 natural amino 
            acids will be used.
        output_type : str, default : 'helm'
            Format of the output peptides, either FASTA or HELM.

        Returns
        -------
        peptides : ndarray of shape (n_peptides, )
            Randomly generated peptides in the desired format (FASTA or HELM).
        scores : ndarray of shape (n_peptides, )
            Scores of the randomly generated peptides.

        """
        msg_error = 'Format (%s) not handled as output. Please use FASTA or HELM format.'
        assert output_type.lower() in ['fasta', 'helm'], msg_error % output_type

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
            s = self.score([p])[0]

            if low_score <= s <= high_score:
                random_peptides.append(p)
                random_peptide_scores.append(s)

            if len(random_peptides) == n_peptides:
                break

        if output_type == 'helm':
            random_peptides = [build_helm_string({'PEPTIDE1': p}, []) for p in random_peptides]
        else:
            random_peptides = random_peptides

        sorted_index = np.argsort(random_peptide_scores)
        random_peptides = np.asarray(random_peptides)[sorted_index]
        random_peptide_scores = np.asarray(random_peptide_scores)[sorted_index]

        return random_peptides, random_peptide_scores
