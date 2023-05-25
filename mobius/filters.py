#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Filters
#

import re

import numpy as np

from .utils import parse_helm


class PeptideSelfAggregationFilter():

    def __init__(self):
        pass

    def apply(self, polymers, **kwargs):
        """
        Filter polymers that self-aggregate using rule-based approach _[1]:
        - No more than 3 consecutive [DRAIENC] amino acids
        - No more than 3 consecutive [WHGKTS] amino acids

        Parameters
        ----------
        polymers : list of str
            List of polymers in HELM format.

        Returns
        -------
        ndarray
            Numpy array of boolean values indicating which polymers 
            passed the filter.

        References
        ----------
        .. [1] Hughes, Z. E., Nguyen, M. A., Wang, J., Liu, Y., Swihart, 
           M. T., Poloczek, M., ... & Walsh, T. R. (2021). Tuning materials-binding 
           peptide sequences toward gold-and silver-binding selectivity with 
           bayesian optimization. ACS nano, 15(11), 18260-18269.

        """
        p = re.compile('[DRAIENC]{4,}|[WHGKTS]{4,}')
        passed = np.ones(shape=(len(polymers),), dtype=bool)
        NATAA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        for i, polymer in enumerate(polymers):
            sequences, connections, _, _ = parse_helm(polymer)

            if connections:
                msg_error = 'Self-aggregration filter only works for linear peptides.'
                raise RuntimeError(msg_error)
            
            for _, sequence in sequences.items():
                if not set(sequence).issubset(NATAA):
                    msg_error = 'Self-aggregration filter only works polymers containing the 20 natural amino acids.'
                    raise RuntimeError(msg_error)
        
                if p.search(''.join(sequence)):
                    passed[i] = False
                    break

        return passed


class PeptideSolubilityFilter():

    def __init__(self, hydrophobe_ratio=0.5, charged_per_amino_acids=5.0):
        self._hydrophobe_ratio = hydrophobe_ratio
        self._charged_per_amino_acids = charged_per_amino_acids

    def apply(self, polymers, **kwargs):
        """
        Filter polymers that are not soluble using rule-based approach _[1]:
        - Keep hydrophobic amino acids to a minimum (less than 50% of the sequence)
        - At least one charged amino acid (D, E, H, K, R) for every 5 amino acids

        Parameters
        ----------
        polymers : list of str
            List of polymers in HELM format.

        Returns
        -------
        ndarray
            Numpy array of boolean values indicating which polymers 
            passed the filter.

        References
        ----------
        .. [1] https://www.sigmaaldrich.com/CH/de/technical-documents/technical-article/protein-biology/protein-labeling-and-modification/peptide-stability

        """
        passed = np.ones(shape=(len(polymers),), dtype=bool)
        NATAA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        APOLAR = ['A', 'F', 'G', 'I', 'L', 'P', 'V', 'W']
        AA_CHARGED = ['D', 'E', 'H', 'K', 'R']

        for i, polymer in enumerate(polymers):
            sequences, connections, _, _ = parse_helm(polymer)

            if connections:
                msg_error = 'Self-aggregration filter only works for linear peptides.'
                raise RuntimeError(msg_error)
            
            for _, sequence in sequences.items():
                if not set(sequence).issubset(NATAA):
                    msg_error = 'Self-aggregration filter only works polymers containing the 20 natural amino acids.'
                    raise RuntimeError(msg_error)
            
                length = len(sequence)

                apolar_monomers = set(sequence).intersection(APOLAR)
                
                if len(apolar_monomers) / length >= self._hydrophobe_ratio:
                    passed[i] = False
                    break

                charged_monomers = set(sequence).intersection(AA_CHARGED)

                if length / len(charged_monomers) >= self._charged_per_amino_acids:
                    passed[i] = False
                    break
    
        return passed
