#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Filters
#

import re

import numpy as np

from .utils import parse_helm


class PeptideSelfAggregationFilter():
    """
    Class for filtering peptides that might self-aggregate.

    """

    def __init__(self, **kwargs):
        """
        Initialize the PeptideSelfAggregationFilter.

        """
        pass

    def apply(self, polymers):
        """
        Filter polymers that self-aggregate using rule-based approach [1]_:

            #. No more than 3 consecutive [DRAIENC] amino acids
            #. No more than 3 consecutive [WHGKTS] amino acids

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

        for i, complex_polymer in enumerate(polymers):
            simple_polymers, connections, _, _ = parse_helm(complex_polymer)

            if connections:
                msg_error = 'Self-aggregration filter only works for linear peptides.'
                raise RuntimeError(msg_error)
            
            for _, simple_polymer in simple_polymers.items():
                if not set(simple_polymer).issubset(NATAA):
                    msg_error = 'Self-aggregration filter only works polymers containing the 20 natural amino acids.'
                    raise RuntimeError(msg_error)
        
                if p.search(''.join(simple_polymer)):
                    passed[i] = False
                    break

        return passed


class PeptideSolubilityFilter():
    """
    Class for filtering peptides that might not be soluble.

    """

    def __init__(self, hydrophobe_ratio=0.5, charged_per_amino_acids=5.0, **kwargs):
        """
        Initialize the PeptideSolubilityFilter.

        Parameters
        ----------
        hydrophobe_ratio : float, default: 0.5
            Maximum ratio of hydrophobic amino acids allowed in the sequence.
        charged_per_amino_acids : float, default: 5.0
            Minimum number of charged amino acids per amino acids.

        """
        self._hydrophobe_ratio = hydrophobe_ratio
        self._charged_per_amino_acids = charged_per_amino_acids

    def apply(self, polymers):
        """
        Filter polymers that are not soluble using rule-based approach [1]_:

            #. Keep hydrophobic amino acids [AFGILPVW] below 50% of the total sequence
            #. At least one charged amino acid [DEKR] for every 5 amino acids

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
        AA_CHARGED = ['D', 'E', 'K', 'R']

        for i, complex_polymer in enumerate(polymers):
            simple_polymers, connections, _, _ = parse_helm(complex_polymer)

            if connections:
                msg_error = 'Self-aggregration filter only works for linear peptides.'
                raise RuntimeError(msg_error)
            
            for _, simple_polymer in simple_polymers.items():
                if not set(simple_polymer).issubset(NATAA):
                    msg_error = 'Self-aggregration filter only works polymers containing the 20 natural amino acids.'
                    raise RuntimeError(msg_error)
            
                length = len(simple_polymer)

                apolar_monomers = set(simple_polymer).intersection(APOLAR)
                
                if len(apolar_monomers) / length >= self._hydrophobe_ratio:
                    passed[i] = False
                    break

                charged_monomers = set(simple_polymer).intersection(AA_CHARGED)

                if len(charged_monomers) == 0:
                    passed[i] = False
                    break
                elif length / len(charged_monomers) >= self._charged_per_amino_acids:
                    passed[i] = False
                    break
    
        return passed
