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

    def __init__(self, hydrophobe_ratio=0.5, charged_ratio=0.25, 
                 ignore_connections=False, ignore_non_natural=False, **kwargs):
        """
        Initialize the PeptideSolubilityFilter.

        Parameters
        ----------
        hydrophobe_ratio : float, default: 0.5
            Maximum ratio of hydrophobic amino acids allowed in the sequence. 
            List of residues considered as hydrophobic: 'M', 'F', 'I', 'L', 'V', 'W', 'Y'
        charged_ratio : float, default: 5.0
            Minimum ratio of charged amino acids in the sequence. List of 
            residues considered as charged: 'D', 'E', 'K', 'R', 'H'

        """
        self._hydrophobe_ratio = hydrophobe_ratio
        self._charged_ratio = charged_ratio
        self._ignore_connections = ignore_connections
        self._ignore_non_natural = ignore_non_natural

    def apply(self, polymers):
        """
        Filter polymers that are not soluble using rule-based approach [1]_:

            #. Keep hydrophobic amino acids [MFILVWY] below 50% of the total sequence
            #. Keep charged amino acids [DEKRH] above 25% of the total sequence

        Parameters
        ----------
        polymers : list of str
            List of polymers in HELM format.

        Returns
        -------
        ndarray
            Numpy array of boolean values indicating which polymers 
            passed the filter.

        """
        passed = np.ones(shape=(len(polymers),), dtype=bool)
        NATAA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        APOLAR = ['M', 'F', 'I', 'L', 'V', 'W', 'Y']
        AA_CHARGED = ['D', 'E', 'K', 'R', 'H']

        for i, complex_polymer in enumerate(polymers):
            simple_polymers, connections, _, _ = parse_helm(complex_polymer)

            if not self._ignore_connections:
                if connections:
                    msg_error = 'Self-aggregration filter only works for linear peptides.'
                    raise RuntimeError(msg_error)

            for _, simple_polymer in simple_polymers.items():
                if not self._ignore_non_natural:
                    if not set(simple_polymer).issubset(NATAA):
                        msg_error = 'Self-aggregration filter only works polymers containing the 20 natural amino acids.'
                        raise RuntimeError(msg_error)

                length = len(simple_polymer)

                apolar_monomers = set(simple_polymer).intersection(APOLAR)

                if len(apolar_monomers) / length >= self._hydrophobe_ratio:
                    passed[i] = False
                    break

                charged_monomers = set(simple_polymer).intersection(AA_CHARGED)

                if len(charged_monomers) / length <= self._charged_ratio:
                    passed[i] = False
                    break

        return passed


class AARepeat():
    """
    Class for filtering peptides that contain sequence of repeated amino acids.
    """

    def __init__(self, max_repeat=2, **kwargs):
        """
        Initialize the AARepeat filter.

        Parameters
        ----------
        max_repeat : int, default: 2
            Maximum number of repeated amino acids allowed in the sequence.

        """
        self._max_repeat = max_repeat
    
    def apply(self, polymers):
        """
        Filter polymers that contain sequence of repeated amino acids.

        Parameters
        ----------
        polymers : list of str
            List of polymers in HELM format.

        Returns
        -------
        ndarray
            Numpy array of boolean values indicating which polymers
            passed the filter.

        """
        # Will start matching after max_repeat + 1 consecutive amino acids
        # if max_repeat = 2, then will match 3 or more consecutive amino acids
        p = re.compile(r'(.)\1{%d,}' % (self._max_repeat))
        passed = np.ones(shape=(len(polymers),), dtype=bool)

        for i, complex_polymer in enumerate(polymers):
            simple_polymers, _, _, _ = parse_helm(complex_polymer)

            for _, simple_polymer in simple_polymers.items():
                if p.search(''.join(simple_polymer)):
                    passed[i] = False
                    break

        return passed

