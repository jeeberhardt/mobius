#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Descriptors
#


import numpy as np

from ..utils import convert_HELM_to_FASTA


class SimplePolymerDescriptors:
    """
    A class for computing polymer descriptors.

    """

    def __init__(self, descriptors, input_type='helm'):
        """
         Constructs a new instance of the polymerDescriptors class.
        
        Parameters
        ----------
        descriptors: pandas DataFrame
            A pandas DataFrame containing the polymer descriptors 
            for the 20 natural amino acids. The first column of the DataFrame 
            should contain the one-letter codes for the amino acids. The 
            remaining columns should contain the values of the descriptors.
        input_type: str, default : 'helm'
            The format of the input polymers. Valid options are 'fasta' and 'helm'.

        Raises
        ------
        AssertionError
            If the `input_type` is not 'fasta' or 'helm'.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
        assert input_type.lower() in ['fasta', 'helm'], msg_error

        self._descriptors = descriptors
        self._input_type = input_type

    def transform(self, polymers):
        """
        Calculates the descriptors for the given polymers.

        Parameters
        ----------
        polymers : str or list of str
            The amino acid polymers to calculate descriptors for. 
            If a single polymer is provided as a string, it will
            be converted to a list of length 1.

        Returns
        -------
        descriptors : numpy.ndarray
            A 2D numpy array of descriptor values for each amino acid 
            in the given polymers. The shape of the array is 
            (n_polymers, polymer_length * n_descriptors), where 
            n_polymers is the number of polymers, polymer_length
            is the length of the longest polymer, and n_descriptors 
            is the number of descriptors.

        """
        transformed = []

        # The other input type is FASTA
        if self._input_type == 'helm':
            polymers = convert_HELM_to_FASTA(polymers)

        for simple_polymer in polymers:
            tmp = [self._descriptors[self._descriptors['AA1'] == m].values[0][2:] for m in simple_polymer]
            transformed.append(np.asarray(tmp).flatten())
        
        transformed = np.asarray(transformed).astype(float)

        return transformed
