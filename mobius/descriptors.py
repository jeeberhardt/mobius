#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Descriptors
#


import numpy as np

from .utils import convert_HELM_to_FASTA


class SequenceDescriptors:
    """
    A class for computing sequence descriptors.

    """

    def __init__(self, descriptors, input_type='helm'):
        """
         Constructs a new instance of the SequenceDescriptors class.
        
        Parameters
        ----------
        descriptors: pandas DataFrame
            A pandas DataFrame containing the sequence descriptors 
            for the 20 natural amino acids. The first column of the DataFrame 
            should contain the one-letter codes for the amino acids. The 
            remaining columns should contain the values of the descriptors.
        input_type: str, default : 'helm'
            The format of the input sequences. Valid options are 'fasta' and 'helm'.

        Raises
        ------
        AssertionError
            If the `input_type` is not 'fasta' or 'helm'.

        """
        msg_error = 'Format (%s) not handled. Please use FASTA or HELM format.'
        assert input_type.lower() in ['fasta', 'helm'], msg_error

        self._descriptors = descriptors
        self._input_type = input_type

    def transform(self, sequences):
        """
        Calculates the descriptors for the given sequences.

        Parameters
        ----------
        sequences : str or list of str
            The amino acid sequences to calculate descriptors for. 
            If a single sequence is provided as a string, it will
            be converted to a list of length 1.

        Returns
        -------
        descriptors : numpy.ndarray
            A 2D numpy array of descriptor values for each amino acid 
            in the given sequences. The shape of the array is 
            (n_sequences, sequence_length * n_descriptors), where 
            n_sequences is the number of sequences, sequence_length
            is the length of the longest sequence, and n_descriptors 
            is the number of descriptors.

        """
        transformed = []

        # The other input type is FASTA
        if self._input_type == 'helm':
            sequences = convert_HELM_to_FASTA(sequences)

        for sequence in sequences:
            tmp = [self._descriptors[self._descriptors['AA1'] == aa].values[0][2:] for aa in sequence]
            transformed.append(tmp)

        return np.asarray(transformed)
